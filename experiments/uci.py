import argparse
import json
import numpy as np
import torch
import os

from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

import data as data_
import nn as nn_
import utils

from experiments import cutils
from nde import distributions, flows, transforms

parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset_name', type=str, default='miniboone',
                    choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'],
                    help='Name of dataset to use.')
parser.add_argument('--train_batch_size', type=int, default=64,
                    help='Size of batch used for training.')
parser.add_argument('--val_frac', type=float, default=1.,
                    help='Fraction of validation set to use.')
parser.add_argument('--val_batch_size', type=int, default=512,
                    help='Size of batch used for validation.')

# optimization
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for optimizer.')
parser.add_argument('--num_training_steps', type=int, default=200000,
                    help='Number of total training steps.')
parser.add_argument('--anneal_learning_rate', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to anneal the learning rate.')
parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
                    help='Value by which to clip norm of gradients.')

# flow details
parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
                    choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
                             'affine-autoregressive', 'quadratic-autoregressive',
                             'rq-autoregressive'],
                    help='Type of transform to use between linear layers.')
parser.add_argument('--linear_transform_type', type=str, default='lu',
                    choices=['permutation', 'lu', 'svd'],
                    help='Type of linear transform to use.')
parser.add_argument('--num_flow_steps', type=int, default=10,
                    help='Number of blocks to use in flow.')
parser.add_argument('--hidden_features', type=int, default=256,
                    help='Number of hidden features to use in coupling/autoregressive nets.')
parser.add_argument('--tail_bound', type=float, default=3,
                    help='Box is on [-bound, bound]^2')
parser.add_argument('--num_bins', type=int, default=8,
                    help='Number of bins to use for piecewise transforms.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks to use in coupling/autoregressive nets.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in coupling/autoregressive nets.')
parser.add_argument('--dropout_probability', type=float, default=0.25,
                    help='Dropout probability for coupling/autoregressive nets.')
parser.add_argument('--apply_unconditional_transform', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to unconditionally transform \'identity\' '
                         'features in coupling layer.')

# logging and checkpoints
parser.add_argument('--monitor_interval', type=int, default=250,
                    help='Interval in steps at which to report training stats.')

# reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# create data
train_dataset = data_.load_dataset(args.dataset_name, split='train')
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    drop_last=True
)
train_generator = data_.batch_generator(train_loader)
test_batch = next(iter(train_loader)).to(device)

# validation set
val_dataset = data_.load_dataset(args.dataset_name, split='val', frac=args.val_frac)
val_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.val_batch_size,
    shuffle=True,
    drop_last=True
)

# test set
test_dataset = data_.load_dataset(args.dataset_name, split='test')
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    drop_last=False
)

features = train_dataset.dim


def create_linear_transform():
    if args.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif args.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif args.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_base_transform(i):
    if args.base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        )
    elif args.base_transform_type == 'quadratic-coupling':
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    else:
        raise ValueError


def create_transform():
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(),
            create_base_transform(i)
        ]) for i in range(args.num_flow_steps)
    ] + [
        create_linear_transform()
    ])
    return transform


# create model
distribution = distributions.StandardNormal((features,))
transform = create_transform()
flow = flows.Flow(transform, distribution).to(device)

n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
if args.anneal_learning_rate:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
else:
    scheduler = None

# create summary writer and write to log directory
timestamp = cutils.get_timestamp()
if cutils.on_cluster():
    timestamp += '||{}'.format(os.environ['SLURM_JOB_ID'])
log_dir = os.path.join(cutils.get_log_root(), args.dataset_name, timestamp)
while True:
    try:
        writer = SummaryWriter(log_dir=log_dir, max_queue=20)
        break
    except FileExistsError:
        sleep(5)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)

tbar = tqdm(range(args.num_training_steps))
best_val_score = -1e10
for step in tbar:
    flow.train()
    if args.anneal_learning_rate:
        scheduler.step(step)
    optimizer.zero_grad()

    batch = next(train_generator).to(device)
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)

    if (step + 1) % args.monitor_interval == 0:
        flow.eval()

        with torch.no_grad():
            # compute validation score
            running_val_log_density = 0
            for val_batch in val_loader:
                log_density_val = flow.log_prob(val_batch.to(device).detach())
                mean_log_density_val = torch.mean(log_density_val).detach()
                running_val_log_density += mean_log_density_val
            running_val_log_density /= len(val_loader)

        if running_val_log_density > best_val_score:
            best_val_score = running_val_log_density
            path = os.path.join(cutils.get_checkpoint_root(),
                                '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
            torch.save(flow.state_dict(), path)

        # compute reconstruction
        with torch.no_grad():
            test_batch_noise = flow.transform_to_noise(test_batch)
            test_batch_reconstructed, _ = flow._transform.inverse(test_batch_noise)
        errors = test_batch - test_batch_reconstructed
        max_abs_relative_error = torch.abs(errors / test_batch).max()
        average_abs_relative_error = torch.abs(errors / test_batch).mean()
        writer.add_scalar('max-abs-relative-error',
                          max_abs_relative_error, global_step=step)
        writer.add_scalar('average-abs-relative-error',
                          average_abs_relative_error, global_step=step)

        summaries = {
            'val': running_val_log_density.item(),
            'best-val': best_val_score.item(),
            'max-abs-relative-error': max_abs_relative_error.item(),
            'average-abs-relative-error': average_abs_relative_error.item()
        }
        for summary, value in summaries.items():
            writer.add_scalar(tag=summary, scalar_value=value, global_step=step)

# load best val model
path = os.path.join(cutils.get_checkpoint_root(),
                    '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
flow.load_state_dict(torch.load(path))
flow.eval()

# calculate log-likelihood on test set
with torch.no_grad():
    log_likelihood = torch.Tensor([])
    for batch in tqdm(test_loader):
        log_density = flow.log_prob(batch.to(device))
        log_likelihood = torch.cat([
            log_likelihood,
            log_density
        ])
path = os.path.join(log_dir, '{}-{}-log-likelihood.npy'.format(
    args.dataset_name,
    args.base_transform_type
))
np.save(path, utils.tensor2numpy(log_likelihood))
mean_log_likelihood = log_likelihood.mean()
std_log_likelihood = log_likelihood.std()

# save log-likelihood
s = 'Final score for {}: {:.2f} +- {:.2f}'.format(
    args.dataset_name.capitalize(),
    mean_log_likelihood.item(),
    2 * std_log_likelihood.item() / np.sqrt(len(test_dataset))
)
print(s)
filename = os.path.join(log_dir, 'test-results.txt')
with open(filename, 'w') as file:
    file.write(s)
