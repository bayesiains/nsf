import argparse
import json
import numpy as np
import torch
import os

from matplotlib import cm, pyplot as plt
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

import data as data_
import nn as nn_
import utils

from experiments import cutils
from nde import distributions, flows, transforms

parser = argparse.ArgumentParser()

# CUDA
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU.')

# data
parser.add_argument('--dataset_name', type=str, default='shannon',
                    help='Name of dataset to use.')
parser.add_argument('--n_data_points', type=int, default=int(1e6),
                    help='Number of unique data points in training set.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Size of batch used for training.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers used in data loaders.')

# model
parser.add_argument('--base_transform_type', type=str, default='rq',
                    help='Which base transform to use.')
parser.add_argument('--hidden_features', type=int, default=64,
                    help='Number of hidden features in coupling layers.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks in coupling layer transform.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in coupling layer transform.')
parser.add_argument('--dropout_probability', type=float, default=0.0,
                    help='Dropout probability for coupling transform.')
parser.add_argument('--num_bins', type=int, default=128,
                    help='Number of bins in piecewise cubic coupling transform.')
parser.add_argument('--apply_unconditional_transform', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to apply unconditional transform in coupling layer.')
parser.add_argument('--min_bin_width', type=float, default=1e-3,
                    help='Minimum bin width for piecewise transforms.')
parser.add_argument('--num_flow_steps', type=int, default=2,
                    help='Number of steps of flow.')

# optimization
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='Learning rate for Adam.')
parser.add_argument('--n_total_steps', type=int, default=int(1.5e6),
                    help='Number of total training steps.')
parser.add_argument('--grad_norm_clip_value', type=float, default=5,
                    help='Value by which to clip gradient norm.')

# logging and checkpoints
parser.add_argument('--visualize_interval', type=int, default=10000,
                    help='Interval in steps at which to report training stats.')
parser.add_argument('--save_interval', type=int, default=10000,
                    help='Interval in steps at which to save model.')

# reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_gpu:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')

# create data
train_dataset = data_.load_face_dataset(
    name=args.dataset_name,
    num_points=args.n_data_points
)
train_loader = data_.InfiniteLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)
dim = 2

# Generate test grid data
num_points_per_axis = 512
bounds = np.array([
    [1e-3, 1 - 1e-3],
    [1e-3, 1 - 1e-3]
])
grid_dataset = data_.TestGridDataset(
    num_points_per_axis=num_points_per_axis,
    bounds=bounds
)
grid_loader = data.DataLoader(
    dataset=grid_dataset,
    batch_size=1000,
    drop_last=False
)

# create model
distribution = distributions.TweakedUniform(
    low=torch.zeros(dim),
    high=torch.ones(dim)
)


def create_base_transform(i):
    if args.base_transform_type == 'rq':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                features=dim,
                even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features:
            nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                num_blocks=args.num_transform_blocks,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            apply_unconditional_transform=False,
        )
    elif args.base_transform_type == 'affine':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                features=dim,
                even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features:
            nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                num_blocks=args.num_transform_blocks,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        )
    else:
        raise ValueError


transform = transforms.CompositeTransform([
    create_base_transform(i) for i in range(args.num_flow_steps)
])

flow = flows.Flow(transform, distribution).to(device)

n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_total_steps)

# create summary writer and write to log directory
timestamp = cutils.get_timestamp()
log_dir = os.path.join(cutils.get_log_root(), args.dataset_name, timestamp)
writer = SummaryWriter(log_dir=log_dir)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)

tbar = tqdm(range(args.n_total_steps))
for step in tbar:
    flow.train()
    scheduler.step(step)
    optimizer.zero_grad()

    batch = next(train_loader).to(device)
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)

    if (step + 1) % args.visualize_interval == 0:
        flow.eval()
        log_density_np = []
        for batch in grid_loader:
            batch = batch.to(device)
            log_density = flow.log_prob(batch)
            log_density_np = np.concatenate(
                (log_density_np, utils.tensor2numpy(log_density))
            )

        figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharex=True, sharey=True)

        cmap = cm.magma
        axes[0].hist2d(utils.tensor2numpy(train_dataset.data[:, 0]),
                       utils.tensor2numpy(train_dataset.data[:, 1]),
                       range=bounds, bins=512, cmap=cmap, rasterized=False)
        axes[0].set_xlim(bounds[0])
        axes[0].set_ylim(bounds[1])
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].pcolormesh(grid_dataset.X, grid_dataset.Y,
                           np.exp(log_density_np).reshape(grid_dataset.X.shape),
                           cmap=cmap)
        axes[1].set_xlim(bounds[0])
        axes[1].set_ylim(bounds[1])
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        with torch.no_grad():
            samples = utils.tensor2numpy(
                flow.sample(num_samples=int(1e6), batch_size=int(1e5)))
        axes[2].hist2d(samples[:, 0], samples[:, 1],
                       range=bounds, bins=512, cmap=cmap, rasterized=False)
        axes[2].set_xlim(bounds[0])
        axes[2].set_ylim(bounds[1])
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()

        path = os.path.join(cutils.get_output_root(), '{}.png'.format(args.dataset_name))
        plt.savefig(path, dpi=300)
        writer.add_figure(tag='viz', figure=figure, global_step=step)
        plt.close()

    if (step + 1) % args.save_interval == 0:
        path = os.path.join(cutils.get_checkpoint_root(),
                            '{}.t'.format(args.dataset_name))
        torch.save(flow.state_dict(), path)

path = os.path.join(cutils.get_checkpoint_root(),
                    '{}-{}.t'.format(args.dataset_name, timestamp))
torch.save(flow.state_dict(), path)
