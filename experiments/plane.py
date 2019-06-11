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
parser.add_argument('--dataset_name', type=str, default='diamond',
                    help='Name of dataset to use.')
parser.add_argument('--n_data_points', default=int(1e6),
                    help='Number of unique data points in training set.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Size of batch used for training.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers used in data loaders.')

# model
parser.add_argument('--num_bins', type=int, default=64,
                    help='Number of bins to use ')

# optimization
parser.add_argument('--learning_rate', default=5e-4,
                    help='Learning rate for Adam.')
parser.add_argument('--num_training_steps', default=int(5e5),
                    help='Number of total training steps.')
parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
                    help='Value by which to clip grad norm.')

# logging and checkpoints
parser.add_argument('--monitor_interval', default=1,
                    help='Interval in steps at which to report training stats.')
parser.add_argument('--visualize_interval', default=10000,
                    help='Interval in steps at which to report training stats.')
parser.add_argument('--save_interval', default=10000,
                    help='Interval in steps at which to save model.')

# reproducibility
parser.add_argument('--seed', default=1638128,
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
train_dataset = data_.load_plane_dataset(args.dataset_name, args.n_data_points)
train_loader = data_.InfiniteLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)

# Generate test grid data
num_points_per_axis = 512
limit = 4
bounds = np.array([
    [-limit, limit],
    [-limit, limit]
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
dim = 2

# create model
distribution = distributions.StandardNormal((2,))

args.base_transform_type = 'affine'


def create_base_transform(i):
    if args.base_transform_type == 'affine':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=dim, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=32,
                num_blocks=2,
                use_batch_norm=True
            )
        )
    else:
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=dim, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=32,
                num_blocks=2,
                use_batch_norm=True
            ),
            tails='linear',
            tail_bound=5,
            num_bins=args.num_bins,
            apply_unconditional_transform=False
        )


transform = transforms.CompositeTransform([
    create_base_transform(i) for i in range(2)
])

flow = flows.Flow(transform, distribution).to(device)

n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)

# create summary writer and write to log directory
timestamp = cutils.get_timestamp()
log_dir = os.path.join(cutils.get_log_root(), args.dataset_name, timestamp)
writer = SummaryWriter(log_dir=log_dir)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)

tbar = tqdm(range(args.num_training_steps))
for step in tbar:
    flow.train()
    optimizer.zero_grad()
    scheduler.step(step)

    batch = next(train_loader).to(device)
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    if (step + 1) % args.monitor_interval == 0:
        s = 'Loss: {:.4f}'.format(loss.item())
        tbar.set_description(s)

        summaries = {
            'loss': loss.detach()
        }
        for summary, value in summaries.items():
            writer.add_scalar(tag=summary, scalar_value=value, global_step=step)

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
            samples = utils.tensor2numpy(flow.sample(int(1e6), batch_size=int(1e5)))
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
