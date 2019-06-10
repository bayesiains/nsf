import argparse
import json
import numpy as np
import os
import torch

import data as data_
import utils

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils import data
from torchvision import datasets as tvdatasets, transforms as tvtransforms
from torchvision.utils import make_grid
from tqdm import tqdm

from conor import cutils
from nde import distributions, flows, transforms

parser = argparse.ArgumentParser()

# CUDA
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU.')

# data
parser.add_argument('--dataset', type=str, default='imagenet-32',
                    help='Which dataset to use.')
parser.add_argument('--resize_shape', type=int, default=256,
                    help='CelebA resize.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training.')
parser.add_argument('--bits', type=int, default=5,
                    help='Number of bits to use for CIFAR-10\CelebA')
parser.add_argument('--checkpoint', action='store_true',
                    help='Whether to checkpoint the model.')
parser.add_argument('--use_glow_preprocessing', type=int, default=False,
                    help='Whether to use Glow preprocessing.')

# model
parser.add_argument('--levels', type=int, default=3,
                    help='Number of levels to use in multiscale architecture.')
parser.add_argument('--steps_per_level', type=int, default=16,
                    help='Number of flow steps to use per level.')
parser.add_argument('--hidden_channels', type=int, default=512,
                    help='Number of hidden channels for coupling transform.')
parser.add_argument('--num_bins', type=int, default=8,
                    help='Number of bins to use for piecewise transforms.')

# optimization
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate for optimizer.')
parser.add_argument('--num_training_steps', type=int, default=int(3e5),
                    help='Number of total training steps.')

# logging and checkpoints
parser.add_argument('--sample_interval', type=int, default=100,
                    help='Interval in steps at which to sample from the model.')
parser.add_argument('--validation_interval', type=int, default=10000,
                    help='Interval in steps at which to compute bits per dim on validation.')
parser.add_argument('--save_interval', type=int, default=1000,
                    help='Interval in steps at which to save the model.')

# reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_gpu and not torch.cuda.is_available():
    raise RuntimeError('use_gpu is True but CUDA is not available')
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')

if args.dataset in ['fashion-mnist', 'mnist']:
    if args.bits != 8:
        print('Only 8 bit Fashion-MNIST and MNIST supported. Using 8 bit images.')
    BITS = 8
else:
    BITS = args.bits

#####################################################
#                   CHECKPOINTING                   #
#####################################################
CHECKPOINT = False
args.checkpoint = CHECKPOINT
print('Checkpoint? {}'.format(args.checkpoint))

#####################################################


def conv2d(in_channels, out_channels, kernel_size):
    same_padding = kernel_size // 2 # Padding that would keep the spatial dims the same
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=same_padding)


def create_transform_step(num_channels, hidden_channels, num_bins):
    def create_convnet(in_channels, out_channels):
        return nn.Sequential(
            conv2d(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            conv2d(hidden_channels, out_channels, kernel_size=3),
        )

    mask = utils.create_mid_split_binary_mask(num_channels)

    return transforms.CompositeTransform([
        transforms.OneByOneConvolution(num_channels),
        transforms.Sigmoid(),
        transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            num_bins=num_bins,
            apply_unconditional_transform=False
        ),
        transforms.Logit()
    ])


def create_transform(channels, height, width,
                     levels, steps_per_level, alpha):

    multiscale_transform = transforms.MultiscaleCompositeTransform(
        num_transforms=levels
    )

    for level in range(levels):
        squeeze_transform = transforms.SqueezeTransform()
        channels, height, width = squeeze_transform.get_output_shape(
            channels,
            height,
            width
        )

        transform_level = transforms.CompositeTransform(
            [squeeze_transform] +
            [create_transform_step(channels, args.hidden_channels, args.num_bins)
             for _ in range(steps_per_level)]
        )

        new_shape = multiscale_transform.add_transform(
            transform_level,
            (channels, height, width)
        )
        if new_shape: # If not last layer
            channels, height, width = new_shape

    if args.use_glow_preprocessing:
        preprocess = transforms.AffineScalarTransform(scale=(1 / 2 ** BITS), shift=-0.5)
    else:
        preprocess = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1 - alpha) / 2 ** BITS, shift=alpha),
            transforms.Logit()
        ])
    transform = transforms.CompositeTransform([
        # Map into unconstrained space
        preprocess,
        # Multiscale transform
        multiscale_transform
    ])

    return transform


def jitter(inputs):
    return inputs + torch.rand_like(inputs) # inputs are in [0, 2 ** BITS]


class Dequantize:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, inputs): # Inputs in [0, 1]
        outputs = torch.floor(inputs * (self.num_bins - 1)) # -> [0, self.num_bins - 1]
        outputs += torch.rand_like(inputs) # -> [0, self.num_bins]
        return outputs

    def inverse(self, inputs): # Inputs in [0, self.num_bins]
        outputs = inputs / self.num_bins # -> [0, 1]
        return outputs


def get_data(dataset, train=True, pad=0):
    if dataset == 'fashion-mnist':
        train_transform = tvtransforms.Compose([
            tvtransforms.Pad((pad, pad)),
            tvtransforms.RandomHorizontalFlip(),
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(jitter)
        ])
        val_transform = tvtransforms.Compose([
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255)
        ])
        assert pad == 2
        c, h, w = (1, 28 + 2 * pad, 28 + 2 * pad)
        train_dataset = tvdatasets.FashionMNIST(os.path.join(utils.get_data_root(), 'fashion-mnist'),
                                                train=train, download=True,
                                                transform=train_transform if train else val_transform)
    elif dataset == 'mnist':
        train_transform = tvtransforms.Compose([
            tvtransforms.Pad((pad, pad)),
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(jitter)
        ])
        val_transform = tvtransforms.Compose([
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255)
        ])
        assert pad == 2
        c, h, w = (1, 28 + 2 * pad, 28 + 2 * pad)
        train_dataset = tvdatasets.MNIST(os.path.join(utils.get_data_root(), 'mnist'),
                                         train=train, download=True,
                                         transform=train_transform if train else val_transform)
    elif dataset == 'cifar-10':
        train_transform = tvtransforms.Compose([tvtransforms.RandomHorizontalFlip(),
                                                tvtransforms.ToTensor(),
                                                tvtransforms.Lambda(lambda x: x * 255),
                                                tvtransforms.Lambda(
                                                    lambda x: torch.floor(x / 2 ** (8 - BITS))),
                                                tvtransforms.Lambda(jitter)])
        val_transform = tvtransforms.Compose([
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(
                lambda x: torch.floor(x / 2 ** (8 - BITS)))
        ])
        c, h, w = 3, 32, 32
        train_dataset = tvdatasets.CIFAR10(os.path.join(utils.get_data_root(), 'cifar-10'),
                                           train=train, download=True,
                                           transform=train_transform if train else val_transform)
    elif dataset == 'imagenet-32':
        train_transform = tvtransforms.Compose([tvtransforms.ToTensor(),
                                                tvtransforms.Lambda(lambda x: x * 255),
                                                tvtransforms.Lambda(
                                                    lambda x: torch.floor(
                                                        x / 2 ** (8 - BITS))),
                                                tvtransforms.Lambda(jitter)])
        val_transform = tvtransforms.Compose([
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(
                lambda x: torch.floor(x / 2 ** (8 - BITS)))
        ])
        c, h, w = 3, 32, 32
        train_dataset = data_.ImageNet32(root=cutils.get_imagenet32_root(), train=train,
                                         download=True, transform=train_transform if train else val_transform)
    elif dataset == 'imagenet-64':
        pass
    elif dataset == 'celeba-hq':
        train_transform = tvtransforms.Compose([
            tvtransforms.Resize(args.resize_shape),
            tvtransforms.RandomHorizontalFlip(),
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(lambda x: torch.floor(x / 2 ** (8 - BITS))),
            tvtransforms.Lambda(jitter)
        ])
        val_transform = tvtransforms.Compose([
            tvtransforms.Resize(args.resize_shape),
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(lambda x: x * 255),
            tvtransforms.Lambda(
                lambda x: torch.floor(x / 2 ** (8 - BITS)))
        ])
        c, h, w = 3, args.resize_shape, args.resize_shape
        train_dataset = data_.CelebAHQ(cutils.get_celeba_root(), train=train, download=True,
                                       transform=train_transform if train else val_transform)
    else:
        raise RuntimeError('Unknown dataset')
    return train_dataset, (c, h, w)


def gridimshow(image, ax):
    if image.shape[0] == 1:
        image = utils.tensor2numpy(image[0, ...])
        ax.imshow(1 - image, cmap='Greys')
    else:
        image = utils.tensor2numpy(image.permute(1, 2, 0))
        ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')


def plot_data(dataset):
    if 'mnist' in dataset:
        pad = 2
    else:
        pad = 0
    train_dataset, (channels, height, width) = get_data(dataset, pad=pad)

    samples = torch.cat([train_dataset[i][0] for i in np.random.randint(0, len(train_dataset), 64)])
    samples /= 2 ** BITS

    fig, ax = plt.subplots(figsize=(10, 10))
    gridimshow(make_grid(samples.view(64, channels, height, width), nrow=8), ax)
    plt.show()
    # fig.savefig(os.path.join(cutils.get_output_root(), 'data.png'))


def run():
    if 'mnist' in args.dataset:
        pad = 2
    else:
        pad = 0
    train_dataset, (channels, height, width) = get_data(args.dataset, train=True, pad=pad)
    train_loader = data_.InfiniteLoader(train_dataset, batch_size=args.batch_size)
    val_dataset, _ = get_data(args.dataset, train=False, pad=pad)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 drop_last=True)

    distribution = distributions.StandardNormal((channels * height * width,))
    transform = create_transform(
        channels=channels,
        height=height,
        width=width,
        levels=args.levels,
        steps_per_level=args.steps_per_level,
        alpha=0.05
    )
    flow = flows.Flow(transform, distribution).to(device)

    print('There are {} parameters in this model.'.format(utils.get_num_parameters(flow)))
    # quit()

    optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)

    # create summary writer and write to log directory
    timestamp = cutils.get_timestamp()
    log_dir = os.path.join(cutils.get_log_root(), args.dataset, timestamp)
    writer = SummaryWriter(log_dir=log_dir, max_queue=20)
    filename = os.path.join(log_dir, 'config.json')
    with open(filename, 'w') as file:
        json.dump(vars(args), file)

    tbar = tqdm(range(args.num_training_steps))
    for step in tbar:
        flow.train()
        optimizer.zero_grad()

        batch, _ = next(train_loader)
        if args.checkpoint:
            batch.requires_grad = True
        _, log_density = flow(batch.to(device))
        loss = - torch.mean(log_density)
        loss.backward()

        optimizer.step()

        writer.add_scalar('loss-{}-bits'.format(BITS), loss.item(), global_step=step)

        if (step + 1) % args.sample_interval == 0:
            flow.eval()

            temperatures = [0.5, 0.75, 1]

            figure, axes = plt.subplots(1, len(temperatures), figsize=(5 * len(temperatures), 5))
            with torch.no_grad():
                for i, temperature in enumerate(temperatures):
                    num_samples = 36
                    noise = flow._distribution.sample(num_samples) * temperature
                    samples, _ = flow._transform.inverse(noise.detach())
                    samples = samples.detach()
                    grid = make_grid(
                        samples.view(num_samples, channels, height, width),
                        nrow=int(num_samples**0.5)
                    )
                    if args.use_glow_preprocessing:
                        grid = torch.floor(grid) * (2 ** (8 - BITS))
                        grid = torch.clamp(grid, 0, 2 ** BITS - 1)
                    else:
                        grid = grid.clamp(0, 2 ** BITS)
                        grid /= 2 ** BITS
                    gridimshow(
                        image=grid,
                        ax=axes[i]
                    )

                    axes[i].set_title('T = {}'.format(temperature))

                    writer.add_figure(tag='samples-{}-bits'.format(BITS), figure=figure, global_step=step)

                    plt.close(figure)

        if (step + 1) % args.validation_interval == 0:
            flow.eval()

            with torch.no_grad():
                val_log_density = 0
                for batch, _ in val_loader:
                    _, log_density = flow(batch.to(device))
                    val_log_density += (- torch.mean(log_density)).item() / (channels * height * width)
                val_log_density /= len(val_loader)
                val_bits_per_dim = val_log_density / np.log(2)
                writer.add_scalar('bits-per-dim-{}-bits'.format(BITS), val_bits_per_dim, global_step=step)

        if (step + 1) * args.save_interval == 0:
            path = os.path.join(cutils.get_checkpoint_root(),
                                '{}-{}'.format(args.dataset, timestamp))
            torch.save(flow.state_dict(), path)

        path = os.path.join(cutils.get_checkpoint_root(),
                            '{}-{}'.format(args.dataset, timestamp))
        torch.save(flow.state_dict(), path)


def main():
    # plot_data(args.dataset)
    run()


if __name__ == '__main__':
    main()
