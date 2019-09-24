import argparse
import json
import numpy as np
import os
import torch

import data as data_
import nn as nn_
import utils
import vae

from functools import partial
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from time import sleep
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms as tvtransforms
from torchvision.transforms import functional as tvF
from torchvision.utils import make_grid
from tqdm import tqdm

from experiments import cutils
from nde import distributions as distributions_, flows, transforms

parser = argparse.ArgumentParser()

parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs with different seeds to do.')
# data
parser.add_argument('--dataset_name', type=str, default='emnist',
                    choices=['mnist', 'emnist', 'omniglot', 'fashion-mnist'],
                    help='Which dataset to use.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training.')

# optimization
parser.add_argument('--num_training_steps', type=int, default=150000,
                    help='Total number of training steps.')
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--kl_multiplier_schedule', type=str, default='linear',
                    choices=['constant', 'linear'],
                    help='Kind of schedule to use for KL multiplier.')
parser.add_argument('--kl_warmup_fraction', type=float, default=0.1,
                    help='Length of training to do KL warmup.')
parser.add_argument('--kl_multiplier_initial', type=float, default=0.5,
                    help='Initial value for KL multiplier.')
parser.add_argument('--kl_multiplier_max', type=float, default=1,
                    help='Max value for KL multiplier.')

# features size
parser.add_argument('--latent_features', type=int, default=32,
                    help='Number of latent features to use.')
parser.add_argument('--context_features', type=int, default=64,
                    help='Number of context features computed by encoder.')

# prior choices
parser.add_argument('--prior_type', type=str, default='rq-autoregressive',
                    choices=['standard-normal', 'affine-coupling',
                             'rq-coupling', 'affine-autoregressive',
                             'rq-autoregressive'],
                    help='Which prior to use.')

# approximate posterior
parser.add_argument('--approximate_posterior_type', type=str, default='rq-autoregressive',
                    choices=['diagonal-normal', 'affine-coupling',
                             'rq-coupling', 'affine-autoregressive',
                             'rq-autoregressive'],
                    help='Which approximate posterior to use.')

# flow details
parser.add_argument('--num_flow_steps', type=int, default=5,
                    help='Number of flow steps to use')
parser.add_argument('--hidden_features', type=int, default=128,
                    help='Number of hidden features for transforms')
parser.add_argument('--num_bins', type=int, default=8,
                    help='Number of bins for NSF transforms.')
parser.add_argument('--tail_bound', type=float, default=3,
                    help='NSF box bound.')
parser.add_argument('--apply_unconditional_transform', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to apply unconditional_transform for coupling NSF.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks to use in base transforms.')
parser.add_argument('--dropout_probability', type=float, default=0.0,
                    help='Dropout prob. for base transforms.')
parser.add_argument('--dropout_probability_encoder_decoder', type=float, default=0.0,
                    help='Dropout prob. for encoder_decoder.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in the base transforms.')
parser.add_argument('--linear_type', type=str, default='lu',
                    choices=['lu', 'svd', 'perm'],
                    help='Which linear transformation to use.')

# monitoring
parser.add_argument('--monitor_interval', type=int, default=100,
                    help='How often to compute validation score and plot samples.')

args = parser.parse_args()

if args.dataset_name == 'omniglot':
    args.dropout_probability_encoder_decoder = 0.2


def run(seed):

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create training data.
    data_transform = tvtransforms.Compose([
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(torch.bernoulli)
    ])

    if args.dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=os.path.join(utils.get_data_root(), 'mnist'),
            train=True,
            download=True,
            transform=data_transform
        )
        test_dataset = datasets.MNIST(
            root=os.path.join(utils.get_data_root(), 'mnist'),
            train=False,
            download=True,
            transform=data_transform
        )
    elif args.dataset_name == 'fashion-mnist':
        dataset = datasets.FashionMNIST(
            root=os.path.join(utils.get_data_root(), 'fashion-mnist'),
            train=True,
            download=True,
            transform=data_transform
        )
        test_dataset = datasets.FashionMNIST(
            root=os.path.join(utils.get_data_root(), 'fashion-mnist'),
            train=False,
            download=True,
            transform=data_transform
        )
    elif args.dataset_name == 'omniglot':
        dataset = data_.OmniglotDataset(
            split='train',
            transform=data_transform
        )
        test_dataset = data_.OmniglotDataset(
            split='test',
            transform=data_transform
        )
    elif args.dataset_name == 'emnist':
        rotate = partial(tvF.rotate, angle=-90)
        hflip = tvF.hflip
        data_transform = tvtransforms.Compose([
            tvtransforms.Lambda(rotate),
            tvtransforms.Lambda(hflip),
            tvtransforms.ToTensor(),
            tvtransforms.Lambda(torch.bernoulli)
        ])
        dataset = datasets.EMNIST(
            root=os.path.join(utils.get_data_root(), 'emnist'),
            split='letters',
            train=True,
            transform=data_transform,
            download=True
        )
        test_dataset = datasets.EMNIST(
            root=os.path.join(utils.get_data_root(), 'emnist'),
            split='letters',
            train=False,
            transform=data_transform,
            download=True
        )
    else:
        raise ValueError

    if args.dataset_name == 'omniglot':
        split = -1345
    elif args.dataset_name == 'emnist':
        split = -20000
    else:
        split = -10000
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4 if args.dataset_name == 'emnist' else 0
    )
    train_generator = data_.batch_generator(train_loader)
    val_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False
    )
    val_batch = next(iter(val_loader))[0]
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    def create_linear_transform():
        if args.linear_type == 'lu':
            return transforms.CompositeTransform([
                transforms.RandomPermutation(args.latent_features),
                transforms.LULinear(args.latent_features, identity_init=True)
            ])
        elif args.linear_type == 'svd':
            return transforms.SVDLinear(args.latent_features, num_householder=4,
                                        identity_init=True)
        elif args.linear_type == 'perm':
            return transforms.RandomPermutation(args.latent_features)
        else:
            raise ValueError

    def create_base_transform(i, context_features=None):
        if args.prior_type == 'affine-coupling':
            return transforms.AffineCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=args.latent_features,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features,
                                               out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_features,
                    context_features=context_features,
                    num_blocks=args.num_transform_blocks,
                    activation=F.relu,
                    dropout_probability=args.dropout_probability,
                    use_batch_norm=args.use_batch_norm
                )
            )
        elif args.prior_type == 'rq-coupling':
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=args.latent_features,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features,
                                               out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_features,
                    context_features=context_features,
                    num_blocks=args.num_transform_blocks,
                    activation=F.relu,
                    dropout_probability=args.dropout_probability,
                    use_batch_norm=args.use_batch_norm
                ),
                num_bins=args.num_bins,
                tails='linear',
                tail_bound=args.tail_bound,
                apply_unconditional_transform=args.apply_unconditional_transform,
            )
        elif args.prior_type == 'affine-autoregressive':
            return transforms.MaskedAffineAutoregressiveTransform(
                features=args.latent_features,
                hidden_features=args.hidden_features,
                context_features=context_features,
                num_blocks=args.num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        elif args.prior_type == 'rq-autoregressive':
            return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=args.latent_features,
                hidden_features=args.hidden_features,
                context_features=context_features,
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

    # ---------------
    # prior
    # ---------------
    def create_prior():
        if args.prior_type == 'standard-normal':
            prior = distributions_.StandardNormal((args.latent_features,))

        else:
            distribution = distributions_.StandardNormal((args.latent_features,))
            transform = transforms.CompositeTransform([
                transforms.CompositeTransform([
                    create_linear_transform(),
                    create_base_transform(i)
                ]) for i in range(args.num_flow_steps)
            ])
            transform = transforms.CompositeTransform([
                transform,
                create_linear_transform()
            ])
            prior = flows.Flow(transform, distribution)

        return prior

    # ---------------
    # inputs encoder
    # ---------------
    def create_inputs_encoder():
        if args.approximate_posterior_type == 'diagonal-normal':
            inputs_encoder = None
        else:
            inputs_encoder = nn_.ConvEncoder(
                context_features=args.context_features,
                channels_multiplier=16,
                dropout_probability=args.dropout_probability_encoder_decoder
            )
        return inputs_encoder

    # ---------------
    # approximate posterior
    # ---------------
    def create_approximate_posterior():
        if args.approximate_posterior_type == 'diagonal-normal':
            context_encoder = nn_.ConvEncoder(
                context_features=args.context_features,
                channels_multiplier=16,
                dropout_probability=args.dropout_probability_encoder_decoder
            )
            approximate_posterior = distributions_.ConditionalDiagonalNormal(
                shape=[args.latent_features],
                context_encoder=context_encoder
            )

        else:
            context_encoder = nn.Linear(args.context_features, 2 * args.latent_features)
            distribution = distributions_.ConditionalDiagonalNormal(
                shape=[args.latent_features],
                context_encoder=context_encoder
            )

            transform = transforms.CompositeTransform([
                transforms.CompositeTransform([
                    create_linear_transform(),
                    create_base_transform(i, context_features=args.context_features)
                ]) for i in range(args.num_flow_steps)
            ])
            transform = transforms.CompositeTransform([
                transform,
                create_linear_transform()
            ])
            approximate_posterior = flows.Flow(transforms.InverseTransform(transform), distribution)

        return approximate_posterior

    # ---------------
    # likelihood
    # ---------------
    def create_likelihood():
        latent_decoder = nn_.ConvDecoder(
            latent_features=args.latent_features,
            channels_multiplier=16,
            dropout_probability=args.dropout_probability_encoder_decoder
        )

        likelihood = distributions_.ConditionalIndependentBernoulli(
            shape=[1, 28, 28],
            context_encoder=latent_decoder
        )

        return likelihood

    prior = create_prior()
    approximate_posterior = create_approximate_posterior()
    likelihood = create_likelihood()
    inputs_encoder = create_inputs_encoder()

    model = vae.VariationalAutoencoder(
        prior=prior,
        approximate_posterior=approximate_posterior,
        likelihood=likelihood,
        inputs_encoder=inputs_encoder
    )

    n_params = utils.get_num_parameters(model)
    print('There are {} trainable parameters in this model.'.format(n_params))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_training_steps,
        eta_min=0
    )

    def get_kl_multiplier(step):
        if args.kl_multiplier_schedule == 'constant':
            return args.kl_multiplier_initial
        elif args.kl_multiplier_schedule == 'linear':
            multiplier = min(step / (args.num_training_steps * args.kl_warmup_fraction), 1.)
            return args.kl_multiplier_initial * (1. + multiplier)

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

    best_val_elbo = -np.inf
    tbar = tqdm(range(args.num_training_steps))
    for step in tbar:
        model.train()
        optimizer.zero_grad()

        batch = next(train_generator)[0].to(device)
        elbo = model.stochastic_elbo(batch, kl_multiplier=get_kl_multiplier(step))
        loss = - torch.mean(elbo)
        loss.backward()
        optimizer.step()
        scheduler.step(step)

        if (step + 1) % args.monitor_interval == 0:
            model.eval()
            with torch.no_grad():
                elbo = model.stochastic_elbo(val_batch.to(device))
                mean_val_elbo = elbo.mean()

            if mean_val_elbo > best_val_elbo:
                best_val_elbo = mean_val_elbo
                path = os.path.join(cutils.get_checkpoint_root(),
                                    '{}-best-val-{}.t'.format(args.dataset_name,
                                                              timestamp))
                torch.save(model.state_dict(), path)

            writer.add_scalar(
                tag='val-elbo',
                scalar_value=mean_val_elbo,
                global_step=step
            )

            writer.add_scalar(
                tag='best-val-elbo',
                scalar_value=best_val_elbo,
                global_step=step
            )

            with torch.no_grad():
                samples = model.sample(64)
            fig, ax = plt.subplots(figsize=(10, 10))
            cutils.gridimshow(make_grid(samples.view(64, 1, 28, 28), nrow=8), ax)
            writer.add_figure(tag='vae-samples', figure=fig, global_step=step)
            plt.close()

    # load best val model
    path = os.path.join(cutils.get_checkpoint_root(),
                        '{}-best-val-{}.t'.format(args.dataset_name,
                                                  timestamp))
    model.load_state_dict(torch.load(path))
    model.eval()

    np.random.seed(5)
    torch.manual_seed(5)

    # compute elbo on test set
    with torch.no_grad():
        elbo = torch.Tensor([])
        log_prob_lower_bound = torch.Tensor([])
        for batch in tqdm(test_loader):
            elbo_ = model.stochastic_elbo(batch[0].to(device))
            elbo = torch.cat([
                elbo,
                elbo_
            ])
            log_prob_lower_bound_ = model.log_prob_lower_bound(batch[0].to(device), num_samples=1000)
            log_prob_lower_bound = torch.cat([
                log_prob_lower_bound,
                log_prob_lower_bound_
            ])
    path = os.path.join(log_dir,
                        '{}-prior-{}-posterior-{}-elbo.npy'.format(
                            args.dataset_name,
                            args.prior_type,
                            args.approximate_posterior_type
                        ))
    np.save(path, utils.tensor2numpy(elbo))
    path = os.path.join(log_dir,
                        '{}-prior-{}-posterior-{}-log-prob-lower-bound.npy'.format(
                            args.dataset_name,
                            args.prior_type,
                            args.approximate_posterior_type
                        ))
    np.save(path, utils.tensor2numpy(log_prob_lower_bound))

    # save elbo and log prob lower bound
    mean_elbo = elbo.mean()
    std_elbo = elbo.std()
    mean_log_prob_lower_bound = log_prob_lower_bound.mean()
    std_log_prob_lower_bound = log_prob_lower_bound.std()
    s = 'ELBO: {:.2f} +- {:.2f}, LOG PROB LOWER BOUND: {:.2f} +- {:.2f}'.format(
        mean_elbo.item(),
        2 * std_elbo.item() / np.sqrt(len(test_dataset)),
        mean_log_prob_lower_bound.item(),
        2 * std_log_prob_lower_bound.item() / np.sqrt(len(test_dataset))
    )
    filename = os.path.join(log_dir, 'test-results.txt')
    with open(filename, 'w') as file:
        file.write(s)


def main():
    for i in range(args.num_runs):
        run(seed=i)


if __name__ == '__main__':
    from torch import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
