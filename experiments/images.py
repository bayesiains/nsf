import os
import time

import numpy as np
import scipy.misc
import sacred

import torch
from torch import nn

from sacred import Experiment, observers
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import autils
from experiments.autils import Conv2dSameSize, LogProbWrapper
from experiments.images_data import get_data, Preprocess

from data import load_num_batches
from torchvision.utils import make_grid, save_image

from nde import distributions, transforms, flows
import utils
import optim
import nn as nn_

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Capture job id on the cluster
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('SLURM_JOB_ID')

runs_dir = os.path.join(utils.get_data_root(), 'runs/images')
ex = Experiment('decomposition-flows-images')

fso = observers.FileStorageObserver.create(runs_dir, priority=1)
# I don't like how sacred names run folders.
ex.observers.extend([fso, autils.NamingObserver(runs_dir, priority=2)])

# For num_workers > 0 and tensor datasets, bad things happen otherwise.
torch.multiprocessing.set_start_method("spawn", force=True)

# noinspection PyUnusedLocal
@ex.config
def config():
    # Dataset
    dataset = 'fashion-mnist'
    num_workers = 0
    valid_frac = 0.01

    # Pre-processing
    preprocessing = 'glow'
    alpha = .05
    num_bits = 8
    pad = 2 # For mnist-like datasets

    # Model architecture
    steps_per_level = 10
    levels = 3
    multi_scale=True
    actnorm = True

    # Coupling transform
    coupling_layer_type = 'rational_quadratic_spline'
    spline_params = {
        'num_bins': 4,
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
    }

    # Coupling transform net
    hidden_channels = 256
    use_resnet = False
    num_res_blocks = 5 # If using resnet
    resnet_batchnorm = True
    dropout_prob = 0.

    # Optimization
    batch_size = 256
    learning_rate = 5e-4
    cosine_annealing = True
    eta_min=0.
    warmup_fraction = 0.
    num_steps = 100000
    temperatures = [0.5, 0.75, 1.]

    # Training logistics
    use_gpu = True
    multi_gpu = False
    run_descr = ''
    flow_checkpoint = None
    optimizer_checkpoint = None
    start_step = 0

    intervals = {
        'save': 1000,
        'sample': 1000,
        'eval': 1000,
        'reconstruct': 1000,
        'log': 10 # Very cheap.
    }

    # For evaluation
    num_samples = 64
    samples_per_row = 8
    num_reconstruct_batches = 10

class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)

@ex.capture
def create_transform_step(num_channels,
                          hidden_channels, actnorm, coupling_layer_type, spline_params,
                          use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob):
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = nn_.ConvResidualNet(in_channels=in_channels,
                                      out_channels=out_channels,
                                      hidden_channels=hidden_channels,
                                      num_blocks=num_res_blocks,
                                      use_batch_norm=resnet_batchnorm,
                                      dropout_probability=dropout_prob)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()
        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = utils.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)


@ex.capture
def create_transform(c, h, w,
                     levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing,
                     multi_scale):
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(
            input_shape=(c,h,w),
            output_shape=(c*h*w,)
        ))
        mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                                shift=-0.5)
    elif preprocessing == 'realnvp':
        preprocess_transform = transforms.CompositeTransform([
            # Map to [0,1]
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            # Map into unconstrained space as done in RealNVP
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - alpha)),
            transforms.Logit()
        ])

    elif preprocessing == 'realnvp_2alpha':
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit()
        ])
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct])

@ex.capture
def create_flow(c, h, w,
                flow_checkpoint, _log):
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(c, h, w)

    flow = flows.Flow(transform, distribution)

    _log.info('There are {} trainable parameters in this model.'.format(
        utils.get_num_parameters(flow)))

    if flow_checkpoint is not None:
        flow.load_state_dict(torch.load(flow_checkpoint))
        _log.info('Flow state loaded from {}'.format(flow_checkpoint))

    return flow

@ex.capture
def train_flow(flow, train_dataset, val_dataset, dataset_dims, device,
               batch_size, num_steps, learning_rate, cosine_annealing, warmup_fraction,
               temperatures, num_bits, num_workers, intervals, multi_gpu, actnorm,
               optimizer_checkpoint, start_step, eta_min, _log):
    run_dir = fso.dir

    flow = flow.to(device)

    summary_writer = SummaryWriter(run_dir, max_queue=100)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers)

    if val_dataset:
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)
    else:
        val_loader = None

    # Random batch and identity transform for reconstruction evaluation.
    random_batch, _ = next(iter(DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0 # Faster than starting all workers just to get a single batch.
    )))
    identity_transform = transforms.CompositeTransform([
        flow._transform,
        transforms.InverseTransform(flow._transform)
    ])

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
        _log.info('Optimizer state loaded from {}'.format(optimizer_checkpoint))

    if cosine_annealing:
        if warmup_fraction == 0.:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=num_steps,
                last_epoch=-1 if start_step == 0 else start_step,
                eta_min=eta_min
            )
        else:
            scheduler = optim.CosineAnnealingWarmUpLR(
                optimizer=optimizer,
                warm_up_epochs=int(warmup_fraction * num_steps),
                total_epochs=num_steps,
                last_epoch=-1 if start_step == 0 else start_step,
                eta_min=eta_min
            )
    else:
        scheduler = None

    def nats_to_bits_per_dim(x):
        c, h, w = dataset_dims
        return autils.nats_to_bits_per_dim(x, c, h, w)

    _log.info('Starting training...')

    best_val_log_prob = None
    start_time = None
    num_batches = num_steps - start_step

    for step, (batch, _) in enumerate(load_num_batches(loader=train_loader,
                                                       num_batches=num_batches),
                                      start=start_step):
        if step == 0:
            start_time = time.time() # Runtime estimate will be more accurate if set here.

        flow.train()

        optimizer.zero_grad()

        batch = batch.to(device)

        if multi_gpu:
            if actnorm and step == 0:
                # Is using actnorm, data-dependent initialization doesn't work with data_parallel,
                # so pass a single batch on a single GPU before the first step.
                flow.log_prob(
                    batch[:batch.shape[0] // torch.cuda.device_count(), ...]
                )

            # Split along the batch dimension and put each split on a separate GPU. All available
            # GPUs are used.
            log_density = nn.parallel.data_parallel(LogProbWrapper(flow), batch)
        else:
            log_density = flow.log_prob(batch)

        loss = -nats_to_bits_per_dim(torch.mean(log_density))

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            summary_writer.add_scalar('learning_rate', scheduler.get_lr()[0], step)

        summary_writer.add_scalar('loss', loss.item(), step)

        if best_val_log_prob:
            summary_writer.add_scalar('best_val_log_prob', best_val_log_prob, step)

        flow.eval() # Everything beyond this point is evaluation.

        if step % intervals['log'] == 0:
            elapsed_time = time.time() - start_time
            progress = autils.progress_string(elapsed_time, step, num_steps)
            _log.info("It: {}/{} loss: {:.3f} [{}]".format(step, num_steps, loss, progress))

        if step % intervals['sample'] == 0:
            fig, axs = plt.subplots(1, len(temperatures), figsize=(4 * len(temperatures), 4))
            for temperature, ax in zip(temperatures, axs.flat):
                with torch.no_grad():
                    noise = flow._distribution.sample(64) * temperature
                    samples, _ = flow._transform.inverse(noise)
                    samples = Preprocess(num_bits).inverse(samples)

                autils.imshow(make_grid(samples, nrow=8), ax)

                ax.set_title('T={:.2f}'.format(temperature))

            summary_writer.add_figure(tag='samples', figure=fig, global_step=step)

            plt.close(fig)

        if step > 0 and step % intervals['eval'] == 0 and (val_loader is not None):
            if multi_gpu:
                def log_prob_fn(batch):
                    return nn.parallel.data_parallel(LogProbWrapper(flow),
                                                     batch.to(device))
            else:
                def log_prob_fn(batch):
                    return flow.log_prob(batch.to(device))

            val_log_prob = autils.eval_log_density(log_prob_fn=log_prob_fn,
                                                   data_loader=val_loader)
            val_log_prob = nats_to_bits_per_dim(val_log_prob).item()

            _log.info("It: {}/{} val_log_prob: {:.3f}".format(step, num_steps, val_log_prob))
            summary_writer.add_scalar('val_log_prob', val_log_prob, step)

            if best_val_log_prob is None or val_log_prob > best_val_log_prob:
                best_val_log_prob = val_log_prob

                torch.save(flow.state_dict(), os.path.join(run_dir, 'flow_best.pt'))
                _log.info('It: {}/{} best val_log_prob improved, saved flow_best.pt'
                          .format(step, num_steps))

        if step > 0 and (step % intervals['save'] == 0 or step == (num_steps - 1)):
            torch.save(optimizer.state_dict(), os.path.join(run_dir, 'optimizer_last.pt'))
            torch.save(flow.state_dict(), os.path.join(run_dir, 'flow_last.pt'))
            _log.info('It: {}/{} saved optimizer_last.pt and flow_last.pt'.format(step, num_steps))

        if step > 0 and step % intervals['reconstruct'] == 0:
            with torch.no_grad():
                random_batch_ = random_batch.to(device)
                random_batch_rec, logabsdet = identity_transform(random_batch_)

                max_abs_diff = torch.max(torch.abs(random_batch_rec - random_batch_))
                max_logabsdet = torch.max(logabsdet)

            # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            # autils.imshow(make_grid(Preprocess(num_bits).inverse(random_batch[:36, ...]),
            #                         nrow=6), axs[0])
            # autils.imshow(make_grid(Preprocess(num_bits).inverse(random_batch_rec[:36, ...]),
            #                         nrow=6), axs[1])
            # summary_writer.add_figure(tag='reconstr', figure=fig, global_step=step)
            # plt.close(fig)

            summary_writer.add_scalar(tag='max_reconstr_abs_diff',
                                      scalar_value=max_abs_diff.item(),
                                      global_step=step)
            summary_writer.add_scalar(tag='max_reconstr_logabsdet',
                                      scalar_value=max_logabsdet.item(),
                                      global_step=step)

@ex.capture
def set_device(use_gpu, multi_gpu, _log):
    # Decide which device to use.
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError('use_gpu is True but CUDA is not available')

    if use_gpu:
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    if multi_gpu and torch.cuda.device_count() == 1:
        raise RuntimeError('Multiple GPU training requested, but only one GPU is available.')

    if multi_gpu:
        _log.info('Using all {} GPUs available'.format(torch.cuda.device_count()))

    return device

@ex.capture
def get_train_valid_data(dataset, num_bits, valid_frac):
    return get_data(dataset, num_bits, train=True, valid_frac=valid_frac)

@ex.capture
def get_test_data(dataset, num_bits):
    return get_data(dataset, num_bits, train=False)

@ex.command
def sample_for_paper(seed):
    run_dir = fso.dir

    sample(output_path=os.path.join(run_dir, 'samples_small.png'),
           num_samples=30,
           samples_per_row=10)

    sample(output_path=os.path.join(run_dir, 'samples_big.png'),
           num_samples=100,
           samples_per_row=10,
           seed=seed + 1)

@ex.command(unobserved=True)
def eval_on_test(batch_size, num_workers, seed, _log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()
    test_dataset, (c, h, w) = get_test_data()
    _log.info('Test dataset size: {}'.format(len(test_dataset)))
    _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))

    flow = create_flow(c, h, w).to(device)

    flow.eval()

    def log_prob_fn(batch):
        return flow.log_prob(batch.to(device))

    test_loader=DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           num_workers=num_workers)
    test_loader = tqdm(test_loader)

    mean, err = autils.eval_log_density_2(log_prob_fn=log_prob_fn,
                                          data_loader=test_loader,
                                          c=c, h=h, w=w)
    print('Test log probability (bits/dim): {:.2f} +/- {:.4f}'.format(mean, err))

@ex.command(unobserved=True)
def sample(seed, num_bits, num_samples, samples_per_row, _log, output_path=None):
    torch.set_grad_enabled(False)

    if output_path is None:
        output_path = 'samples.png'

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    _, _, (c, h, w) = get_train_valid_data()

    flow = create_flow(c, h, w).to(device)
    flow.eval()

    preprocess = Preprocess(num_bits)

    samples = flow.sample(num_samples)
    samples = preprocess.inverse(samples)

    save_image(samples.cpu(), output_path,
               nrow=samples_per_row,
               padding=0)

@ex.command(unobserved=True)
def num_params(_log):
    _, _, (c, h, w) = get_train_valid_data()
    # c, h, w = 3, 256, 256
    create_flow(c, h, w)

@ex.command(unobserved=True)
def eval_reconstruct(num_bits, batch_size, seed, num_reconstruct_batches, _log, output_path=''):
    torch.set_grad_enabled(False)

    device = set_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset, _, (c, h, w) = get_train_valid_data()

    flow = create_flow(c, h, w).to(device)
    flow.eval()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    identity_transform = transforms.CompositeTransform([
        flow._transform,
        transforms.InverseTransform(flow._transform)
    ])

    first_batch = True
    abs_diff = []
    for batch,_ in tqdm(load_num_batches(train_loader, num_reconstruct_batches),
                        total=num_reconstruct_batches):
        batch = batch.to(device)
        batch_rec, _ = identity_transform(batch)
        abs_diff.append(torch.abs(batch_rec - batch))

        if first_batch:
            batch = Preprocess(num_bits).inverse(batch[:36, ...])
            batch_rec = Preprocess(num_bits).inverse(batch_rec[:36, ...])

            save_image(batch.cpu(), os.path.join(output_path, 'invertibility_orig.png'),
                       nrow=6,
                       padding=0)

            save_image(batch_rec.cpu(), os.path.join(output_path, 'invertibility_rec.png'),
                       nrow=6,
                       padding=0)

            first_batch = False

    abs_diff = torch.cat(abs_diff)

    print('max abs diff: {:.4f}'.format(torch.max(abs_diff).item()))


@ex.command(unobserved=True)
def profile(batch_size, num_workers):
    train_dataset, _, _ = get_train_valid_data()

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers)
    for _ in tqdm(load_num_batches(train_loader, 1000),
                  total=1000):
        pass

@ex.command(unobserved=True)
def plot_data(num_bits, num_samples, samples_per_row, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset, _, _ = get_train_valid_data()

    samples = torch.cat([train_dataset[i][0] for i in np.random.randint(0, len(train_dataset),
                                                                        num_samples)])
    samples = Preprocess(num_bits).inverse(samples)

    save_image(samples.cpu(),
               'samples.png',
               nrow=samples_per_row,
               padding=0)

@ex.automain
def main(seed, _log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    train_dataset, val_dataset, (c, h, w) = get_train_valid_data()

    _log.info('Training dataset size: {}'.format(len(train_dataset)))

    if val_dataset is None:
        _log.info('No validation dataset')
    else:
        _log.info('Validation dataset size: {}'.format(len(val_dataset)))

    _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))

    flow = create_flow(c, h, w)

    train_flow(flow, train_dataset, val_dataset, (c, h, w), device)
