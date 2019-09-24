# Neural Spline Flows

Code and experiments for the paper:

> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, _Neural Spline Flows_, NeurIPS 2019.
> [[arXiv]](https://arxiv.org/abs/1906.04032) [[bibtex]](https://gpapamak.github.io/bibtex/neural_spline_flows.bib)

## Dependencies

See `environment.yml` for required Conda/pip packages, or use this to create a Conda environment with 
all dependencies:
```bash
conda env create -f environment.yml
```

Tested with Python 3.5 and PyTorch 1.1.

## Data

Data for density-estimation experiments is available at https://zenodo.org/record/1161203#.Wmtf_XVl8eN.

Data for VAE and image-modeling experiments is downloaded automatically using either `torchvision` or custom 
data providers.

## Usage

`DATAROOT` environment variable needs to be set before running experiments.

### 2D toy density experiments

Use `experiments/face.py` or `experiments/plane.py`.

### Density-estimation experiments

Use `experiments/uci.py`.

### VAE experiments

Use `experiments/vae_.py`.

### Image-modeling experiments

Use `experiments/images.py`.

[Sacred](https://github.com/IDSIA/sacred) is used to organize image experiments. See the 
[documentation](http://sacred.readthedocs.org) for more information.

`experiments/image_configs` contains .json configurations used for RQ-NSF (C) experiments. For baseline experiments use `coupling_layer_type='affine'`.

For example, to run RQ-NSF (C) on CIFAR-10 8-bit:
```bash
python experiments/images.py with experiments/image_configs/cifar-10-8bit.json
```

Corresponding affine baseline run:
```bash
python experiments/images.py with experiments/image_configs/cifar-10-8bit.json coupling_layer_type='affine'
```

To evaluate on the test set:
```bash
python experiments/images.py eval_on_test with experiments/image_configs/cifar-10-8bit.json flow_checkpoint='<saved_checkpoint>'
```

To sample:
```bash
python experiments/images.py sample with experiments/image_configs/cifar-10-8bit.json flow_checkpoint='<saved_checkpoint>'
```







