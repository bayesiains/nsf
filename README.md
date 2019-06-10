# Neural Spline Flows

Implementation of modules and experiments for Neural Spline Flows paper.

## Dependencies

See `environment.yml` for required Conda/pip packages, or use it create a Conda environment with 
all dependencies:
```bash
conda env create -f environment.yml
```

Tested with Python 3.5 and PyTorch 1.1.

## Data

Pre-processed UCI data is available at https://zenodo.org/record/1161203#.Wmtf_XVl8eN.

Data for VAE and image experiments is downloaded automatically using either `torchvision` or custom 
data providers.

## Usage

`DATAROOT` environment variable needs to be set before running experiments.

### 2D image density experiments

Use `conor/experiments/face.py` or `conor/experiments/plane.py`.

### UCI experiments

Use `conor/experiments/uci.py`.

### VAE experiments

Use `conor/experiments/vae.py`.

### Image experiments

Use `artur/images.py`.

[Sacred](https://github.com/IDSIA/sacred) is used to organize image experiments. See the 
[documentation](http://sacred.readthedocs.org) for more information.

`artur/image_configs` contains .json configurations used for RQ-NSF (C) experiments. For baseline
experiments use `coupling_layer_type='affine'`.

For example, to run RQ-NSF (C) on CIFAR-10 8-bit:
```bash
python artur/images.py with artur/image_configs/cifar-10-8bit.json
```

Corresponding affine baseline run:
```bash
python artur/images.py with artur/image_configs/cifar-10-8bit.json coupling_layer_type='affine'
```

To evaluate on the test set:
```bash
python artur/images.py eval_on_test with artur/image_configs/cifar-10-8bit.json 
flow_checkpoint='<saved_checkpoint>'
```

To sample:
```bash
python artur/images.py sample with artur/image_configs/cifar-10-8bit.json 
flow_checkpoint='<saved_checkpoint>'
```







