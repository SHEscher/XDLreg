# XDLreg

*A package to study explainable deep learning (XDL) for regression tasks on a simulated image dataset.*

`[Last update: 2021-04-08]`

<img src="Pumpkin.png" alt="PumpkinNet" width="350">

[![Python.pm](https://img.shields.io/badge/python-3.7≥version≥3.6-brightgreen.svg?maxAge=259200)](#) [![version](https://img.shields.io/badge/version-1.0.0-yellow.svg?maxAge=259200)](#)

## xdlreg
`xdlreg/` contains the project code.

### PumpkinNet
`xdlreg/PumpkinNet.py` contains the implementation of a 2D-convolutional neural network (CNN) model for the prediction of *age* from 2D simulated images (*pumpkins*). The model is a 2D-adaptation of the brain-age prediction model reported in [Hofmann et al. (2021)](DOI), which was trained on 3D-MRIs.

### Pumpkin dataset
`xdlreg/SimulationData.py` is the code for the simulation of 2D-images of *pumpkin heads*. Ageing ist simulated in form of added *atrophies* and *lesions*. For details visit the added `jupyter notebook`.

### LRP analysis
In `xdlreg/LRP/` one can find the implementation of the [*Layer-wise relevance propagation*](https://depositonce.tu-berlin.de/handle/11303/8813) (LRP) algorithm, which highlights information in the input space that is relevant for the given model prediction. <br>
Here, LRP will be applied on the `PumpkinNet` extracting pixels in the simulated images that were relevant for the model prediction.

`xdlreg/LRP/create_heatmaps.py` contains functions to create relevance objects. That is, per sample the script can analyse the model prediction of age.

`xdlreg/LRP/apply_heatmap.py` contains functions for visualising LRP heatmaps (or *relevance maps*).

## Setup
It is recommended to use a new virtual environment `virtualenv` for installing and running the pipeline.

Note, the package was only tested for `Python3.6.9` and `Python3.7.6`. Since, the pipeline applies *native* `Keras` (v.2.2.4) with a respective older `tensorflow` (tf) backend (v.1.14.0rc1), later *Python* versions (v.3.8+) do not work. Also, up until now, there is no compatibility between older *tf* versions (< v.2.+) and *M1* chips (since late 2020) by *Apple*.

```console
# Create virtual environment
virtualenv --python=/PATH/TO/PYTHON3.7 /PATH/TO/VENV_DIR

# activate it
source /PATH/TO/VENV_DIR/bin/activate

# create your project path
mkdir YOUR_PATH
cd YOUR_PATH
```

Installing goes via
```console
# after downloading/cloning the repository
pip install PATH/TO/XDLreg

# OR
pip install git+https://github.com/SHEscher/XDLreg.git
```

## Get started
After installation, there are two ways to run the full pipeline, predicting age from the simulated images (*regression*) and explaining the model predictions with *LRP*:

1. Dynamically load script within `python3.7`

```python
from xdlreg.run_simulation import run_simulation

run_simulation(path=YOUR_PATH, n_samples=N_SAMPLES, uniform=TRUE/FALSE,
               target_bias=TARGET_BIAS, epochs=N_EPOCHS,
               plot_n_heatmaps=N_HEATMAPS_TO_PLOT)
```
2. Or via a console (all *flags* are optional):

```console
xdlreg /
  --path YOUR_PATH /
  --n_samples N_SAMPLES /
  --uniform BOOL /
  --target_bias TARGET_BIAS/
  --epochs NUMBER_OF_EPOCHS/
  --plot_n_heatmaps
```
For and more information on each FLAG, type:
```console
xdlreg -h
```

## Explore simulated data, model, and relevance maps
The folder `notebook/` contains a [`jupyter notebook`](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) for the exploration of the functionality of the introduced pipeline.

If you use `pip install` to get access to the `xdlreg` module, the `notebook` must be downloaded separately, and started within the same environment (e.g., `virtualenv`), where the module is installed.

```console
# If necessary:
pip install jupyterlab

# Add kernel
ipython kernel install --name "VENV_DIR" --user

# Start notebook
jupyter lab PATH/TO/notebook/LRP4RegressionSimulation.ipynb
```

## Citation

In case of adaptation, and/or usage of this code, please cite:

*add citation*
