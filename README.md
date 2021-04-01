# XDLreg

*Simulation study on explainable deep learning (XDL) for regression tasks.*

`[Last update: 2021-04-01]`

<img src="Pumpkin.jpg" alt="PumpkinNet" width="350">

[![Python.pm](https://img.shields.io/badge/python-3.7-brightgreen.svg?maxAge=259200)](#) [![version](https://img.shields.io/badge/version-1.0.0-yellow.svg?maxAge=259200)](#)

## xdlreg
`./xdlreg/` contains the project code.

### PumpkinNet
`./xdlreg/PumpkinNet.py` contains the implementation of a 2D-convolutional neural network (CNN) model for the prediction of *age* from 2D simulated images (*pumpkins*). The model is a 2D-adaptation of the brain-age prediction model reported in [Hofmann et al., 2021](DOI), which was trained on 3D-MRIs.

### Pumpkin dataset
`./xdlreg/SimulationData.py` is the code for the simulation of 2D-images of *pumpkin heads*. Ageing ist simulated in form of added *atrophies* and *lesions*.

### LRP analysis
In `./xdlreg/LRP/` one can find the implementation of the *Layer-wise relevance propagation algorithm* (LRP), which highlights information in the input space being relevant for the given model prediction. <br>
Here, the LRP will be applied on the `PumpkinNet` extracting pixels in the simulated images that were relevant for the model prediction.

`./xdlreg/LRP/create_heatmaps.py` contains functions to create relevance objects. That is, per sample the script can analyse the model prediction.

`./xdlreg/LRP/apply_heatmap.py` contains functions for visualising LRP heatmaps.

## Setup
It is recommended to use a new virtual environment `virtualenv` for installing and running the pipeline.

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
pip install github.ADDRESS
```

## Get started
After installation, there are two ways to run the full pipeline, predicting age from the simulated images (*regression*) and explaining the model predictions with *LRP*:

1. Dynamically load script within `python3.7`

```python
from xdlreg.run_simulation import run_simulation

run_simulation(path=YOUR_PATH, n_samples=N_SAMPLES, uniform=TRUE/FALSE, target_bias=TARGET_BIAS, epochs=N_EPOCHS, plot_n_heatmaps=N_HEATMAPS_TO_PLOT)

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

## Subanalysis
The subfolder `./Subanalysis/`  contains mainly jupyter notebooks for sub-analyses.

## Cite

*add citation*
