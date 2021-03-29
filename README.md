# XDLreg

`[Last update: 2021-03-29]`

![PumpkinNet](Pumpkin.jpg)

Simulation study on explainable deep learning (XDL) for regression tasks.

## PumpkinNet
In `./PumpkinNet/` is the implementation of 2D-convolutional neural networks (CNN) model for the prediction of *age* from 2D simulated images (*Pumpkins*).  

## LRP
In `./LRP/` one can find the implementation of the *Layer-wise relevance propagation algorithm* (LRP), which highlights information in the input space being relevant for the given model prediction. <br>
Here, the LRP will be applied on the predictions of the PumpkinNet extracting pixels in the simulated images that were relevant for the prediction.

## Subanalysis
The subfolder `./Subanalysis/`  contains mainly jupyter notebooks for subanalyses.
