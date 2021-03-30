"""
Functions for LRP on PumpkinNet

# iNNvestigate
* Source https://github.com/albermax/innvestigate
* contains most pragmatic implementation of LRP.
* requirements: prediction model must be implemented in (native) Keras

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""
# %% Import

import os
import numpy as np
import innvestigate
import matplotlib.pyplot as plt

from utils import cprint
from PumpkinNet.visualize import prep_save_folder, plot_mid_slice
from LRP.apply_heatmap import apply_colormap, create_cmap, gregoire_black_firered


# %% << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def analyze_model(mri, analyzer_type, model_, norm, neuron_selection=None):
    # Create analyzer
    analyzer = innvestigate.create_analyzer(
        analyzer_type, model_,
        disable_model_checks=True,
        neuron_selection_mode="index" if isinstance(neuron_selection, int) else "max_activation")

    # analyzer.fit(x_train)  # necessary for e.g., PatternNet
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(mri, neuron_selection=neuron_selection)

    # Aggregate along color channels
    a = a[0, ..., 0]  # (198, 198, 198)

    if norm:
        # Normalize to [-1, 1]
        a /= np.max(np.abs(a))

    # print(f"{analyzer_type} min: {np.min(a):.3f} | max: {np.max(a):.3f}\n")

    return a


def plot_heatmap(sub_nr, t, t_y, ipt, analyzer_obj, analyzer_type, fn_suffix="", save_folder="",
                 save_plot=True, **kwargs):

    a = analyzer_obj.copy()

    save_folder = prep_save_folder(os.path.join("./processed/Keras/Interpretation/", save_folder))

    # # Plot heatmap over T1 image
    # Figure name
    sub_nr = f"S{sub_nr}" if str(sub_nr).isnumeric() else sub_nr
    figname = f"{analyzer_type}_{sub_nr}_groundtruth={t:{'.2f' if isinstance(t, float) else ''}}_" \
              f"pred={t_y:{'.2f' if isinstance(t_y, (float, np.floating)) else ''}}{fn_suffix}"

    # Extract kwargs
    cintensifier = kwargs.pop("cintensifier", 1.)
    clipq = kwargs.pop("clipq", 1e-2)
    min_sym_clip = kwargs.pop("min_sym_clip", True)
    true_scale = kwargs.pop("true_scale", False)
    wbg = kwargs.pop("wbg", False)  # white background, quick&dirty implementation (remove if no benefit)

    # Render image

    if wbg:  # make background white
        # Naively make very small values white = 1.
        # ipt[ipt < 0.01] = 1.
        # Different way: Mirror values
        ipt = -1 * ipt + 1
        # # Yet another mirroring
        # pre_max = ipt.max()
        # ipt = -1 * ipt + pre_max
        # ipt /= ipt.max()  # stretches a bit the value distribution

    colored_a = apply_colormap(R=a, inputimage=ipt, cmapname='black-firered',
                               cintensifier=cintensifier, clipq=clipq, min_sym_clip=min_sym_clip,
                               gamma=.2, true_scale=true_scale)

    cbar_range = (-1, 1) if not true_scale else (-colored_a[2], colored_a[2])

    plot_mid_slice(mri=colored_a[0], figname=figname,
                   cmap=create_cmap(gregoire_black_firered), c_range="full",
                   cbar=True, cbar_range=cbar_range, edges=False,
                   save=save_plot, save_folder=save_folder, **kwargs)

# << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< END
