"""
Create LRP heatmaps for prediction model.

# iNNvestigate
* Source https://github.com/albermax/innvestigate
* contains most pragmatic implementation of LRP.
* requirements: prediction model must be implemented in (native) Keras

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import
import os
import numpy as np
import matplotlib.pyplot as plt
import innvestigate

from xdlreg.utils import p2results, save_obj, load_obj
from xdlreg.SimulationData import split_simulation_data
from xdlreg.PumpkinNet import load_trained_model, get_model_data, is_binary_classification
from xdlreg.LRP.apply_heatmap import apply_colormap, create_cmap, gregoire_black_firered


# %% Set global paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
p2relevance = os.path.join(p2results, "relevance")


# %% Create heatmaps & plots << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


def create_relevance_dict(model_name: str, subset: str = "test",
                          analyzer_type: str = "lrp.sequential_preset_a", save: bool = True):
    """
    Compute relevance maps of given analyzer type (default is LRP) for a model and all subjects it was
    evaluated on.

    :param model_name: name of model
    :param subset: heatmaps are most representative for test set. But also possible to plot heatmaps of
                   subjects in the 'validation' and 'training' set.
    :param analyzer_type: type of analyzer for model decisions. Note colormaps might be off for other
                          types of analyzer, which iNNvestigate offers.
    :param save: Whether to save dict externally, for quick access.
    :return:
    """
    try:
        rel_obj = load_obj(name=f"relevance-maps_{subset}-set", folder=os.path.join(p2relevance,
                                                                                    model_name))
    except FileNotFoundError:

        _model = load_trained_model(model_name)
        _x, _y = get_model_data(model_name=model_name)

        if subset == "test":
            xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        else:
            # xdata, ydata = ...
            raise NotImplementedError("Implement for other subsets if required!")

        analyzer = innvestigate.create_analyzer(analyzer_type, _model, disable_model_checks=True,
                                                neuron_selection_mode="max_activation")
        rel_obj = analyzer.analyze(xdata, neuron_selection=None).squeeze()

        if save:
            save_obj(obj=rel_obj, name=f"relevance-maps_{subset}-set", folder=os.path.join(p2relevance,
                                                                                           model_name))

    return rel_obj


def plot_simulation_heatmaps(model_name: str, n_subjects: int = 20, subset: str = "test",
                             analyzer_type: str = "lrp.sequential_preset_a", pointers: bool = True,
                             cbar: bool = False, true_scale: bool = False, fm: str = "pdf"):

    """
    Plot heatmaps/relevance maps for a set of subjects.

    :param model_name: name of model
    :param n_subjects: number of subjects/plots.
    :param subset: heatmaps are most representative for test set. But also possible to plot heatmaps of
                   subjects in the 'validation' and 'training' set.
    :param analyzer_type: type of analyzer for model decisions. Note colormaps might be off for other
                          types of analyzer, which iNNvestigate offers.
    :param pointers: Plots include markers which point to simulated lesions and atrophies in the image.
    :param cbar: Whether to add a color bar to the plot.
    :param true-scale: True: do not normalize the relevance values between (-1, 1)
    :param fm: saving format of plot, including 'pdf', 'png' ...
    """

    # Get model
    _model = load_trained_model(model_name)

    # Get relevance maps
    rel_obj = create_relevance_dict(model_name=model_name, subset=subset, analyzer_type=analyzer_type)

    if is_binary_classification(model_name):
        # TODO
        raise NotImplementedError("Plotting of heatmaps for (binary) classification must be implemented!")

    # Prep data
    pdata = get_model_data(model_name=model_name, for_keras=False)
    _x, _y = pdata.data2numpy(for_keras=True)
    if subset == "test":
        xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        didx = split_simulation_data(xdata=_x, ydata=_y, return_idx=True, only_test=True)
    else:
        # xdata, ydata = ...
        raise NotImplementedError("Not implemented yet for other subsets than 'test'")

    # Check whether too many plots are requested, and adjust if necessary
    n_subjects = n_subjects if len(ydata) >= n_subjects else len(ydata)

    for sub in range(n_subjects):
        img = xdata[sub].copy()
        img = img[np.newaxis, ...]
        a = rel_obj[sub]

        col_a = apply_colormap(robj=a, inputimage=img.squeeze(), cintensifier=5., gamma=.2,
                               true_scale=true_scale)

        sub_y = ydata[sub]
        sub_yt = _model.predict(img).item()

        fig = plt.figure(num=f"S{sub}, age={int(sub_y)}, pred={sub_yt:.2f}")
        ax = fig.add_subplot(1, 1, 1)
        aximg = plt.imshow(col_a[0], cmap=create_cmap(gregoire_black_firered))

        if cbar:
            cbar_range = (-1, 1) if not true_scale else (-col_a[2], col_a[2])
            caxbar = fig.colorbar(aximg, ax=ax, fraction=0.048, pad=0.04)
            caxbar.set_ticks(np.linspace(0, 1, 7), True)
            caxbar.ax.set_yticklabels(labels=[f"{tick:.2g}" for tick in np.linspace(
                cbar_range[0], cbar_range[1], len(caxbar.get_ticks()))])

        plt.tight_layout()

        # Save plots
        parent_dir = os.path.join(p2relevance, _model.name, "plots")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        plt.savefig(os.path.join(parent_dir, f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}.{fm}"))

        if pointers:
            phead = pdata.data[didx[0]+sub]

            # Mark atrophies
            for coord in phead.atrophy_coords:
                plt.plot(coord[1], coord[0], "s", color="#D3F5D4", ms=2, alpha=.9)  # ms=4: full-pixel

            # Arrows to lesions
            for coord in phead.lesion_coords:
                # Shadow
                plt.annotate(text='', xy=coord[::-1],
                             xytext=np.array(coord[::-1]) + [-4.6, 5.4],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             arrowprops=dict(arrowstyle='simple', color="black", alpha=.5))

                # Arrow
                plt.annotate(text='', xy=coord[::-1],
                             xytext=np.array(coord[::-1]) + [-5, 5],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             arrowprops=dict(arrowstyle='simple', color="#E3E7E3", alpha=.9))

            plt.tight_layout()

            # Save plots
            parent_dir = os.path.join(p2relevance, _model.name, "plots")
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            plt.savefig(os.path.join(parent_dir,
                                     f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}_pointer.{fm}"))
        plt.close()

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
