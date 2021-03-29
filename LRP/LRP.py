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

        # TODO Try norm 0, 1
        # a = normalize(a, 0, 1)

    # print(f"{analyzer_type} min: {np.min(a):.3f} | max: {np.max(a):.3f}\n")

    return a


def non_zero_analyzer_and_clim(sub_nr, a, t, t_y, analy_type, save_folder):
    bincnt = np.histogram(a.flatten(), 100)

    a_no_zero = a[np.where(np.logical_or(bincnt[1][np.max([np.argmax(bincnt[0]) - 1, 0])] > a,
                                         a > bincnt[1][np.argmax(bincnt[0]) + 1]))]  # flattened

    clim = (
        np.mean(a_no_zero) - np.std(a_no_zero) * 1.5, np.mean(a_no_zero) + np.std(a_no_zero) * 1.5)
    # print(f"clim[0]: {clim[0]:.3f} | clim[1]: {clim[1]:.3f}\n")
    # TODO calc glob distribution of relevance range

    # clim = (np.percentile(a_no_zero, 5), np.percentile(a_no_zero, 95))

    hist_fig = plt.figure(analy_type + " Hist", figsize=(10, 5))
    plt.subplot(1, 2, 1)
    bincnt_no_zero = plt.hist(a_no_zero, bins=100, log=False)
    plt.vlines(x=clim[1], ymin=0, ymax=np.max(bincnt_no_zero[0]))
    plt.vlines(x=clim[0], ymin=0, ymax=np.max(bincnt_no_zero[0]))
    plt.title("Leave peak out Histogram")
    plt.subplot(1, 2, 2)
    _ = plt.hist(a.flatten(), bins=100, log=True)  # bincnt
    plt.vlines(x=clim[1], ymin=0, ymax=np.max(bincnt_no_zero[0]))
    plt.vlines(x=clim[0], ymin=0, ymax=np.max(bincnt_no_zero[0]))
    plt.title("Full Histogram")
    plt.tight_layout()

    plt.savefig(f"{save_folder}{analy_type}_S{sub_nr}_groundtruth={t}_"
                f"pred={t_y:{'.2f' if isinstance(t_y, float) else ''}}_heatmap_hist.png")
    plt.close(hist_fig)

    return clim


def plot_heatmap(sub_nr, t, t_y, ipt, analyzer_obj, analyzer_type, fix_clim=None, fn_suffix="",
                 save_folder="", save_plot=True, **kwargs):
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


def apply_analyzer_and_plot_heatmap(subjects, mris, targets, pre_model, ls_analy_type, binary_cls,
                                    fix_clim=None, ls_suptitles=None,
                                    neuron_analysis=False, classes=None, save_folder=""):
    # Check argument
    if ls_suptitles:
        assert len(ls_suptitles) == len(subjects), "'ls_suptitles' must have same length as subjects!"

    # Prepare types of analysis
    ls_analy_type = ls_analy_type if isinstance(ls_analy_type, list) else [ls_analy_type]
    ls_analy_type = ls_analy_type if ls_analy_type[0] != "input" else ls_analy_type[1:]  # If, rm 'input'

    # Prepare folder
    fullp2save_folder = prep_save_folder(os.path.join("./processed/Keras/Interpretation/", save_folder))

    # Run LRP through all given subjects
    for subidx, sub_nr in enumerate(subjects):
        suptitle = ls_suptitles[subidx] if ls_suptitles else None

        # Check whether LRP was done already
        sub_analyser = []  # list for subject specfic analyser
        for an in ls_analy_type:
            if not any([(an in fi and f"S{sub_nr}_" in fi) for fi in os.listdir(fullp2save_folder)]):
                sub_analyser.append(an)  # only apply analyser type which haven't been used before
        if len(sub_analyser) == 0:
            continue  # in case all analyser were applied already, continue with next subject

        # Get model performance & LRP heatmap on individual
        i_sub = np.array([sub_nr])
        cprint(f"Subject {sub_nr}", fm="ul")

        i_mri = mris[i_sub]
        if i_mri.ndim < 5:  # should be e.g.:  (1, 198, 198
            # For incomplete sequence data (e.g. SWI) & return_nones=True
            i_mri = i_mri[0][0]
            if i_mri is None:
                cprint(f"No MRI for Subject {sub_nr} in given dataset!", 'r')
                continue
            else:
                i_mri = i_mri.reshape((1, *list(i_mri.shape), 1))  # (1, 198, 198, 198, 1)

        i_pred = classes[int(np.argmax(pre_model.predict(x=i_mri)))] if binary_cls else \
            pre_model.predict(x=i_mri)[0][0].item()  # ,item(): np.float => float
        i_target = classes[int(np.argmax(targets[i_sub]))] if binary_cls else targets[i_sub][0]
        try:
            i_target = i_target.item()  # np.float => float
        except AttributeError:
            pass
        print(f"prediction:\t{i_pred:{'.2f' if isinstance(i_pred, float) else ''}}")
        print(f"target:\t\t{i_target}")

        # Get model prediction for given MRI
        t = int(i_target) if isinstance(i_target, float) else i_target  # else str
        t_y = np.round(i_pred, 2) if isinstance(i_pred, float) else i_pred
        # t_y = pre_model.predict(x=mri)[0][0]

        neuron_selection = range(pre_model.output_shape[-1]) if neuron_analysis else [None]  # n classes

        ipt = analyze_model(mri=i_mri, analyzer_type="input", model_=pre_model, norm=True)

        for idx, analy_type in enumerate(sub_analyser):

            for neuro_i in neuron_selection:
                a = analyze_model(mri=i_mri, analyzer_type=analy_type, model_=pre_model, norm=True,
                                  **({"neuron_selection": neuro_i} if isinstance(neuro_i, int) else {}))

                plot_heatmap(sub_nr=sub_nr, t=t, t_y=t_y, ipt=ipt, analyzer_obj=a,
                             analyzer_type=analy_type,
                             fix_clim=fix_clim,
                             fn_suffix=f"_N-{classes[neuro_i]}" if isinstance(neuro_i, int) else "",
                             save_folder=save_folder,
                             suptitle=suptitle)

            # TODO plot 3d (maybe scatter 3d) ?!
            # multi_slice_viewer(mri=a, cmap="seismic", clim=clim)

# << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< END
