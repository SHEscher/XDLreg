# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for LRP on MRInet

# iNNvestigate
* https://github.com/albermax/innvestigate
*  Most efficient/pragmatic implementation, however tf model must be re-implemented in (native) Keras
* see also implementation of quantitive evaluation (Samek et al., 2018)
* Contrastive Layer-wise Relevance Propagation or CLRP : https://github.com/albermax/CLRP

# LRP toolbox
Sebastian Lapuschkin (FH HHI): https://github.com/sebastian-lapuschkin/lrp_toolbox

# LRP wrappers for tensorflow
* Vignesh Srinivasan (FH HHI): https://github.com/VigneshSrinivasan10/interprettensor
* Niels Warncke (?): https://github.com/nielsrolf/tensorflow-lrp

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2019, 2020
"""
# %% Import

import innvestigate
import nibabel as nb

from utils import root_path
from PumpkinNet.train_kerasMRInet import load_trained_model, crop_model_name
from PumpkinNet.visualize import *
from apply_heatmap import apply_colormap, create_cmap, gregoire_black_firered

# import sys
# import os
# sys.path.append((os.path.abspath(".").split("DeepAge")[0] + "DeepAge/Analysis/Modelling/MRInet/"))

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Paths

p2intrprt_rslts = os.path.join(root_path, "Results/Interpretation/")


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

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
                 rendermode="alpha", save_folder="", save_plot=True, **kwargs):
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
    if rendermode.lower() == "overlay":

        assert fix_clim is None or isinstance(fix_clim,
                                              float), "fix_clim must be None or float!"  # e.g. 0.02

        # Define color-lims for input-image & heatmap
        iptlim = non_zero_analyzer_and_clim(sub_nr=sub_nr, a=ipt, t=t, t_y=t_y, analy_type="input",
                                            save_folder=save_folder)
        clim = non_zero_analyzer_and_clim(sub_nr=sub_nr, a=a, t=t, t_y=t_y, analy_type=analyzer_type,
                                          save_folder=save_folder)
        # Center colour map
        clim = list(clim)
        minmaxlim = fix_clim if fix_clim else np.min(np.abs(clim))  # np.max for more fine-grained col-map
        clim[0], clim[1] = -1 * minmaxlim, minmaxlim
        clim[0], clim[1] = -1 * minmaxlim, minmaxlim
        # clim[0], clim[1] = -1*.01, .01
        clim = tuple(clim)
        # Plot T1-image
        plot_mid_slice(mri=ipt, figname=figname, cmap="binary", clim=iptlim, alpha=.3, save=False)
        # Plot heatmap over T1
        plot_mid_slice(mri=a, figname=figname, cmap="seismic", clim=clim, c_range=None, alpha=.8,
                       cbar=True,
                       save=save_plot, save_folder=save_folder, kwargs=kwargs)

    elif rendermode.lower() == "alpha":

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

    else:
        raise ValueError("rendermode must be either 'alpha' or 'overlay'")


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


def create_heatmap_nifti(sic, model_name, analyzer_type='lrp.sequential_preset_a', analyzer_obj=None,
                         save=False, logging=False, **kwargs):
    # TODO function works currently only for models which were trained on pruned & cubified MRIs

    model_name = crop_model_name(model_name=model_name)  # remove '_final.h5' from name

    p2file = os.path.join(p2intrprt_rslts, model_name, sic, f"{analyzer_type}_relevance_maps.nii.gz")
    p2org = p2file.replace(".nii.", ".pkl.")
    # p2file = os.path.join(p2intrprt_rslts, model_name, sic, f"{analyzer_type}_hm.nii.gz")

    # Load or create nifti heatmap
    if os.path.isfile(p2file):
        fullsize_a_nii = nb.load(p2file)

    else:  # Create
        if check_system() != "MPI":
            raise EnvironmentError("Function works only on MPI servers")

        from load_mri_data import load_sic_mri, mri_sequences

        # Reverse pruning
        space = kwargs.pop("space", None)
        if space is None:
            space = "mni" if "MNI" in model_name else "raw" if "RAW" in model_name else "fs"

        mri_sequence = kwargs.pop("mri_sequence", None)
        if mri_sequence is None:
            mri_sequence = [seq for seq in mri_sequences if seq.upper() in model_name][0]
        global_max = get_global_max_axis(space=space, mri_sequence=mri_sequence)

        _, mri_org = load_sic_mri(_sic=sic, mri_sequence=mri_sequence, bm=True, norm=True,
                                  regis=space == "mni", raw=space == "raw", as_numpy=False)

        # Check whether original MRI is there
        if mri_org is None:
            if logging:
                with open("./logs/log_data_issue.txt", "r+") as file:  # r+ read & write/append mode
                    for line in file:
                        if sic in line:
                            # print(f"{sic} is already in log file!")
                            break
                    else:
                        file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {model_name}, {sic}\n")
            return None

        # Prepare saving path
        if save:
            # Create parent dirs if not there
            if not os.path.exists("/".join(p2file.split("/")[:-1])):
                os.makedirs("/".join(p2file.split("/")[:-1]))

        # Create analyser object if necessary
        if analyzer_obj is None:

            # Check whether it was computed before:
            if os.path.isfile(p2org):
                analyzer_obj = load_obj(name=p2org.split("/")[-1], folder="/".join(p2org.split("/")[:-1]),
                                        functimer=False)

            else:
                analyzer_obj = analyze_model(mri=prune_mri(x3d=mri_org.get_fdata(),
                                                           # here we assume pruning
                                                           make_cube=True,
                                                           max_axis=global_max).reshape(
                    [1, *[global_max] * 3, 1]),
                    analyzer_type=analyzer_type,
                    model_=load_trained_model(model_name=model_name),
                    norm=False,  # can be normalized later, too
                    **kwargs)  # neuron_selection (for classification)

                # Save also original heatmap for given model
            save_obj(obj=analyzer_obj, name=p2org.split("/")[-1], folder="/".join(p2org.split("/")[:-1]),
                     as_zip=True, functimer=False)

        # Check whether MRI was pruned
        if analyzer_obj.shape[0] == global_max:

            fullsize_a = reverse_pruning(original_mri=mri_org.get_fdata(),
                                         pruned_mri=prune_mri(x3d=mri_org.get_fdata(),
                                                              make_cube=True,
                                                              max_axis=global_max),
                                         pruned_stats_map=analyzer_obj, via_affine=False)

            # print("Shape of re-sized heatmap:", fullsize_a.shape)

        else:
            fullsize_a = analyzer_obj

        # Create NifTi version out of it:
        fullsize_a_nii = nb.Nifti1Image(dataobj=fullsize_a, affine=mri_org.affine)

        # Save
        if save:
            fullsize_a_nii.to_filename(p2file)

    return fullsize_a_nii


def create_heatmap_surface(sic, p2hm_nii, return_output=False):
    """
    Use here freesurfer's mri_vol2surf for both hemispheres, e.g. via nipype:
    ```
    mri_vol2surf --mov MY_HEATMAP.nii --o lh.MY_HEATMAP.mgz --regheader SIC --hemi lh  (and for rh)
    ```

    From nipype documentation:
    * https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.freesurfer.utils.html

    Activate freesurfer environment before running this script (if necessary)
    """

    if check_system() != "MPI":
        raise EnvironmentError("Function works only on MPI servers")

    import nipype.interfaces.freesurfer as fs

    return_output_ls = []

    for fsaverage in [False, True]:

        for hemi in ["lh", "rh"]:
            sampler = fs.SampleToSurface(hemi=hemi)
            sampler.inputs.subjects_dir = "/data/pt_life_freesurfer/freesurfer_all/"  # subjects directory
            sampler.inputs.source_file = p2hm_nii  # --mov
            sampler.inputs.reg_header = True  # --regheader
            sampler.inputs.subject_id = sic  # --regheader SIC
            sampler.inputs.sampling_method = "point"  # 'average' , 'max'
            sampler.inputs.sampling_range = 0  # default in FS, should be only for average
            sampler.inputs.sampling_units = "frac"
            sampler.inputs.out_file = f"{p2hm_nii.rstrip('.nii')}" \
                                      f"{'_fsavg' if fsaverage else ''}_{hemi}.mgz"  # --o
            if fsaverage:
                sampler.inputs.target_subject = "fsaverage"

            res = sampler.run()

            # Collect outputs
            return_output_ls.append(res)

    if return_output:
        return return_output_ls

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
