# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply LRP on (trained) MRInet

Sebastian L.:
    Use LrpPreset*-Analyzer for decomposition (current best practice).
    That is, alpha-beta for conv-layer, epsilon for dense-layer,
    optional flat-LRP for invariance w.r.t. scaling in input/lowest layer

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2019, 2020
"""
# %% Import
# import sys
# import os
# sys.path.append((os.path.abspath(".").split("DeepAge")[0] + "DeepAge/Analysis/Modelling/MRInet/"))

import matplotlib.pyplot as plt
import nilearn as nl
import seaborn as sns
from scipy import stats

from LRP import (apply_analyzer_and_plot_heatmap, analyze_model, plot_heatmap, create_heatmap_nifti,
                 p2intrprt_rslts)
from load_mri_data import (load_raw_study_table, get_mni_template, get_global_max_axis, prune_mri,
                           pred_classes, mri_sequences, age_of_sic, pd, nb)
from meta_functions import *
from train_kerasMRInet import (load_trained_model, get_model_data, get_sub_ensemble_predictions,
                               load_datasplit, get_target, get_classes, is_binary_classification,
                               crop_model_name)
from transformation import get_list_ants_warper, file_to_ref_orientation

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Set paths
p2keras = "./processed/Keras/"
p2logs = p2keras + "logs/"
p2pred = p2keras + "Predictions/"
p2interpret = p2keras + "Interpretation/"


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Functions for __main__:

def has_average_plots(_model_name):
    """
    Check whether average heatmap plots for given model are already there:
    We want for each class:
        * correctly classified       2
        * incorrectly classified   + 2
    :param _model_name: name of model
    :return: True if the corresponding 4 plots are already created
    """
    # sum(["_S_all_" in fi for fi in os.listdir(f"{p2interpret}{model.name}/")]) == 4:
    ctn = 0
    for fn in os.listdir(f"{p2interpret}{_model_name}/"):
        if "_S_all_" in fn:
            ctn += 1
    return ctn >= 4


def mean_confidence_interval(data, axis=None, confidence=0.95):
    m = np.mean(data, axis=axis)
    se = stats.sem(1. * data, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., len(m) - 1)
    return m, m - h, m + h


def plot_prediction(_model, xdata, ydata, predictions=None, _submodel=None, big_fs=False):
    # TODO also enable bigger FS for other plots

    # Get SIC data
    exclusion = True
    tab_name = f"../../../Data/subject_tables/sic_tab{'_reduced' if exclusion else ''}.csv"
    sics_split = load_datasplit(_model.name)
    mri_tab = pd.read_csv(tab_name, index_col=0)
    mri_tab.rename(columns={"SIC_FS": "SICs", "AGE_FS": "age"}, inplace=True)
    mri_tab.set_index("SICs", inplace=True)
    _target = get_target(_model.name)
    _classes = get_classes(_model.name)

    # # Calculate model-performance
    if is_binary_classification(_model.name):

        from sklearn.metrics import classification_report, confusion_matrix

        # TODO Check code from website
        #  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        #  """ from sklearn import metrics
        #  import pandas as pd
        #  from ggplot import *
        #  #note in kera model.predict() will return predict probabilities
        #  pred_prob =  model.predict(Xtest, verbose=0)
        #  #preds = clf.predict_proba(Xtest)[:,1]
        #  fpr, tpr, threshold = metrics.roc_curve(ytest, pred_prob)
        #  roc_auc = metrics.auc(fpr, tpr)
        #  method I: plt
        #  import matplotlib.pyplot as plt
        #  plt.title('Receiver Operating Characteristic')
        #  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        #  plt.legend(loc = 'lower right')
        #  plt.plot([0, 1], [0, 1],'r--')
        #  plt.xlim([0, 1])
        #  plt.ylim([0, 1])
        #  plt.ylabel('True Positive Rate')
        #  plt.xlabel('False Positive Rate')
        #  plt.show()"""

        # # Get model predictions
        # TODO adapt for ensemble:
        m_pred = _model.predict(x=xdata) if predictions is None else predictions  # takes a while
        correct = (np.argmax(m_pred, axis=1) == np.argmax(ydata, axis=1)) * 1
        accuracy = sum(correct) / len(correct)
        # accuracy = _model.evaluate(xdata, ydata, batch_size=1)[1]  # , verbose=2)
        # cprint(f'test acc: {accuracy:.3f}', 'y')

        # # Save individ predics (added: 'true-y' == [_target'_categ'] to doublecheck [can be del])
        pred_per_sic = pd.DataFrame(data=list(zip(pd.Categorical.from_codes(np.argmax(m_pred,
                                                                                      axis=1),
                                                                            categories=_classes),
                                                  pd.Categorical.from_codes(np.argmax(ydata,
                                                                                      axis=1),
                                                                            categories=_classes))),
                                    index=sics_split["test"], columns=["pred", "true-y"])
        pred_per_sic.index.name = "SICs"

        # Adapt table
        mri_tab = mri_tab[mri_tab.index.isin(pred_per_sic.index)]  # rm SICs not in test-set
        mri_tab[f"{_target}_categ"] = pd.cut(mri_tab[_target],
                                             bins=[0, 29.131 if _target == "bmi" else 60, 99],
                                             labels=_classes)
        mri_tab = mri_tab.join(pred_per_sic, how="outer")  # merge tables
        mri_tab["pred_correct"] = mri_tab["pred"] == mri_tab["true-y"]

        # # Classification report
        y_true = np.argmax(ydata, axis=1)
        y_pred = np.argmax(m_pred, axis=1)
        cprint(classification_report(y_true, y_pred, target_names=_classes), 'y')
        # Precision: tp/(tp+fp); tp: n_true-positives & fp: n_false-positives
        # Precision is (intuitively): ability of model not to label a sample as pos that is neg
        # Recall (=sensitivity): tp/(tp+fn); fn: n_false-negatives.
        # Recall is (intuitively) the ability of the classifier to find all the positive samples.
        # See also: Specificity (true negative-rate: tn/(tn+fp)) and others ...

        # Create confusion matrix
        confmat = confusion_matrix(y_true, y_pred, normalize='true')  # normalize=None

        # # Plot confusion matrix
        df_confmat = pd.DataFrame(data=confmat, columns=_classes, index=_classes)
        df_confmat.index.name = 'TrueValues'
        df_confmat.columns.name = f'Predicted {_target.upper()}'
        _fig = plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.4)  # for label size
        _ax = sns.heatmap(df_confmat, cmap="Blues", annot=True,
                          annot_kws={"size": 16})  # "ha": 'center', "va": 'center'})
        _ax.set_ylim([0, 2])  # because labelling is off otherwise, OR downgrade matplotlib==3.1.0
        _fig.savefig(p2pred + f"{_model.name}_confusion-matrix.png")
        plt.close()
        # cprint(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred, normalize='true')}", 'b')

        # TODO >>>>> >>>>> >>>>> >>>>> >>>>> TODO >>>>> >>>>> >>>>> >>>>> >>>>> TODO >>>>> >>>>> >
        # # TODO first plot: Brain-AGE-DIFF ...
        # TODO <<<<< <<<<< <<<<< <<<<< <<<<< TODO <<<<< <<<<< <<<<< <<<<< <<<<< TODO <<<<< <<<<< <

    else:
        # # Get model predictions
        pred_model = _model if _submodel is None else _submodel
        m_pred = pred_model.predict(x=xdata) if predictions is None else predictions  # takes a while

        mae = np.absolute(ydata - m_pred[:, 0]).mean()

        # # Save individual predics (we add 'true-y' == ['_target'] to double check [could be del])

        # # Jointplot
        if _submodel is not None:
            if not os.path.isdir(p2pred + _model.name):
                os.mkdir(p2pred + _model.name)

        plot_path = p2pred + f"{_model.name if _submodel is None else _submodel.name}_" \
                             f"predictions_MAE={mae:.2f}.png"

        sns.jointplot(x=m_pred[:, 0], y=ydata, kind="reg", height=10,
                      marginal_kws=dict(bins=int(round((ydata.max() - ydata.min()) / 3))),
                      xlim=(np.min(m_pred) - 10, np.max(m_pred) + 10),
                      ylim=(np.min(ydata) - 10, np.max(ydata) + 10)).plot_joint(
            sns.kdeplot, zorder=0, n_levels=6).set_axis_labels("Predictions",
                                                               f"True-{_target.upper()}")

        if big_fs:
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(f"Predicted {_target.lower()}", fontsize=20)
            plt.ylabel(f"Chronological {_target.lower()}", fontsize=20)

            plot_path = plot_path.replace(".png", "_bigFS.png")

        plt.tight_layout()
        plt.savefig(plot_path)  # Save plot
        plt.close()

        # # Residuals
        # sns.jointplot(m_pred[:, 0], ydata, kind="resid").set_axis_labels("Predictions",
        #                                                                   "residuals")
        # plt.tight_layout()

        plot_path = p2pred + f"{_model.name if _submodel is None else _submodel.name}_" \
                             f"prediction-residuals_MAE={mae:.2f}.png"

        _fig = plt.figure(f"{_target.title()} Prediction Model Residuals MAE={mae:.2f}",
                          figsize=(10, 8))
        _ax2 = _fig.add_subplot(1, 1, 1)
        _ax2.set_title(f"Residuals w.r.t. {_target.upper()} (MAE={mae:.2f})")
        rg_ = sns.regplot(x=ydata, y=m_pred[:, 0] - ydata)
        # sns.lmplot(x="true-y", y="pred_diff", data=mri_tab, order=5)  # poly-fit
        plt.hlines(y=0, xmin=min(ydata) - 3, xmax=max(ydata) + 3, linestyles="dashed", alpha=.5)
        plt.vlines(x=np.median(ydata), ymin=min(m_pred[:, 0] - ydata) - 2,
                   ymax=max(m_pred[:, 0] - ydata) + 2,
                   linestyles="dotted", color="red", alpha=.5, label="median test set")
        _ax2.set_xlabel(f"True-{_target.upper()}")
        _ax2.set_ylabel("Prediction Error (pred-t)")
        _ax2.legend()

        if big_fs:
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(f"Predicted {_target.lower()}", fontsize=20)
            plt.ylabel(f"Prediction Error (pred-t)", fontsize=20)

            plot_path = plot_path.replace(".png", "_bigFS.png")

        plt.tight_layout()
        _fig.savefig(plot_path)  # Save
        plt.close()

        # TODO >>>>> >>>>> >>>>> >>>>> >>>>> TODO >>>>> >>>>> >>>>> >>>>> >>>>> TODO >>>>> >>>>> >
        # TODO first plots: big-Brain-AGE-DIFF
        done = False
        if done:

            # Load data
            pred_per_sic = pd.DataFrame(data=list(zip(m_pred.flatten(), ydata)),
                                        index=sics_split["test"], columns=["pred", "true-y"])

            pred_per_sic.index.name = "SICs"
            # Adapt table
            mri_tab = mri_tab[mri_tab.index.isin(pred_per_sic.index)]  # rm SICs not in test-set
            mri_tab = mri_tab.join(pred_per_sic, how="outer")  # merge tables
            labels = pred_classes(_target)
            labels.insert(1, "middle")
            mri_tab[f"{_target}_categ"] = pd.cut(mri_tab[_target],
                                                 bins=[0,
                                                       (24. if _target == "bmi" else 46),
                                                       (29.130 if _target == "bmi" else 59), 99],
                                                 labels=labels)
            mri_tab["pred_diff"] = mri_tab["pred"] - mri_tab["true-y"]

            # Regress data (out)
            from sklearn.linear_model import LinearRegression
            reg_model = LinearRegression()
            reg_model.fit(X=mri_tab["true-y"].to_numpy().reshape((-1, 1)),
                          y=mri_tab["pred_diff"].to_numpy().reshape((-1, 1)))
            test_y = reg_model.intercept_ + np.arange(18, 82 + 1, 1) * reg_model.coef_

            for ord in range(1, 6):
                sns.lmplot(x="true-y", y="pred_diff", data=mri_tab, order=ord)  # poly-fit
                plt.title(f"Order {ord}")
                plt.plot(np.arange(18, 82 + 1, 1), test_y.flatten())

            # Compare
            # sns.lmplot(x="true-y", y="pred_diff", data=mri_tab, order=1)  # same as rg_
            rg_ = sns.regplot(x="true-y", y="pred_diff", data=mri_tab)
            plt.plot(np.arange(18, 82 + 1, 1), test_y.flatten())

            # Regress bias out

            mod_m_pred = mri_tab.pred.to_numpy() - (
                    mri_tab.pred.to_numpy() * reg_model.coef_ + reg_model.intercept_)
            mri_tab["pred_modified"] = mod_m_pred.flatten()
            mri_tab["pred_diff_modified"] = mri_tab["pred_modified"] - mri_tab["true-y"]
            sns.regplot(x="true-y", y="pred_diff_modified", data=mri_tab)

            # Pred-diff vs. target
            # mri_tab["sex"] = mri_tab["sex"].astype("category")
            sns.lmplot(x=_target, y="pred_diff", hue="sex", data=mri_tab)
            plt.tight_layout()
            plt.savefig(f"TEST5cont_target-{_target}-diff-{_target}-sex_model-"
                        f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                        f"MAE={mae:.2f}.png")
            plt.close()

            sns.lmplot(x="bmi", y="pred_diff", hue="sex", data=mri_tab)
            plt.tight_layout()
            plt.savefig(
                f"TEST5cont_target-{_target}-diff-bmi-sex_model-"
                f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                f"MAE={mae:.2f}.png")
            plt.close()

            sns.violinplot(x="smoking_status_curated", y="pred_diff", hue="sex", split=True, data=mri_tab)
            plt.hlines(y=0, xmin=-.5, xmax=2.5, linestyles="dashed", alpha=.5)
            plt.savefig(f"TEST5cont_target-{_target}-diff-sex-smoking_model-"
                        f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                        f"MAE={mae:.2f}.png")
            plt.close()

            sns.violinplot(x="smoking_status_curated", y="pred_diff", hue=f"{_target}_categ",
                           data=mri_tab)
            # int_tab = mri_tab.loc[mri_tab[f"{_target}_categ"] == "old"]
            # int_tab.age_categ = "old"
            # sns.violinplot(x="smoking_status_curated", y="pred_diff", hue=f"{_target}_categ",
            #                data=int_tab)
            # sns.violinplot(x="smoking_status_curated", y="pred_diff_modified", hue=f"{_target}_categ",
            #                data=int_tab)
            # sns.violinplot(x="smoking_status_curated", y="pred_diff_modified", hue=f"{_target}_categ",
            #                data=mri_tab)
            plt.hlines(y=0, xmin=-.5, xmax=2.5, linestyles="dashed", alpha=.5)
            plt.savefig(f"TEST5cont_target-{_target}-diff-age-cat-smoking_model-"
                        f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                        f"MAE={mae:.2f}.png")
            plt.close()

            sns.violinplot(x="hypertension_med_y_n", y="pred_diff", hue="sex", split=True, data=mri_tab)
            plt.hlines(y=0, xmin=-.5, xmax=1.5, linestyles="dashed", alpha=.5)
            plt.savefig(
                f"TEST5cont_target-{_target}-diff-sex-hyper_model-"
                f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                f"MAE={mae:.2f}.png")
            plt.close()

            # sns.violinplot(x="hypertension_med_y_n", y="pred_diff", hue="age_categ", data=int_tab)
            # sns.violinplot(x="hypertension_med_y_n", y="pred_diff_modified", hue="age_categ",
            #                data=int_tab)
            sns.violinplot(x="hypertension_med_y_n", y="pred_diff", hue="age_categ", data=mri_tab)
            plt.hlines(y=0, xmin=-.5, xmax=1.5, linestyles="dashed", alpha=.5)
            plt.savefig(f"TEST5cont_target-{_target}-diff-hyper-age-cat_model-"
                        f"{_model.name if _submodel is None else _submodel.name.replace('/', '-')}_"
                        f"MAE={mae:.2f}.png")
            plt.close()
            # TODO Compare to model-bias-corrected data
            plt.figure()
            sns.violinplot(x="hypertension_med_y_n", y="pred_diff_modified", hue="age_categ",
                           data=mri_tab)
            plt.hlines(y=0, xmin=-.5, xmax=1.5, linestyles="dashed", alpha=.5)
            plt.title("Corrected for model-prediction-bias")

        # TODO <<<<< <<<<< <<<<< <<<<< <<<<< TODO <<<<< <<<<< <<<<< <<<<< <<<<< TODO <<<<< <<<<< <


def plot_training_process(_model):
    history_file = f"{p2logs}{_model.name}_history.npy"
    if os.path.isfile(history_file) and not os.path.isfile(history_file.replace(".npy", ".png")):
        _binary_cls = is_binary_classification(_model.name)
        model_history = np.load(f"{p2logs}{_model.name}_history.npy", allow_pickle=True).item()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
        fig.suptitle(f"{_model.name}")
        ax1.plot(model_history['acc' if _binary_cls else 'mean_absolute_error'])
        ax1.plot(model_history['val_acc' if _binary_cls else 'val_mean_absolute_error'])
        ax1.set_title('Performance')
        ax1.set_ylabel('accuracy' if _binary_cls else 'mean absolute error')
        ax1.set_xlabel('training epoch')
        ax1.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        ax2.plot(model_history['loss'])
        ax2.plot(model_history['val_loss'])
        ax2.set_title('Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('training epoch')
        ax2.legend(['train', 'test'], loc='upper left')
        plt.tight_layout()
        fig.savefig(history_file.replace(".npy", ".png"))
        plt.close()


# TODO parallize average_heatmap()
def average_heatmap(grp, grp_name, binary_class, _model, xdata, _target, _anal_type):
    """Create average heatmaps over each class/group"""

    n_grp = len(grp)
    _avg_a = np.zeros(xdata[0, ..., 0].shape)
    _avg_ipt = np.zeros(_avg_a.shape)

    start = datetime.now()
    for ctn, i_ in enumerate(grp):
        i_sub_ = np.array([i_])
        _avg_ipt += analyze_model(mri=xdata[i_sub_], analyzer_type='input', model_=_model,
                                  norm=False)
        _avg_a += analyze_model(
            mri=xdata[i_sub_], analyzer_type=_anal_type, model_=_model, norm=False,
            neuron_selection=({pred_classes(_target)[0]: 0,
                               pred_classes(_target)[1]: 1}[grp_name]) if binary_class else None)

        loop_timer(start_time=start, loop_length=n_grp, loop_idx=ctn, add_daytime=True)

    _avg_ipt /= n_grp
    _avg_a /= n_grp

    # # Normalize to [-1, 1]
    _avg_ipt /= np.max(np.abs(_avg_ipt))
    _avg_a /= np.max(np.abs(_avg_a))

    return _avg_ipt, _avg_a


def load_sic_heatmap(sic: str, model_name: str, mni: bool, aggregated=None,
                     analyzer_type='lrp.sequential_preset_a', pruned_cube=False, verbose=True):
    # Check which kind of model is given
    try:
        if load_trained_model(model_name).is_multilevel_ensemble():  # multi-level ensemble case
            if verbose:
                cprint("Specify the sub-ensemble (for an aggregated heatmap) and/or basemodel "
                       "(for single model heatmap)!", col='r')
            return None
        else:  # Sub-ensemble case
            if not aggregated:
                if verbose:
                    cprint(f"Since no basemodel was specified the aggregated heatmap of '{sic}' for the "
                           f"given ensemble '{model_name}' will be returned", col='y')
                aggregated = True
    except AttributeError:  # base-model case
        aggregated = False
        pass

    # Define file-base name
    fn = f"{analyzer_type}_relevance_maps"

    # Depending on model load different files
    if "MNI" in model_name:
        # *.nii.gz is the file in original MNI space for models trained on MNI images, i.e. not pruned,
        # whereas ".pkl.gz" is also MNI but pruned and (usually) cubified
        fn += "nii.gz" if mni else ".pkl.gz"
    else:
        # *2mni.nii.gz is the warped heatmap to the original MNI space
        fn += "2mni.nii.gz" if mni else ".pkl.gz"

    aggregated = "aggregated" if aggregated else ""
    hm_dir = os.path.join(p2intrprt_rslts, model_name, aggregated, sic)

    # Load heatmap
    if mni:
        hm = nb.load(filename=os.path.join(hm_dir, fn))

        if pruned_cube:  # prune MNI heatmap and reorient to model-training space
            # Note: this now returns a np.array instead of a nii
            hm = file_to_ref_orientation(image_file=hm)
            mnitmp = get_mni_template(reorient=True, prune=False, mask=True)
            hm = prune_mri(x3d=hm.get_fdata() * mnitmp + mnitmp,  # mask & temp. add MNI templated
                           make_cube=True, max_axis=get_global_max_axis(space="mni"))
            hm -= get_mni_template(reorient=True, prune=True, mask=True)  # subtract MNI template again
            # adding & later subtracting MNI template is done to find proper brain-edges while pruning

    else:
        hm = load_obj(name=fn, folder=hm_dir, functimer=False)

    return hm


def create_aggregate_heatmap_per_subject(model_name, analyzer_type='lrp.sequential_preset_a',
                                         sics_list=None, verbose=False):
    """
    Takes [currently] the average heatmap over all basemodels of a given ensemble per given/available
    SIC and saves it. This function looks for both file-formats in MNI space (*.nii.gz) and original space
    that was used for training the model (".pkl.gz").
    :param model_name: name of model
    :param analyzer_type: type of LRP analyzer
    :param sics_list: list of SICs whose heatmaps are to be aggregated OR None: for all available SICs
    :param verbose: verbose or not
    """
    # TODO list
    #  1) delete log line if heatmap computed after all
    #  2) check whether aggregated heatmap already there to not compute it again

    fnb = f"{analyzer_type}_relevance_maps"  # base-filename + ("2mni.nii.gz" if mni else ".pkl.gz")

    ens = load_trained_model(model_name=model_name)

    if sics_list is None:
        sics_list = load_datasplit(model_name)
        sics_list = sics_list["validation"] + sics_list["test"]

    if ens.is_multilevel_ensemble():
        for subens_name in ens.list_of_submodels:
            subens = ens.get_submodel(subens_name)

            create_aggregate_heatmap_per_subject(model_name=subens.name, analyzer_type=analyzer_type,
                                                 sics_list=sics_list)

    else:
        p2hms = os.path.join(p2intrprt_rslts, ens.name)
        p2dest = os.path.join(p2hms, "aggregated")
        n_submodels = len(ens.list_of_submodels)
        if not os.path.isdir(p2dest):
            os.mkdir(p2dest)

        start = datetime.now()
        for i, sic in enumerate(sics_list):

            # Find all paths to heatmap files for given SIC
            avail_hms = list(Path(p2hms).glob(pattern=f"[!aggregated]*/{sic}/{fnb}*"))  # exclude aggr.dir

            # Prep save folder
            sic_save_dir = os.path.join(p2dest, sic)

            # Seperate NIfTI and numpy files
            nii_hms = [None for f in avail_hms if "nii.gz" in str(f)]  # NIfTI MNIs
            np_hms = [None for f in avail_hms if "pkl.gz" in str(f)]  # numpy training format in MNI or FS

            # NifTi (MNI) heatmaps
            if len(nii_hms) > 0 and (len(nii_hms) % n_submodels == 0):  # check if all HMs per bm there
                idx = 0
                for fn in avail_hms:
                    if "nii.gz" in str(fn):
                        fnii = str(fn)
                        nii_hms[idx] = nb.load(fnii)
                        idx += 1

                # Take aggregate [currently only average]:
                avg_nii_hm = nl.image.mean_img(imgs=nb.concat_images(nii_hms))

                # Save it
                if not os.path.isdir(sic_save_dir):
                    os.mkdir(sic_save_dir)
                avg_nii_hm.to_filename(filename=os.path.join(sic_save_dir, fnii.split("/")[-1]))

            else:
                if verbose:
                    cprint(f"{sic} has not all NIfTI heatmaps (MNI) for model '{ens.name}'", col='r')
                # Write in log
                with open(os.path.join(p2dest, "nii_missing_heatmaps.log"), "a") as file:
                    file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {sic} | only "
                               f"{len(nii_hms)}/{n_submodels} [MNI] *nii.gz heatmaps found.\n")

            # Numpy (MNI/FS) heatmaps
            if len(np_hms) > 0 and (len(np_hms) % n_submodels == 0):
                fpkl = None  # init
                avg_np_hm = None
                for fp in avail_hms:
                    if "pkl.gz" in str(fp):
                        # Load sub-model heatmap file
                        fpkl = str(fp).split("/")[-1]
                        temp_hm = load_obj(name=fpkl, folder="/".join(str(fp).split("/")[:-1]),
                                           functimer=False)

                        # Add heatmaps to each other
                        avg_np_hm = temp_hm if avg_np_hm is None else (avg_np_hm + temp_hm)

                # Take aggregate [currently only average]:
                avg_np_hm /= n_submodels  # divide by number of heatmaps

                # Save it
                if not os.path.isdir(sic_save_dir):
                    os.mkdir(sic_save_dir)
                save_obj(obj=avg_np_hm, name=fpkl, folder=sic_save_dir, as_zip=True, functimer=False)

            else:
                if verbose:
                    cprint(f"{sic} has not all numpy heatmaps for model '{ens.name}'", col='r')
                # Write in log
                with open(os.path.join(p2dest, "np_missing_heatmaps.log"), "a") as file:
                    file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {sic} | only "
                               f"{len(np_hms)}/{n_submodels} *pkl.gz heatmaps found.\n")

            # Time it
            loop_timer(start_time=start, loop_length=len(sics_list), loop_idx=i,
                       loop_name=f"Aggregate Heatmaps | {model_name}")


def create_all_heatmaps(model_name, analyzer_type='lrp.sequential_preset_a', subset="test"):
    """
    For given model create all heatmaps of all subjects and save them
    :param model_name: name of model
    :param analyzer_type: type of LRP analyzer
    :param subset: "test", "validation" or "both"
    """

    if is_binary_classification(model_name):
        raise NotImplementedError("create_all_heatmaps() is not adjusted yet for binary classification "
                                  "models!")

    assert subset in ["test", "validation", "both"], "subset must be one of the following: 'test', " \
                                                     "'validation' OR 'both'"

    # Load model
    model = load_trained_model(model_name=model_name)

    # Check whether we have an model ensemble
    if hasattr(model, "is_multilevel_ensemble"):
        model.summary()
        raise ValueError(f"Please specify for which "
                         f"{'sub-ensemble and ' if model.is_multilevel_ensemble() else ''}basemodel "
                         f"relevance maps shall be computed!")

    # Get model data
    if subset == "both":
        _, _, x_val, y_val, x_test, y_test, sics_list = get_model_data(model_or_name=model.name,
                                                                       only_data=True, all_data=True,
                                                                       return_sics=True)

        xdata = np.concatenate([x_val, x_test], axis=0)  # shape (402, 198, 198, 198, 1)
        ydata = np.concatenate([y_val, y_test], axis=0)  # shape (402, )
        del _, x_val, y_val, x_test, y_test
        sics_list = sics_list["validation"] + sics_list["test"]  # len() == 402

    elif subset == "validation":
        _, _, xdata, ydata, _, _, sics_list = get_model_data(model_or_name=model.name, only_data=True,
                                                             all_data=True, return_sics=True)
        sics_list = sics_list[subset]
        # Not always == load_datasplit(model_name=model.name)[subset], e.g. when no MRI for specific SICs

    else:  # subset == "test"
        xdata, ydata, sics_list = get_model_data(model_or_name=model.name, only_data=True, all_data=False,
                                                 return_sics=True)

    assert xdata.shape[0] == len(ydata) == len(sics_list), "Data lengths must be the same!"  # TEST

    # Init relevance dict Or load it
    rel_dict_name = model.name + f"_{analyzer_type}_relevance_maps"
    sicfn = f"{analyzer_type}_relevance_maps.pkl.gz"

    try:
        rel_dict = load_obj(name=rel_dict_name, folder=p2intrprt_rslts)
    except FileNotFoundError:
        rel_dict = {sic: None for sic in sics_list}

    # Run for all subjects
    ctn = -1
    start_time = datetime.now()
    for x_sub, y_sub, sic in zip(xdata, ydata, sics_list):

        ctn += 1

        # Set paths for SIC
        p2sicdir = os.path.join(p2intrprt_rslts, model.name, sic)

        # If data object is loaded then check whether relevance map was computed already
        if (sic in rel_dict.keys() and rel_dict[sic] is not None) or os.path.isfile(
                os.path.join(p2sicdir, sicfn)):
            # TODO TEMPORARY save SICs individually from dict remove part after run through
            if not os.path.isdir(p2sicdir):
                os.makedirs(p2sicdir)
            if not os.path.isfile(os.path.join(p2sicdir, sicfn)):
                save_obj(obj=rel_dict[sic], name=sicfn, folder=p2sicdir, as_zip=True, functimer=False)
                print(f"{str(ctn).zfill(3)} {os.path.join(model_name, sic)}", end="\r")  # TODO testing
            continue

        # Compute relevance maps
        rel_map = analyze_model(mri=x_sub[np.newaxis, :], analyzer_type=analyzer_type, model_=model,
                                norm=False,  # Important! Can be done later via: rel_map/np.max(rel_map)
                                neuron_selection=None)  # TODO implement for binary class, too

        # Save some heatmap plots for validation
        if ctn % 50 == 0:
            plot_heatmap(sub_nr=sic, t=y_sub, t_y='', ipt=x_sub.squeeze(),
                         analyzer_obj=rel_map.copy(),
                         analyzer_type=analyzer_type,
                         fix_clim=0.02,
                         fn_suffix=f"_TEST_HM_{sic}",
                         save_plot=True,
                         save_folder=os.path.join(p2intrprt_rslts, model.name),
                         suptitle=f"Subject {sic}: {model.name}")

        # # Save each SIC individiually
        if not os.path.isdir(p2sicdir):
            os.makedirs(p2sicdir)
        if not os.path.isfile(os.path.join(p2sicdir, sicfn)):
            save_obj(obj=rel_map, name=sicfn, folder=p2sicdir, as_zip=True,
                     functimer=False)  # to suppress funct-timing output (since very fast here)

        # Looping
        if (ctn + 1) % 10 == 0:
            loop_timer(start_time=start_time, loop_length=len(sics_list), loop_idx=ctn,
                       loop_name=f"Compute Relevance Maps for '{model.name}'", add_daytime=True)

    # TODO if still required to save dict do it at the end after loading single files
    #  can be additional function
    # if ((ctn + 1) % 50 == 0) or (ctn == len(sics_list) - 1):  # TODO ADAPT
    #     save_obj(obj=rel_dict, name=rel_dict_name,
    #              folder=p2intrprt_rslts, hp=True, as_zip=True)
    # Zipping for about 200 subjects takes about 3-4 min on dalmatiner


# TODO explore relevance distrubtions after MNI-transformation in z_playground.py
@only_mpi
def heatmaps2mni(model_name, analyzer_type='lrp.sequential_preset_a', save_original_hm_as_nifti=False,
                 subset="test", sics_list=None):
    """
    Function warps LRP-heatmaps to MNI space for given model.
    NOTE: The warping effects the overall distribution of relevance on subject level.
    This effect needs to be carefully explored, and results should be taken with caution, especially when
    heatmaps are compared across subjects.
    :param model_name: name of model
    :param analyzer_type: type of LRP analyzer [Default: 'lrp.sequential_preset_a']
    :param save_original_hm_as_nifti: True: save original (i.e. none-MNI) heatmap as NifTi file
    :param subset: "test", "validation", "both", OR None
    :param sics_list: if a list of SICs is given, then only for these SICs heatmaps will be warped to MNI
    """

    # Imports
    from mri_registration import p2data, ants

    # MNI T1-filename
    fn_t1_mni = "T1_brain2mni.nii.gz"
    rel2mni_filename = fn_t1_mni.replace("T1_brain", f"{analyzer_type}_relevance_maps")
    relorg_filename = f"{analyzer_type}_relevance_maps.pkl.gz"  # heatmap in original space

    # # Load original heatmaps of given model (on testset)
    # rel_dict_name = model_name + f"_{analyzer_type}_relevance_maps"  # OLD
    # rel_dict = load_obj(name=rel_dict_name, folder=p2intrprt_rslts)  # OLD

    # Get SICs of subset(s)
    assert subset in ["test", "validation", "both", None], \
        "subset must be one of the following: 'test', 'validation' OR 'both' if sics_list not given"

    if sics_list is None:
        sics_list = load_datasplit(model_name=model_name)
        sics_list = (sics_list["validation"] + sics_list["test"]) if subset == "both" else \
            sics_list[subset]

    # Get MRI data configurations
    seq = [seq for seq in mri_sequences if seq.upper() in model_name][0]
    mri_space = "mni" if "MNI" in model_name else "raw" if "RAW" in model_name else "fs"

    if mri_space == "mni":
        cprint("Given model was trained on MRIs in MNI-space. Hence, there is no warping of relevance "
               "maps required.", 'y')
        # return None
    elif mri_space == "raw":
        raise NotImplementedError("Given model was trained on MRIs in raw-space. The current "
                                  "implementations of heatmaps2mni() and create_heatmap_nifti() assume"
                                  "that training data was pruned & cubified. Revisit these functions if"
                                  "you want to proceed!")

    start_time = datetime.now()
    for i, sic in enumerate(sics_list):

        # Prepare folder and filename of MNI-heatmap per subject
        p2savedir = os.path.join(p2intrprt_rslts, model_name, sic)
        if not os.path.exists(p2savedir):
            os.makedirs(p2savedir)
        p2savefile = os.path.join(p2savedir, rel2mni_filename)

        # See if file was already computed
        if os.path.isfile(p2savefile):
            continue

        else:
            # Resize & transform original heatmap to T1-native space
            if not os.path.isfile(os.path.join(p2savedir, relorg_filename)):  # path 2 original HM file
                cprint(f"No relevance map in {p2savedir} !", col='y')
                rel_map = None
            else:
                try:  # TODO REMOVE try-except after run-through (was only for single issue)
                    rel_map = load_obj(name=relorg_filename, folder=p2savedir,
                                       functimer=False)  # switch function timer off
                except Exception:
                    cprint(f"Issue with loading {sic} in {p2savedir}", 'r')
                    _ = input("Press Enter to continue: ")
                    continue

                # TODO Update create_heatmap_nifti() for mri_space=='raw' & pruning/cubify -issue
                assert rel_map.shape[0] == rel_map.shape[1] == rel_map.shape[2], \
                    "Apparently, model was trained on non-cubified data. Update heatmaps2mni() to proceed"

            r_nifti = create_heatmap_nifti(sic=sic, model_name=model_name, analyzer_type=analyzer_type,
                                           analyzer_obj=rel_map, mri_sequence=seq, space=mri_space,
                                           save=save_original_hm_as_nifti, logging=True)

            if r_nifti is None:
                cprint(f"There is an heatmap issue for {sic} in {model_name}\n", 'r')
                continue

            # Here we compute the MNI-heatmap NifTis for missing SICs
            if mri_space != "mni":
                # # Transform to MNI
                # Get T1 in MNI space as fixed reference & corresponding warping file:
                t1_mni = nb.load(filename=f"{p2data}{sic}/" + fn_t1_mni)
                # load_sic_mri(regis=True) has wrong orientation for this case

                # Load transformation/warping file:
                mnitx = get_list_ants_warper(folderpath=f"{p2data}{sic}/" + "transforms2mni",
                                             inverse=False)

                if mri_space == "raw":
                    # Add raw-to-t1/fs-space transformation
                    mnitx += get_list_ants_warper(folderpath=f"{p2data}{sic}/{seq}/", inverse=False,
                                                  only_linear=True)
                    # We add it to the end of transform-list, since the last transform-operation will be
                    # executed first (as in a stack), see also : https://github.com/ANTsX/ANTs/issues/531

                # Apply forward warping on given heatmap to MNI-space
                heatmap_mni = ants.apply_transforms(fixed=ants.from_nibabel(t1_mni),
                                                    moving=ants.from_nibabel(r_nifti),
                                                    transformlist=mnitx,
                                                    verbose=False)

                # Save
                heatmap_mni.to_file(filename=p2savefile)
                # heatmap_mni = heatmap_mni.to_nibabel()  # transform into nibabel

        # Looper
        loop_timer(start_time=start_time, loop_length=len(sics_list), loop_idx=i,
                   loop_name=f"Heatmaps-2-MNI | '{model_name}'", add_daytime=True)


def concatenate_hm_nii(model_name, sic_list, analyzer_type='lrp.sequential_preset_a'):
    """
    Create 4D NifTi image with all heatmaps of given list of SICs
    :param model_name: name of model which produced the heatmaps
    :param sic_list: list of SICs
    :param analyzer_type: type of LRP analyzer
    :return: 4D NIfTI image
    """
    concat_hms_nii = [None] * len(sic_list)
    for i, sic in enumerate(sic_list):
        concat_hms_nii[i] = load_sic_heatmap(sic=sic, model_name=model_name,
                                             mni=True,  # imgs must share same space & must be NIfTIs
                                             aggregated=None,  # fct finds out whether necessary
                                             analyzer_type=analyzer_type, verbose=False)

    return nb.concat_images(concat_hms_nii)


def concatenate_sic_hm_nii(model_name: str, sic: str, mni: bool, analyzer_type='lrp.sequential_preset_a'):
    """
    Create 4D NifTi image with all heatmaps of given SIC across the basemodel of the given sub-ensemble
    :param model_name: name of model which produced the heatmaps [must be sub-ensemble]
    :param sic: SIC
    :param mni: heatmaps in MNI format: True/False
    :param analyzer_type: type of LRP analyzer
    :return: 4D NIfTI image
    """

    sub_ens = load_trained_model(model_name=model_name)

    assert not sub_ens.is_multilevel_ensemble(), \
        "Given model can't be a multi-level ensemble. Provide name of sub-ensemble instead!"

    n_bm = len(sub_ens.list_of_submodels)  # N basemodels

    concat_hms_nii = [None] * n_bm

    for i, bm_name in enumerate(sub_ens.list_of_submodels):

        bmn = crop_model_name(bm_name)

        bm_hm = load_sic_heatmap(sic=sic, model_name=os.path.join(sub_ens.name, bmn), mni=mni,
                                 aggregated=False, analyzer_type=analyzer_type, verbose=True)

        if not mni:
            bm_hm = create_heatmap_nifti(sic=sic, model_name=os.path.join(sub_ens.name, bmn),
                                         analyzer_type=analyzer_type, analyzer_obj=bm_hm,
                                         save=False,  # consider as arg above
                                         logging=False)
            # Note: This reverses pruning of the HM to original MRI space (e.g. 198**3 => 256**3 for 'fs')

        concat_hms_nii[i] = bm_hm

    return nb.concat_images(concat_hms_nii)


def create_continuum_gif(_model, analyse_type='lrp.sequential_preset_a', wrt_prediction=True, morph=True):
    """
    Creates LRP plots of a given order of subject. Either the order is w.r.t.
        1) the model prediction (default), or
        2) the ground-truth (wrt_prediction=False)
    See for shell gif command: http://www.imagemagick.org/Usage/anim_mods/
    :param _model: prediction model
    :param analyse_type: type of model analyzer
    :param wrt_prediction: True: sorts given subject w.r.t. prediction, otherwise w.r.t. ground-truth
    :param morph: True: creates animation with morph between different brains (takes time & memory)
    """
    assert not is_binary_classification(_model.name), \
        "Can only create continuum-gif for regression models"
    assert "MNI" in _model.name, "Continuum-GIFs make only sense on registered data (MNI)."

    _model, xtest, ytest, _target, _, _binary_cls, _, _mri_type, _mri_space = get_model_data(
        model_or_name=_model)
    m_pred = _model.predict(x=xtest)

    # Sort arrays
    sort_idx = np.argsort(m_pred.flatten()) if wrt_prediction else np.argsort(ytest)
    m_pred = m_pred[sort_idx]
    xtest = xtest[sort_idx]
    ytest = ytest[sort_idx]
    _subjects = range(len(xtest))

    save_folder = _model.name + f"_{'prediction' if wrt_prediction else 'groundtruth'}-continuum_GIF"

    # Apply analyzer and plot heatmaps (if necessary):
    if not os.path.exists(p2interpret + save_folder):
        os.mkdir(p2interpret + save_folder)

    if sum(["mid-slice" in f for f in os.listdir(p2interpret + save_folder)]) != len(_subjects):

        # Create (super-) titles for LRP plots
        suptitles = [f"predicted-{_target} = {m_pred[i, 0]:.2f} | true-{_target} = {ytest[i]}"
                     for i in _subjects]

        apply_analyzer_and_plot_heatmap(subjects=_subjects, mris=xtest, targets=ytest, pre_model=_model,
                                        ls_analy_type=analyse_type, binary_cls=_binary_cls,
                                        fix_clim=0.02, ls_suptitles=suptitles,
                                        neuron_analysis=_binary_cls, classes=None,
                                        save_folder=save_folder)

        # Rename files
        ctn = sum([".png" in file for file in os.listdir(p2interpret + save_folder)])  # count N of images

        for file in os.listdir(p2interpret + save_folder):
            if ".png" in file:
                oldfn = file
                newfn = "MRI_mid-slice" + file.lstrip(analyse_type + "_S").split("_")[0].zfill(
                    oom(ctn) + 1) + ".png"
                os.rename(os.path.join(p2interpret, save_folder, oldfn),
                          os.path.join(p2interpret, save_folder, newfn))

    # Create Gif
    gif_name = f"{os.path.join(p2interpret, save_folder)}/{_model.name}{'_morph' if morph else ''}.gif"
    if not os.path.isfile(gif_name):
        cprint("Create gif animation from sorted LRP heatmaps ... [lengthy process]", 'y')
        os.system(f"convert -delay 20 {'-morph 5 ' if morph else ''}-loop 0 "
                  f"{os.path.join(p2interpret, save_folder)}/MRI_mid-slice*.png {gif_name}")
        cprint(f"Animation job is done! Find it at {gif_name}!", 'b')


@only_mpi
def create_hm_nii_for_sic_stats(model_name: str, sic: str, mni: bool, name_of_test="across_basemodels",
                                analyzer_type='lrp.sequential_preset_a', absolute_hm=True,
                                create_fsl_bash=True):
    sub_ens = load_trained_model(model_name=model_name)

    assert not sub_ens.is_multilevel_ensemble(), \
        "Given ensemble can't be multi-level ensemble. Specify sub-ensemble!"

    # Check what kind of test:
    against_zero = True  # one-sample t-test
    # TODO could do also greater and smaller than zero tests for non-absoulte_hm
    #  check at 15:20 min: https://www.youtube.com/watch?v=Ukl1VWobviw
    if against_zero and not absolute_hm:
        cprint("For test against zero we need to take the absolute of each heatmaps!", col='y')
        absolute_hm = True

    # Prep file-path
    warp = "MNI" not in model_name
    rel_filename = f"{analyzer_type}_relevance_maps{'2mni' if warp else ''}.nii.gz"

    # TODO adapt groups if no data e.g. check via Path.glob() whether data there
    # for sic in group_a_sics:
    #     x = load_sic_heatmap(sic=sic, model_name=model_name, mni=True, aggregated=True,
    #                          analyzer_type=analyzer_type, verbose=True)
    #     print(x.shape)
    #
    #     get
    #     find(fname=rel_filename, folder=)
    pass

    grp_suffix = f"{sic}_{name_of_test}_{len(sub_ens.list_of_submodels)}-0"
    grp_fname = rel_filename.replace(".nii.gz", f"_{grp_suffix}.nii.gz")
    p2save = os.path.join(p2intrprt_rslts, model_name, "statistics")
    # p2save = os.path.join(p2intrprt_rslts, model_name, grp_name)

    # Concat HMs of SIC
    done = None
    while True:
        try:
            bms_hm_nii = concatenate_sic_hm_nii(model_name=model_name, sic=sic, mni=mni,
                                                analyzer_type=analyzer_type)  # niftis of basemodels

            break
        except FileNotFoundError:
            # TODO this is still for single base-models
            if done:
                break
            heatmaps2mni(model_name=model_name, analyzer_type=analyzer_type,
                         save_original_hm_as_nifti="MNI" in model_name, subset=None,
                         sics_list=[sic])
            done = True

    # Take absolute if requested
    if absolute_hm:
        bms_hm_nii = nb.Nifti1Image(dataobj=nb.casting.int_abs(bms_hm_nii.get_fdata()),
                                    affine=bms_hm_nii.affine, header=bms_hm_nii.header)

    # Save concatenated groups as nii
    if not os.path.isdir(p2save):
        os.mkdir(p2save)
    bms_hm_nii.to_filename(filename=os.path.join(p2save, grp_fname))

    # Create FSL bash file if needed
    if create_fsl_bash:
        bash_fpath = os.path.join(p2save, f"run_randomise_{grp_suffix}.sh")

        # One-sample t-test
        with open(bash_fpath, "w") as file:
            file.write(f"mkdir {grp_suffix}\n"
                       f"randomise -i '{grp_fname}' \\\n"
                       f"\t\t\t\t\t-o {grp_suffix}/{grp_suffix} \\\n"
                       f"\t\t\t\t\t-1 \\\n"
                       f"\t\t\t\t\t-n 5000 \\\n"
                       f"\t\t\t\t\t-T\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                       f"--thresh=0.95 > {grp_suffix}/cluster_tstat1.tsv\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                       f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat1_mm.tsv")

        cprint(f"Execute '{bash_fpath.split('/')[-1]}' in FSL environment within this folder:\n"
               f"'{'/'.join(bash_fpath.split('/')[:-1])}'", 'b')


@only_mpi
def create_hm_nii_for_group_stats(name_of_test: str, model_name: str, group_a_sics, group_b_sics=None,
                                  analyzer_type: str = 'lrp.sequential_preset_a', absolute_hm=False,
                                  create_fsl_bash: bool = True, auto_adapt_groups: bool = False):
    # TODO could do also greater and smaller than zero tests for non-absoulte_hm
    #  check at 15:20 min: https://www.youtube.com/watch?v=Ukl1VWobviw

    # Check what kind of test:
    against_zero = True if group_b_sics is None else False  # against_zero: one-sample t-test
    group_b_sics = [] if against_zero else group_b_sics
    if against_zero and not absolute_hm:
        cprint("For test against zero we need to take the absolute of each heatmaps!", col='y')
        absolute_hm = True

    # Prep file-path
    warp = "MNI" not in model_name
    rel_filename = f"{analyzer_type}_relevance_maps{'2mni' if warp else ''}.nii.gz"

    # Adapt groups if no data
    if auto_adapt_groups:
        if not hasattr(load_trained_model(model_name=model_name), "is_multilevel_ensemble"):
            raise NotImplemented("auto_adapt_groups not implemented for basemodels yet!")
        else:
            available_hms = find(fname=rel_filename, folder=os.path.join(p2intrprt_rslts,
                                                                         model_name,
                                                                         "aggregated"),
                                 typ="file", exclusive=False, verbose=False)

            # Filter SICs out of groups with no HMs
            old_n_a = len(group_a_sics)
            group_a_sics = list(filter(lambda _x: any([_x in y for y in available_hms]), group_a_sics))
            # == [sic for sic in group_a_sics if any(sic in _sic for _sic in available_hms)]
            if len(group_a_sics) < old_n_a:
                cprint(f"For {model_name} not all HMs of group A are available. Group size shrank to "
                       f"{len(group_a_sics)} of initially {old_n_a}!", 'y')

            if group_b_sics is not None:
                old_n_b = len(group_b_sics)
                group_b_sics = list(filter(lambda x: any([x in y for y in available_hms]), group_b_sics))
                if len(group_b_sics) < old_n_b:
                    cprint(f"For {model_name} not all HMs of group B are available. Group size shrank to "
                           f"{len(group_b_sics)} of initially {old_n_b}!", 'y')

    grp_suffix = f"{name_of_test}_{len(group_a_sics)}-{len(group_b_sics)}"
    grp_fname = rel_filename.replace(".nii.gz", f"_{grp_suffix}.nii.gz")
    p2save = os.path.join(p2intrprt_rslts, model_name, "statistics")
    # p2save = os.path.join(p2intrprt_rslts, model_name, grp_name)

    # Concat Group A (and B)
    done = None
    while True:
        try:
            # Note: this checks automatically whether to take aggregated heatmaps
            # That is the case for using a sub-ensemble with no given basemodel
            group_a_hm_nii = concatenate_hm_nii(model_name=model_name, analyzer_type=analyzer_type,
                                                sic_list=group_a_sics)

            group_b_hm_nii = None if against_zero else concatenate_hm_nii(model_name=model_name,
                                                                          analyzer_type=analyzer_type,
                                                                          sic_list=group_b_sics)
            break
        except FileNotFoundError:
            # TODO this is still for single base-models
            if done:
                break
            if hasattr(load_trained_model(model_name=model_name), "is_multilevel_ensemble"):
                raise ValueError("For ensemble models, aggregated heatmaps (HMs) must be computed before "
                                 "running this function 'create_hm_nii_for_group_stats', and all SICs of "
                                 "each given group must have a corresponding HM. Alternatively, toggle "
                                 "auto_adapt_groups in this function.")
            else:
                heatmaps2mni(model_name=model_name, analyzer_type=analyzer_type,
                             save_original_hm_as_nifti="MNI" in model_name, subset=None,
                             sics_list=group_a_sics + group_b_sics)
            done = True

    group_ab_hm_nii = group_a_hm_nii if against_zero else nb.concat_images([group_a_hm_nii,
                                                                            group_b_hm_nii], axis=-1)

    # Take absolute if requested
    if absolute_hm:
        group_ab_hm_nii = nb.Nifti1Image(dataobj=nb.casting.int_abs(group_ab_hm_nii.get_fdata()),
                                         affine=group_ab_hm_nii.affine, header=group_ab_hm_nii.header)

    # Save concatenated groups as nii
    if not os.path.isdir(p2save):
        os.mkdir(p2save)
    group_ab_hm_nii.to_filename(filename=os.path.join(p2save, grp_fname))

    # Create FSL bash file if needed
    if create_fsl_bash:
        bash_fpath = os.path.join(p2save, f"run_randomise_{grp_suffix}.sh")
        if against_zero:
            # One-sample t-test
            with open(bash_fpath, "w") as file:
                file.write(f"mkdir {grp_suffix}\n"
                           f"randomise -i '{grp_fname}' \\\n"
                           f"\t\t\t\t\t-o {grp_suffix}/{grp_suffix} \\\n"
                           f"\t\t\t\t\t-1 \\\n"
                           f"\t\t\t\t\t-n 5000 \\\n"
                           f"\t\t\t\t\t-T\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                           f"--thresh=0.95 > {grp_suffix}/cluster_tstat1.tsv\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                           f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat1_mm.tsv")

        else:
            # Two-sample t-test: group comparison
            with open(bash_fpath, "w") as file:
                file.write(f"design_ttest2 design_{grp_suffix} {len(group_a_sics)} {len(group_b_sics)}\n"
                           f"mkdir {grp_suffix}\n"
                           f"randomise -i '{grp_fname}' \\\n"
                           f"\t\t\t\t\t-o {grp_suffix}/{grp_suffix} \\\n"
                           f"\t\t\t\t\t-d design_{grp_suffix}.mat \\\n"
                           f"\t\t\t\t\t-t design_{grp_suffix}.con \\\n"
                           f"\t\t\t\t\t-n 5000 \\\n"
                           f"\t\t\t\t\t-T\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                           f"--thresh=0.95 > {grp_suffix}/cluster_tstat1.tsv\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                           f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat1_mm.tsv\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat2.nii.gz "
                           f"--thresh=0.95 > {grp_suffix}/cluster_tstat2.tsv\n"
                           f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat2.nii.gz "
                           f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat2_mm.tsv")

        cprint(f"Execute '{bash_fpath.split('/')[-1]}' in FSL environment within this folder:\n"
               f"'{'/'.join(bash_fpath.split('/')[:-1])}'", 'b')


@only_mpi
def create_hm_nii_for_glm(name_of_test: str, model_name: str, subject_df,
                          analyzer_type: str = 'lrp.sequential_preset_a', absolute_hm: bool = False,
                          create_fsl_bash: bool = True, auto_adapt_groups: bool = False):
    """
    Create 4D Nifti of concatenated heatmaps for statistical test (GLM).
    The model takes the form:
            heatmap ~ regressor [1st col] (+ 'corrected' regressor(s) [further cols])
    :param name_of_test: Name of statistical test
    :param model_name: name of model
    :param subject_df: dataframe [pandas] that contains regressors/independent variables,
                       including those which are to be controlled for
    :param analyzer_type: type of prediction analyzer
    :param absolute_hm: whether to use only absolute heatmaps [bool]
    :param create_fsl_bash: whether to write a corresponding FSL bash file for analysis [bool]
    :param auto_adapt_groups: True: if heatmaps for specific SICs not available the group(s) will adapted
                                    accordingly.
    :return: None
    """

    # Extract regressors
    main_var = subject_df.columns[0]  # main regressors
    corr_vars = subject_df.columns[1:].to_list()  # vars to correct for, if None: empty list
    cprint(f"Create GLM (via FSL):\n"
           f"\theatmap ~ {main_var}"
           f"{(' + ' + ' + '.join(corr_vars)) if len(corr_vars) > 0 else ''}\n", col='b')

    # Load model
    sub_ens = load_trained_model(model_name=model_name)

    assert not sub_ens.is_multilevel_ensemble(), \
        "Given ensemble can't be multi-level ensemble. Specify sub-ensemble!"

    # Prep file-path
    warp = "MNI" not in model_name
    rel_filename = f"{analyzer_type}_relevance_maps{'2mni' if warp else ''}.nii.gz"

    # Adapt groups if no data
    if auto_adapt_groups:
        if not hasattr(load_trained_model(model_name=model_name), "is_multilevel_ensemble"):
            raise NotImplemented("auto_adapt_groups not implemented for basemodels yet!")
        else:
            available_hms = find(fname=rel_filename, folder=os.path.join(p2intrprt_rslts,
                                                                         model_name,
                                                                         "aggregated"),
                                 typ="file", exclusive=False, verbose=False)

            # Filter SICs out of groups with no HMs
            old_n_a = subject_df.shape[0]
            group_sics = list(filter(lambda _x: any([_x in y for y in available_hms]),
                                     subject_df.index.to_list()))
            # == [sic for sic in group_a_sics if any(sic in _sic for _sic in available_hms)]
            if len(group_sics) < old_n_a:
                cprint(f"For {model_name} not all HMs are available. Group size shrank to "
                       f"{len(group_sics)} of initially {old_n_a}!", col='y')

            # Remove SICs from dataframe which have no HM
            subject_df.drop([sic for sic in subject_df.index if sic not in group_sics], inplace=True)

    grp_suffix = f"{name_of_test}_{subject_df.shape[0]}-0"
    grp_fname = rel_filename.replace(".nii.gz", f"_{grp_suffix}.nii.gz")
    p2save = os.path.join(p2intrprt_rslts, model_name, "statistics")

    # Concatenate HMs
    group_hm_nii = concatenate_hm_nii(model_name=model_name, analyzer_type=analyzer_type,
                                      sic_list=subject_df.index.to_list())

    # Take absolute if requested
    if absolute_hm:
        group_hm_nii = nb.Nifti1Image(dataobj=nb.casting.int_abs(group_hm_nii.get_fdata()),
                                      affine=group_hm_nii.affine, header=group_hm_nii.header)

    # Save concatenated groups as nii
    if not os.path.isdir(p2save):
        os.mkdir(p2save)
    group_hm_nii.to_filename(filename=os.path.join(p2save, grp_fname))

    # Create FSL bash file if needed
    if create_fsl_bash:
        bash_fpath = os.path.join(p2save, f"run_randomise_{grp_suffix}.sh")
        design_fpath = os.path.join(p2save, f"design_{grp_suffix}")

        # GLM
        #  1) Create design-matrix [design.mat], incl. variables to be corrected for
        desig_mat_str = f"/NumWaves {1 + subject_df.shape[1]}\n" \
                        f"/NumPoints {subject_df.shape[0]}\n" \
                        f"/PPheights 1 {' '.join([str(subject_df[col].max()-subject_df[col].min()) for col in subject_df.columns])}\n" \
                        f"/Matrix\n"
        for sic in subject_df.index:
            desig_mat_str += "1\t" + "\t".join(
                [str(subject_df.loc[sic][col]) for col in subject_df.columns]) + "\n"
        with open(design_fpath + ".mat", "w") as file:
            file.write(desig_mat_str)

        #  2) Create contrast matrix
        desig_con_str = f"/NumWaves {1 + subject_df.shape[1]}\n" \
                        f"/NumContrasts 1\n" \
                        f"/PPheights 0 1" + " 0"*(subject_df.shape[1] - 1) + "\n" \
                        f"/Matrix\n" \
                        f"0 1" + " 0"*(subject_df.shape[1] - 1)
        with open(design_fpath + ".con", "w") as file:
            file.write(desig_con_str)

        # Write bash
        with open(bash_fpath, "w") as file:
            file.write(f"mkdir {grp_suffix}\n"
                       f"randomise -i '{grp_fname}' \\\n"
                       f"\t\t\t\t\t-o {grp_suffix}/{grp_suffix} \\\n"
                       f"\t\t\t\t\t-d design_{grp_suffix}.mat \\\n"
                       f"\t\t\t\t\t-t design_{grp_suffix}.con \\\n"
                       f"\t\t\t\t\t-n 5000 \\\n"
                       f"\t\t\t\t\t-T\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                       f"--thresh=0.95 > {grp_suffix}/cluster_tstat1.tsv\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat1.nii.gz "
                       f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat1_mm.tsv\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat2.nii.gz "
                       f"--thresh=0.95 > {grp_suffix}/cluster_tstat2.tsv\n"
                       f"cluster --in={grp_suffix}/{grp_suffix}_tfce_corrp_tstat2.nii.gz "
                       f"--thresh=0.95 --mm > {grp_suffix}/cluster_tstat2_mm.tsv")

        cprint(f"Execute '{bash_fpath.split('/')[-1]}' in FSL environment within this folder:\n"
               f"'{'/'.join(bash_fpath.split('/')[:-1])}'", 'b')


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

if __name__ == "__main__":

    # Compute stats heatmaps for
    #  1) for some individuals
    #  2) whole age-range per sequence (sub-ensemble)
    #  3a) specific group comparisons young (<40 years) vs old (60+ years)
    #  3b) specific group comparisons e.g. normal vs. high brain-age in 60-65 year olds

    # Define model
    grand_ens = load_trained_model("2020-10-07_15-26_AGE_Grand_ens10")
    # Best basemodels: T1: bm X | FLAIR: bm X | SWI: bm X
    # grand_ens = load_trained_model("2020-08-23_17-11_AGE_Grand_ens10")
    # Best basemodels: T1: bm 8 | FLAIR: bm 7 | SWI: bm 9

    # # Extract SICs for group comparison: due to few samples, use val + test set
    # grand_preds = grand_ens.get_headmodel_predictions()
    # grand_preds["error"] = grand_preds.predictions - grand_preds.target
    # elderly_grand_preds = grand_preds.loc[grand_preds.target.between(60, 65, inclusive=True)]
    # # high brain-age: N=4
    # high_ba_grp = elderly_grand_preds.loc[elderly_grand_preds.error >= 5].sics_sorted.to_list()
    # # normal BA: N=17
    # norm_grp = elderly_grand_preds.loc[elderly_grand_preds.error.between(
    #     -5, 5, inclusive=False)].sics_sorted.to_list()

    # Compute stats
    for sens in grand_ens.list_of_submodels:
        subens_name = os.path.join(grand_ens.name, sens)
        sics_dict = grand_ens.get_sics(subset=None, submodel_name=sens)
        sics4stats = sics_dict["validation"] + sics_dict["test"]

        # 1) chose some individuals and compute between basemodel stats
        if grand_ens.name not in ["2020-10-07_15-26_AGE_Grand_ens10"]:  # DONE-list
            for sic in ["LI02820234", "LI00033113", "LI02818617"]:  # SICs for publication
                create_hm_nii_for_sic_stats(model_name=subens_name, sic=sic, mni=False, absolute_hm=True,
                                            create_fsl_bash=True)

        # 2) For whole age range: one-sample t-test [DONE]
        if grand_ens.name not in ["2020-10-07_15-26_AGE_Grand_ens10"]:  # DONE-list
            create_hm_nii_for_group_stats(
                name_of_test="whole_age_range_one_sample_ttest",
                model_name=subens_name,
                group_a_sics=sics4stats, group_b_sics=None, absolute_hm=True,
                create_fsl_bash=True, auto_adapt_groups=True)

        # 3a) two-sample t-test: young vs old (<=40; 60+)
        if grand_ens.name not in ["2020-10-07_15-26_AGE_Grand_ens10"]:  # DONE-list
            young_grp = [sic for sic in sics4stats if age_of_sic(sic) <= 40]
            elderly_grp = [sic for sic in sics4stats if age_of_sic(sic) >= 60]
            # print("young group, N =", len(young_grp))  # N=61
            # print("elderly group, N =", len(elderly_grp))  # N=243

            create_hm_nii_for_group_stats(
                name_of_test="young40-_vs_elderly60+_test_two_sample",
                model_name=subens_name,
                group_a_sics=young_grp, group_b_sics=elderly_grp, absolute_hm=False,
                create_fsl_bash=True, auto_adapt_groups=True)

        # 3b) two-sample t-test: define groups
        if grand_ens.name not in ["2020-10-07_15-26_AGE_Grand_ens10"]:  # DONE-list

            # 3b.1 high DBA vs. normal:
            # Since multi-level is trained on test set (CV). we take average prediction across basemodels
            # to receive the DBA of subjects in the validation set and test set

            # elderly_grp = [sic for sic in sics4stats if (70 >= age_of_sic(sic) >= 60)]
            mean_valtest_preds = np.concatenate([grand_ens.get_headmodel_data(
                multi_level=False, subset="validation")[0].mean(axis=1),
                                                 grand_ens.get_headmodel_data(
                                                     multi_level=False, subset="test")[0].mean(axis=1)])

            _sics = np.concatenate([grand_ens.get_headmodel_data(multi_level=False, subset="validation",
                                                                 return_sics=True)[-1],
                                    grand_ens.get_headmodel_data(multi_level=False, subset="test",
                                                                 return_sics=True)[-1]])

            preddf = pd.DataFrame(data=mean_valtest_preds,
                                  # (156, 30).mean(axis=1) => (156,)
                                  columns=["mean_BM_preds"],
                                  index=_sics)

            preddf["age"] = [age_of_sic(sic) for sic in preddf.index]  # retrieve age info
            preddf["DBA"] = preddf.mean_BM_preds - preddf.age  # calc DBA
            preddf = preddf[preddf["age"].between(60, 65)]  # set age range

            high_DBA = preddf[preddf["DBA"].between(4, np.inf, inclusive=False)]  # DBA > 4, len=17
            norm_DBA = preddf[preddf["DBA"].between(-1, 1)]  # |DBA| <= 1, len=32

            print("N high DBA = ", len(high_DBA))
            print("N norm DBA = ", len(norm_DBA))

            # high_DBA.age.hist(bins=6, label="high DBA")
            # norm_DBA.age.hist(bins=6, alpha=.5, label="norm")
            # plt.legend()
            # plt.show()

            create_hm_nii_for_group_stats(
                name_of_test=f"high_vs_normal_BA_in_60-65_test_two_sample",
                model_name=subens_name,
                group_a_sics=list(norm_DBA.index), group_b_sics=list(high_DBA.index), absolute_hm=False,
                create_fsl_bash=True, auto_adapt_groups=True)

        if grand_ens.name not in ["2020-10-07_15-26_AGE_Grand_ens10"]:  # DONE-list
            # 3b.2 healthy vs. diabetes:
            raw_tab = load_raw_study_table(exclusion=True, full_table=True).set_index("SIC")
            # cprint(f"Diabetes columns: {[c for c in raw_tab.columns if 'diab' in c.lower()]}", fm="bo")
            # print(raw_tab['diabetes_y_n'].unique())

            elderly_grp = [sic for sic in sics4stats if (75 >= age_of_sic(sic) >= 50)]
            # elderly_grp = [sic for sic in sics4stats if age_of_sic(sic) >= 60]
            healthy_grp = [sic for sic in elderly_grp if raw_tab.loc[sic]['diabetes_y_n'] == "no"]
            diab_grp = [sic for sic in elderly_grp if raw_tab.loc[sic]['diabetes_y_n'] == "yes"]
            print(f"N-healthy: {len(healthy_grp)} | N-diabetes: {len(diab_grp)}")

            create_hm_nii_for_group_stats(
                name_of_test="healthy_vs_diabetes_BA_in_75-50_test_two_sample",
                model_name=subens_name,
                group_a_sics=healthy_grp, group_b_sics=diab_grp, absolute_hm=False,
                create_fsl_bash=True, auto_adapt_groups=True)

        # 4) TODO GLM on DBA (correct for age) in cohort 50+
        if grand_ens.name not in []:  # DONE-list: "2020-10-07_15-26_AGE_Grand_ens10"
            extended_set = False
            min_age = 50  # elderly_grp
            poly_degree = 1  # for polynomial fit on age

            if extended_set:  # for age >= 50: N = 224
                # Since multi-level is trained on test set (CV). we take average prediction across
                # basemodels to receive the DBA of subjects in the validation set and test set
                _preds = mean_valtest_preds = np.concatenate([grand_ens.get_headmodel_data(
                    multi_level=False, subset="validation")[0].mean(axis=1),
                                                              grand_ens.get_headmodel_data(
                                                                  multi_level=False,
                                                                  subset="test")[0].mean(1)])

                _sics = np.concatenate([grand_ens.get_headmodel_data(multi_level=False,
                                                                     subset="validation",
                                                                     return_sics=True)[-1],
                                        grand_ens.get_headmodel_data(multi_level=False,
                                                                     subset="test",
                                                                     return_sics=True)[-1]])

                preddf = pd.DataFrame(data=_preds,
                                      # (156, 30).mean(axis=1) => (156,)
                                      columns=["predictions"],
                                      index=_sics)

                preddf["age"] = [int(age_of_sic(sic)) for sic in preddf.index]  # retrieve age info
                preddf["DBA"] = (preddf["predictions"] - preddf.age).round(3)  # calc DBA

            else:  # for age >= 50: N = 134
                # Here, we take only the predictions of sub-ensembles on the test set. This is closer to
                # the predictions of the multi-level ensemble than the 'extended_set' approach above,
                # since MLens are trained and evaluated on the sub-ens-predictions on the test set via CV.

                preddf = grand_ens.get_predictions(subset="test", submodel_name=sens)
                preddf.set_index("sic", inplace=True)
                preddf.rename(columns={"test_y": "age"}, inplace=True)
                preddf.age = preddf.age.astype(int)
                pred_col = [p for p in preddf.columns if "pred" in p][0]
                preddf["DBA"] = (preddf[pred_col] - preddf.age).round(3)

            preddf = preddf[["DBA", "age"]]  # change order since "age" is secondary regressor

            # Add polynomial if requested (poly_degree > 1), as in: heatmap ~ DBA + age + age**2
            for pol in range(poly_degree-1):
                preddf[f"age**{pol+2}"] = preddf.age**(pol+2)

            create_hm_nii_for_glm(name_of_test=f"GLM_DBA_corrAge_{min_age}+", model_name=subens_name,
                                  subject_df=preddf[preddf.age >= min_age],
                                  absolute_hm=False, auto_adapt_groups=True)

        # Final) Merge scripts and run in FSL
        p2stats = os.path.join(p2intrprt_rslts, subens_name, "statistics")
        with open(os.path.join(p2stats, "run_merged.sh"), "w") as fmerg:
            fmerg.write("# Run merged bash script:\n\n")
        for f in find(fname=".sh", folder=p2stats,
                      typ="file", exclusive=False, fullname=False, abs_path=True, verbose=True):
            if "run_merged" not in f:
                with open(os.path.join(p2stats, "run_merged.sh"), "a") as fmerg:
                    fmerg.write(f"bash {f.split('/')[-1]}\n")
                    # # Alternatively copy all lines:
                    # with open(f, "r") as frun:
                    #     for line in frun.readlines():
                    #         fmerg.write(line)
                    # fmerg.write("\n\n")

    # %% Compute aggregated heatmaps across sub-ensembles >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # create_aggregate_heatmap_per_subject(model_name=...)
    # DONE: ["2020-08-23_17-11_AGE_Grand_ens10", "2020-10-07_15-26_AGE_Grand_ens10",
    #        "2020-08-26_10-38_AGE_GrandREG_ens5", "2020-10-13_17-22_AGE_GrandREG_ens5"]

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
