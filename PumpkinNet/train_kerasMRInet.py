# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Keras model equivalent to predLIFE.py

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2019-2020
"""

# %% Import
import copy

import joblib
from sklearn.linear_model import Ridge

from MRInet import create_keras_model, cubify, keras
from load_mri_data import *

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Global vars
p2resulttab = "./processed/Keras/Predictions/overview_model_performances.csv"
p2datasplit = "./processed/Keras/Datasplit/"
p2models = "./processed/Keras/Models/"
p2logs = "./processed/Keras/logs/"


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Prepare data & model training
def save_datasplit(model_name, split_dict):
    # Define data-split file as save it
    if not os.path.exists(p2datasplit):
        os.mkdir(p2datasplit)
    p2datasplit_model = p2datasplit + f"{model_name}_data_split.npy"
    cprint(f"\nSave datasplit to '{p2datasplit_model}'.", 'b')
    np.save(p2datasplit_model, split_dict)


def load_datasplit(model_name, verbose=True):
    # Load data-split dict for given model

    # Check whether basemodel of an ensemble is given
    if "Grand" in model_name:
        # Remove sub-ensemble & basemodel-part from name
        model_name = model_name.split("/")[0]
    elif "_model" in model_name:
        # Remove basemodel-part from name
        model_name = "/".join(model_name.split("/")[:-1])

    p2datasplit_model = f"{p2datasplit}{model_name}_data_split.npy"
    if verbose:
        cprint(f"\nLoading data split from '{p2datasplit_model.split('/')[-1]}'.", 'b')
    split_dict = np.load(p2datasplit_model, allow_pickle=True).item()
    return split_dict


def cubify_whole_set(mriset, newshape):
    start = datetime.now()
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() * 4) as executor:
        cub_mriset = executor.map(cubify, mriset, [newshape] * len(mriset))

    cub_mriset = np.array(list(cub_mriset))
    duration = chop_microseconds(datetime.now() - start)

    print("New shape of MRIset:", cub_mriset.shape)

    print(f"Duration of cubifying mri set {duration} [h:m:s]")

    return cub_mriset


def train_and_test_keras_model(target, mri_space="fs", mri_type="T1", region=None,
                               augment_data=False, transform_types: (list, str) = None, n_augment=None,
                               bg_noise: (bool, float) = False,
                               norm=(0, 1), class_task=False,
                               ensemble=None, ensemble_head="both",
                               split_dict=None, save_split=True, parent_folder=None,
                               epochs=40, batch_size=2,
                               seed=False, **kwargs):
    # Input argument test(s)
    assert isinstance(norm, tuple) or norm is None, "'norm' must be tuple OR None!"
    assert isinstance(ensemble, int) or ensemble is None, "'ensemble' must be int [2-10] OR None!"

    if ensemble and ensemble > 1:
        # Max 10 models for ensemble
        ensemble = np.clip(a=ensemble, a_min=2, a_max=10)
        es = True
    else:
        ensemble = 1
        es = False

    if mri_type.lower() == "all":
        ensemble = len(mri_sequences)  # 3
        mri_type = mri_sequences[0]  # starting with the first sequence ('t1')
        sequence_es = True
    else:
        sequence_es = False

    if region is not None and region.lower() == "all":
        ensemble = len(brain_regions)  # 3
        region = None  # switch off for region ensemble
        region_es = True
    else:
        region_es = False

    assert np.sum([region_es, sequence_es, es]) <= 1, "Train only one ensemble type!"
    assert not (augment_data * (ensemble > 1)), "Ensemlbe data can't be augmented yet [NOT IMPLEMENTED]."
    assert isinstance(transform_types, (list, str)) or transform_types is None, \
        "'transform_types' must be list, str OR None!"

    target = target.lower()

    if seed:
        np.random.seed(42)

    # Compile name of neural network
    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M_')
    netname = tstamp + "MRIkerasNet" + target.upper() + (
        "" if sequence_es else f"_{mri_type.upper()}")
    if isinstance(region, str):
        netname += f"_{region.upper()[0:3]}"
    if augment_data:
        netname += "_augm" + (str(n_augment) if n_augment is not None else "")
    if transform_types:
        netname += f"-{transform_types[0:3]}" if isinstance(transform_types, str) else \
            f"-{''.join([tt[0:3] for tt in transform_types])}"
    if bg_noise:
        netname += "_BGn" + (f"{bg_noise:.3f}".lstrip("0") if isinstance(bg_noise, float) else "")
    if mri_space.lower() != "fs":
        netname += f"_{mri_space.upper()}"  # eg. _MNI
    if norm == (-1, 1):
        netname += "_N-11"
    if class_task:
        netname += "_BinClassi"
    cf = None  # init
    if "correct_for" in kwargs.keys():
        cf = kwargs["correct_for"].lower()
        netname += "CF" + cf
    lrelu = kwargs.pop("leaky_relu", True)
    if not lrelu:
        netname += "_relu"
    bnorm = kwargs.pop("bnorm", False)
    if bnorm:
        netname += "_bnorm"
    if es:
        netname += f"_ens{ensemble}"
    if region_es:
        netname += f"_region-ens"
    if sequence_es:
        netname += f"_sequence-ens"

    if parent_folder is not None:  # mainly for grand ensemble
        netname = os.path.join(parent_folder, netname)

    # # Get Data
    print(f"Load data for model: {netname} ...")

    # For model-training, load data with random split and save the information:
    life_data = get_life_data(seed=seed, mri_sequence=mri_type, region=region, target=target,
                              target_scale="softmax" if class_task else "linear",
                              augment="minority" if augment_data else None,
                              transform_types=transform_types, n_augment=n_augment,
                              bg_noise=bg_noise,
                              mri_space=mri_space, mri_scale=norm,
                              split_dict=split_dict, **kwargs)

    # # Prepare data split for train, val, & test-set:
    # Save split dictionary for reproducibility
    split_dict = create_split_dict(life_data) if split_dict is None else split_dict
    if save_split:
        save_datasplit(model_name=netname.split("/")[-1],  # split for potential parent_folder
                       split_dict=split_dict)

    if augment_data:
        plot_set_distributions(life_data=life_data,
                               save_path=f"{p2datasplit}{netname}_set_distribution.png")

    x_train, y_train = life_data["train"].to_keras()  # x.shape(1614, 196, 196, 196, 1)
    free_memory(life_data["train"].mri)
    x_val, y_val = life_data["validation"].to_keras()  # (201, 196, 196, 196, 1)
    free_memory(life_data["validation"].mri)
    x_test, y_test = life_data["test"].to_keras()  # x.shape (201, ...); y:'one-hot' if classific.)
    free_memory(life_data["train"].mri)
    tbias = life_data["train"].target_bias  # same for all subsets

    # Shuffle order of training data
    if augment_data:
        # Since augmented samples are just attached at the end of dataset, we want to mingle them
        print("Shuffle order of training data ...")
        shufidx = np.arange(len(y_train))
        np.random.shuffle(shufidx)
        x_train = x_train[shufidx]
        y_train = y_train[shufidx]

    # # Build model(s)
    def dt(_x):
        """Placeholder function for region-ensemble (dt = lambda _x: _x  # identity mapping)"""
        return _x

    for mi in range(ensemble):

        # Prepare ensemble (if required)
        if ensemble > 1:  # == es or region_es or sequence_es:

            # Make dirs
            if mi == 0:
                os.makedirs(name=p2models + netname)
                os.makedirs(name=p2logs + netname)

            if region_es:
                # Adapt data transformer
                region = list(brain_regions.keys())[mi]  # region i (e.g., "cerebellum")
                region_mask = create_region_mask(region=region,
                                                 # Remove redundant cortex in harvard-oxford-subcortical:
                                                 reduce_overlap=True)

                def dt(_x):
                    """Overwrite func dt = lambda _x: stamp_region(dataset=_x, region_mask=region_mask)"""
                    _xc = _x.copy()
                    return stamp_region(dataset=_xc, region_mask=region_mask)

                nname = f"{netname}/{mi}_{region}_model"

            elif sequence_es:
                mri_type = mri_sequences[mi]  # ["t1", "flair", "swi"][mi]
                nname = f"{netname}/{mi}_{mri_type}_model"

                # For Sequences-Ensembles we need to reload the data for each further sequence
                if mi > 0:
                    life_data = get_life_data(seed=seed, mri_sequence=mri_type, region=region,
                                              target=target,
                                              target_scale="softmax" if class_task else "linear",
                                              augment="minority" if augment_data else None,
                                              transform_types=transform_types, n_augment=n_augment,
                                              bg_noise=bg_noise,
                                              mri_space=mri_space, mri_scale=norm,
                                              split_dict=split_dict, **kwargs)

                    x_train, y_train = life_data["train"].to_keras()
                    free_memory(life_data["train"].mri)
                    x_val, y_val = life_data["validation"].to_keras()
                    free_memory(life_data["validation"].mri)
                    x_test, y_test = life_data["test"].to_keras()
                    free_memory(life_data["test"].mri)
                    # tbias = life_data["train"].target_bias  # Same for all MRI sequences

            else:
                nname = f"{netname}/{mi}_model"

        else:
            nname = netname

        input_shape = dt(x_val)[0, ..., 0].shape
        model = create_keras_model(name=nname, target_bias=tbias, input_shape=input_shape,
                                   class_task=class_task, leaky_relu=lrelu, batch_norm=bnorm)

        # # Save model progress (callbacks)
        # See also: https://www.tensorflow.org/tutorials/keras/save_and_load
        callbacks = [keras.callbacks.ModelCheckpoint(
            filepath=f"{p2models}{nname}" + "_{epoch}.h5",
            save_best_only=True,
            save_weights_only=False,
            period=10,
            monitor="val_loss",
            verbose=1),
            keras.callbacks.TensorBoard(log_dir=f"{p2logs}{nname}/")]
        # , keras.callbacks.EarlyStopping()]

        # # Train the model
        cprint('Fit model on training data ...', 'b')
        history = model.fit(dt(x_train), y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(dt(x_val), y_val))

        # Save final model (weights+architecture)
        model.save(filepath=f"{p2models}{nname}_final.h5")  # HDF5 file

        # Report training metrics
        # print('\nhistory dict:', history.history)
        np.save(file=f"{p2logs}{nname}_history", arr=history.history)

        # # Evaluate the model on the test data
        cprint(f'\nEvaluate {nname} on test data ...', 'b')
        performs = model.evaluate(dt(x_test), y_test, batch_size=1)  # , verbose=2)
        cprint(f'test loss, test performance: {performs}', 'y')

        # Save results
        with open(f"{p2logs}{nname}_results.txt", "w") as result_file:
            result_file.write(f"test loss, test performance: {performs}\n"
                              f"target={target}\n"
                              f"mri_sequence={mri_type}\n"
                              f"mri_space={mri_space.lower()}\n"
                              f"augment_data={augment_data}"
                              f"{'' if n_augment is None else ' (n=' + str(n_augment) + ')'}\n"
                              f"transform_types={transform_types}\n"
                              f"bg_noise={bg_noise}\n"
                              f"norm={norm}\n"
                              f"class_task={class_task}\n"
                              f"seed={seed}\n"
                              f"netname={netname}\n"
                              f"ensemble={es}"
                              f"{(' (' + str(mi + 1) + '/' + str(ensemble) + ')') if es else ''}\n"
                              f"sequence-ensemble={sequence_es}"
                              f"{(' (' + str(mi + 1) + '/' + str(ensemble) + ')') if sequence_es else ''}"
                              f"\nregion-ensemble={region_es}"
                              f"{(' (' + str(mi + 1) + '/' + str(ensemble) + ')') if region_es else ''}\n"
                              f"region={region}\n"
                              f"kwargs={kwargs}\n")

        # Save model predictions on validation and test set per SIC (useful for later ensemble training)
        if ensemble > 1:  # == if es or region_es or sequence_es
            # TODO adapt for classification task (since pred have shape (N, n_classes=2)
            # Collect predictions in tables
            sub_m_pred_val = pd.DataFrame(columns=["sic", "val_pred", "val_y"])  # init
            sub_m_pred_test = pd.DataFrame(columns=["sic", "test_pred", "test_y"])  # init

            sub_m_pred_val["sic"] = life_data["validation"].sics
            sub_m_pred_val["val_pred"] = model.predict(x=dt(x_val), batch_size=1).squeeze()
            sub_m_pred_val["val_y"] = y_val

            sub_m_pred_test["sic"] = life_data["test"].sics
            sub_m_pred_test["test_pred"] = model.predict(x=dt(x_test), batch_size=1).squeeze()
            sub_m_pred_test["test_y"] = y_test

            # Save the predictions (for later training of ensemble head-model)
            sub_m_pred_val.to_csv(path_or_buf=f"{p2models}{nname}_val_predictions.csv", index=False)
            sub_m_pred_test.to_csv(path_or_buf=f"{p2models}{nname}_test_predictions.csv", index=False)

    # Do ensemble predictions
    if ensemble > 1:  # == es or region_es or sequence_es:
        cprint(f'\nEvaluate {netname} ensemble on test data ...', 'b')

        # Fit ensemble (head) model
        ensemble_model = fit_ensemble(model_name=netname,
                                      # Use val-data for training since 'unseen' by base-models:
                                      xdata=None if sequence_es else x_val,
                                      ydata=None if sequence_es else y_val,
                                      valdata=None if sequence_es else (x_test, y_test),
                                      head_model_type=ensemble_head.lower(), save=True)

        # Evaluate on test set
        htypes = ["linear", "nonlinear"] if ensemble_head.lower() == "both" else [ensemble_head.lower()]

        performs = eval_ensemble(model_name=netname, head_model_list=ensemble_model,
                                 head_type=ensemble_head.lower())

        p_str = ", ".join("'{}': {:.5f}".format(*p) for p in zip(htypes, performs))  # convert to string
        # e.g., "'linear': 4.112, 'nonlinear': 16.472"

        # Save results
        with open(f"{p2logs}{netname}_results.txt", "w") as result_file:
            # result_file.write(f"test performance: {list(zip(htypes, performs))}\n"
            result_file.write(f"test performance: {p_str}\n"
                              f"target={target}\n"
                              f"mri_sequence={mri_type}\n"
                              f"mri_space={mri_space}\n"
                              f"augment_data={augment_data}"
                              f"{'' if n_augment is None else ' (n=' + str(n_augment) + ')'}\n"
                              f"transform_types={transform_types}\n"
                              f"bg_noise={bg_noise}\n"
                              f"norm={norm}\n"
                              f"class_task={class_task}\n"
                              f"seed={seed}\n"
                              f"netname={netname}\n"
                              f"ensemble={es}\n"
                              f"region-ensemble={region_es}\n"
                              f"sequence-ensemble={sequence_es}\n"
                              f"region={region}\n"
                              f"kwargs={kwargs}\n")

        write_basemodel_performance_table(model_name=netname, verbose=True)

    # Write results in model overview table
    cols = ["date", "model_name", "target", "mri_type", "mri_space", "intensity_scale",
            "background_noise", "augmentation", "transformation_types", "classification_task",
            "corrected_for", "ensemble",
            "performance"]

    varis = [tstamp[:-1], netname, target, "all" if sequence_es else mri_type, mri_space, norm,
             bg_noise, augment_data, transform_types, class_task,
             cf, False if ensemble == 1 else ensemble,
             performs if es else performs[-1]]

    # Create table if necessary
    perform_tab = pd.read_csv(p2resulttab) if os.path.isfile(p2resulttab) else pd.DataFrame(columns=cols)
    # perform_tab.set_index("model-name", inplace=True)
    perform_tab = perform_tab.append(dict(zip(cols, varis)), ignore_index=True)
    perform_tab.to_csv(p2resulttab, index=False)


def train_grand_ensemble(target, regional=False, data_split=None, n_basemodels=10, **kwargs):
    class_task = False  # Could be implemented

    # # Check vars
    if regional and n_basemodels > 5:
        raise ValueError("For regional grand ensemble, keep the number of base models <= 5.")
        # For regional there are 9 combinations (N_sequences x N_regions). This multiplied with
        # N base models can become too long to compute

    # # Compile name for the grand ensemble
    start_time = datetime.now()
    tstamp = start_time.strftime('%Y-%m-%d_%H-%M_')
    grandname = f"{tstamp}{target.upper()}_Grand{'REG' if regional else ''}_ens{n_basemodels}"

    # # Set params
    mri_space = "mni" if regional else "fs"

    # Check-in data split
    if data_split is None:
        # 1) Create train-val-test data split
        data_split = create_split_dict(get_life_data(mri_sequence="t1",  # T1 has most data
                                                     mri_space=mri_space))

    # 2) Save data split
    save_datasplit(model_name=grandname, split_dict=data_split)

    # 3) Plot data split
    plot_set_distributions(data_split=data_split, target=target,
                           save_path=f"{p2datasplit}{grandname}.png")

    # 4) Start training
    cprint(f"\nStart training grand ensemble '{grandname}' ...\n", col='b', fm='bo')

    if not regional:
        for seq in mri_sequences:
            train_and_test_keras_model(target=target, mri_space=mri_space, mri_type=seq,
                                       ensemble=n_basemodels, split_dict=data_split,
                                       parent_folder=grandname,
                                       save_split=False, **kwargs)

    else:
        for reg in brain_regions:
            for seq in mri_sequences:
                train_and_test_keras_model(target=target, mri_space=mri_space, region=reg, mri_type=seq,
                                           ensemble=n_basemodels, split_dict=data_split,
                                           parent_folder=grandname,
                                           save_split=False, **kwargs)

    # 5) Evaluate grand ensamble
    ensemble_model = fit_ensemble(model_name=grandname, head_model_type="both", multi_level=False,
                                  save=True)  # headmodel trained an all basemodel predictions
    ensemble_model_mh = fit_ensemble(model_name=grandname, head_model_type="linear", multi_level=True,
                                     save=True)  # headmodel trained an all subensemble predictions via CV
    # TODO Top-model training could be done via cross-validation, too (see z_playground.py)

    # 6) Evaluate on test set
    htypes = ["linear", "nonlinear"]
    performs = eval_ensemble(model_name=grandname, head_model_list=ensemble_model,
                             head_type="both", multi_level=False)
    performs_mh = eval_ensemble(model_name=grandname, head_model_list=ensemble_model_mh,
                                head_type="linear", multi_level=True)

    p_str = ", ".join("'{}': {:.5f}".format(*p) for p in zip(htypes, performs))  # convert to string
    p_str_mh = ", ".join("'{}': {:.5f}".format(*p) for p in zip(["linear"], performs_mh))

    # 7a) Save results of grand ensemble (adapt below)
    with open(f"{p2logs}{grandname}_results.txt", "w") as result_file:
        # result_file.write(f"test performance: {list(zip(htypes, performs))}\n"
        result_file.write(f"test performance (flat): {p_str}\n"
                          f"test performance (multi-level): {p_str_mh}\n"
                          f"target={target}\n"
                          f"mri_space={mri_space}\n"
                          f"class_task={class_task}\n"
                          f"netname={grandname}\n"
                          f"grand_ensemble=True\n")

    # 7b) Write results in model overview table
    cols = ["date", "model_name", "target", "mri_type", "mri_space", "intensity_scale",
            "background_noise", "augmentation", "transformation_types", "classification_task",
            "corrected_for", "ensemble",
            "performance"]

    varis = [tstamp[:-1], grandname, target, "all", mri_space, (0, 1), False, False, None, class_task,
             None, n_basemodels,
             performs + performs_mh]  # ['linear', 'nonlinear', 'linear multi-head']

    # Create table if necessary
    perform_tab = pd.read_csv(p2resulttab) if os.path.isfile(p2resulttab) else pd.DataFrame(
        columns=cols)
    # perform_tab.set_index("model-name", inplace=True)
    perform_tab = perform_tab.append(dict(zip(cols, varis)), ignore_index=True)
    perform_tab.to_csv(p2resulttab, index=False)

    # Some print-outs
    cprint(f"Grand ensemble '{grandname}' trained and tested!", col='b', fm='bo')
    n_models = len(mri_sequences) * (len(brain_regions) if regional else 1)
    print(datetime.now().strftime('%Y-%m-%d, %H:%M'))
    cprint(f"Training {n_models} sub-ensembles with {n_basemodels} base models each ("
           f"{n_models * n_basemodels} base models in total) took: "
           f"{chop_microseconds(datetime.now() - start_time)} [hh:mm:ss]", 'b')


def crop_model_name(model_name):
    if model_name.endswith(".h5"):
        model_name = model_name[0:-len(".h5")]

    if model_name.endswith("_final"):
        model_name = model_name[0:-len("_final")]

    return model_name


class TrainedEnsembleModel:
    def __init__(self, model_name):
        self.path2ensemble = os.path.abspath(p2models + model_name)
        self.name = model_name
        self.list_of_headmodels = [hm for hm in os.listdir(self.path2ensemble) if "headmodel" in hm]
        self.active_model = None
        if self.is_multilevel_ensemble():
            self.list_of_submodels = [sens for sens in os.listdir(
                self.path2ensemble) if os.path.isdir(os.path.join(self.path2ensemble, sens))]
        else:
            self.list_of_submodels = sorted([sm for sm in os.listdir(
                self.path2ensemble) if "_final.h5" in sm])

    def summary(self):
        # Ensemble name
        cprint(f"\n'{self.name}' has:", 'b')
        # Basemodels
        cprint(f"\n{len(self.list_of_submodels)} submodels:", fm="ul")
        print("", *self.list_of_submodels, "", sep="\n\t")
        # Headmodel(s)
        cprint(f"{len(self.list_of_headmodels)} headmodels:", fm="ul")
        print("", *self.list_of_headmodels, "", sep="\n\t")

    def is_region_ensemble(self):
        return "region-ens" in self.name or "GrandREG" in self.name

    def is_multilevel_ensemble(self):
        return "grand" in self.name.split("/")[-1].lower()

    def get_submodel(self, submodel_name):
        return load_trained_model(model_name=os.path.join(self.name, submodel_name))

    def get_sics(self, subset=None, submodel_name=None, dropnan=True):

        # TODO for train we get all sics: implemet dropnan also for train sics
        if subset is not None and "train" in subset.lower() and dropnan:
            raise NotImplementedError("dropnan for training SICs must be implemented still!")

        if submodel_name is None or not dropnan:
            sics_dict = load_datasplit(model_name=self.name)

        else:
            if self.is_multilevel_ensemble():
                submodel_name = submodel_name.split("/")[-1] if "/" in submodel_name else submodel_name
                smidx = np.where(np.array(self.list_of_submodels) == submodel_name)[0][0]
                val_sics = get_sub_ensemble_predictions(model_name=self.name, subset="val",
                                                        as_numpy=False, verbose=False)
                val_sics = val_sics[val_sics.columns[smidx + 1]]
                val_sics.dropna(inplace=True)
                val_sics = val_sics.index.to_list()
                test_sics = get_sub_ensemble_predictions(model_name=self.name, subset="test",
                                                         as_numpy=False, verbose=False)
                test_sics = test_sics[test_sics.columns[smidx + 1]]
                test_sics.dropna(inplace=True)
                test_sics = test_sics.index.to_list()

            else:

                val_sics = self.get_predictions(subset="val", submodel_name=submodel_name).sic.to_list()
                test_sics = self.get_predictions(subset="test", submodel_name=submodel_name).sic.to_list()

            sics_dict = {"validation": val_sics,
                         "test": test_sics}

        # Return
        if subset is None:
            return sics_dict
        else:
            subset = "validation" if "val" in subset.lower() else \
                "test" if "test" in subset.lower() else "train"
            return sics_dict[subset]

    def get_headmodel(self, head_model_type, multi_level: bool, cv=None):
        if not self.is_multilevel_ensemble() or multi_level is False:
            cv = False
        else:
            assert cv is not None, "cv must be True OR False!"

        return load_ensemble(model_name=self.name, head_model_type=head_model_type,
                             multi_level=multi_level, cv=cv)

    def set_active_model(self, model_name=None):
        if np.any([model_name in subm for subm in self.list_of_submodels]):
            # For submodel
            self.active_model = self.get_submodel(model_name)
            cprint(f"Current active model is {self.active_model.name}", 'y')

        elif np.any([model_name in hm for hm in self.list_of_headmodels]):
            # For headmodel
            self.active_model = self.get_headmodel(
                head_model_type="nonlinear" if "_nonlin" in model_name else "linear",
                multi_level="multilevel" in model_name,
                cv="_cv_" in model_name)
            cprint(f"Current active model is {self.active_model.name}", 'y')

        else:
            cprint(f"No sub-/head-model with the name {model_name} found!\n > active_model = None", 'y')
            self.active_model = None

    def dt(self, _x):
        """
        Data transformer (dt) for (specifically region) ensemble. dt adapts MRI data (_x) according to
        active model in self.
        """
        if self.is_region_ensemble() and self.active_model is not None and np.any(
                [hm.rstrip("_final.h5") in self.active_model.name for hm in self.list_of_submodels]):
            _xc = _x.copy()
            return stamp_region(dataset=_xc,
                                region_mask=create_region_mask(region=get_region(self.active_model.name),
                                                               reduce_overlap=True))

        else:
            return _x

    def get_headmodel_data(self, multi_level: bool, subset="test", dropnan=True, return_sics=False,
                           verbose=True):

        # Set subset vars correctly:
        _sset = subset if subset == "test" else "val"
        subset = subset if subset == "test" else "train"
        # since if subset = 'validation' OR = 'val': this can only be the training data for the headmodel

        # Loading data
        if not multi_level:
            if verbose:
                cprint(f"These are the predictions on the {_sset}-set of all basemodels"
                       f"{' of each sub-ensemble' if self.is_multilevel_ensemble() else ''}!", col='b')

            # Train (i.e. predictions on validation set!) OR test-set
            p2df = f"{p2models}{self.name}/{subset.lower()}_data_ensemble_head.sav"
            _x, _y = joblib.load(filename=p2df)  # xdata, ydata
            # if return_sics:
            _sset = _sset if _sset == "test" else "validation"
            sics_subset = np.array(load_datasplit(model_name=self.name)[_sset])

        else:  # multi_level:
            # Check arguments
            if not self.is_multilevel_ensemble():
                msg = "A non-multi-level ensemble has no multi-level headmodel data! Return None!"
                raise ValueError(msg)
                # cprint(msg, col='r')  # in case no Error is required
                # return None
            if verbose:
                cprint(f"These are the predictions on the {_sset}-set of all sub-ensembles!", col='b')
            data = get_sub_ensemble_predictions(model_name=self.name, subset=_sset, as_numpy=False)
            sics_subset = data.index.to_numpy()
            _y = data.pop(f"{_sset}_y").to_numpy().astype("float32")
            _x = data.to_numpy().astype("float32")
            if verbose:
                subens_pred_order = [col for col in data.columns if "pred" in col]
                cprint(f"Column-order of sub-ensemble predictions: {subens_pred_order}", col='y')

        # Dropnan
        if dropnan:
            (_x, _y), nan_idx = remove_nan_ensemble_data(_x, _y, return_nan_index=True)
            sics_subset = np.delete(arr=sics_subset, obj=[] if nan_idx is None else nan_idx)
            assert len(_x) == len(_y) == len(sics_subset), "Length must be equal, revisit implementation!"
            # TODO for regional multi-level ensemble there is an issue here

        # Return
        if return_sics:
            return _x, _y, sics_subset
        else:
            return _x, _y

    def get_headmodel_predictions(self, return_single_performs=False):
        """
        Get concatenated predictions of linear headmodels trained via CV.
        Works only for multi-level ensembles.
        Note that the order of the returned data is different to self.get_headmodel_data(...) due to the
        randomization in the CV split.
        The corresponding mixing indices can be found in the dict returned via
        self.get_headmodel(..., cv=True)
        """
        if not self.is_multilevel_ensemble():
            cprint("This works only for multi-level ensembles", 'r')
            return None

        cprint("Get predictions of linear headmodel which was trained via cross-validation...", 'b')
        # Note that the order differs

        xdata, ydata, sics = self.get_headmodel_data(multi_level=True, subset="test", dropnan=True,
                                                     return_sics=True, verbose=False)
        cv_headmodels = self.get_headmodel(head_model_type="linear", multi_level=True,
                                           cv=True)["head_models"]
        # dict_keys(['head_models', 'data_indices'])

        split_indices = self.get_headmodel_cv_sort_indices(concat=False)
        sort_indices = np.concatenate(split_indices)  # == self.get_headmodel_cv_sort_indices(concat=True)
        # necessary since split_in_n_bins() changes order when remainders are present

        # # Collect predictions
        # Concatenate all CV test-sets (of best splitset), and the corresponding model predictions
        all_preds = None
        all_testy = None  # only for testing

        maes = []  # Collect performances per split
        r2s = []
        for sp in range(len(split_indices)):
            test_indices = split_indices[sp].copy()
            _testx = xdata[test_indices].copy()
            _testy = ydata[test_indices].copy()
            _preds = cv_headmodels[sp].predict(_testx).copy()

            # Collect predictions over different splits (note different orders)
            all_preds = _preds if all_preds is None else np.concatenate([all_preds, _preds])
            all_testy = _testy if all_testy is None else np.concatenate([all_testy, _testy])
            # Note: Adapt order of data: xdata[sort_indices] ~ ydata[sort_indices] ~ all_preds

            maes.append(np.mean(np.abs(_preds - _testy)))  # MAE per split
            r2s.append(cv_headmodels[sp].score(_testx, _testy))  # R2
            # print("len test:", len(test_indices))
        cprint(f"MAE = {np.mean(maes):.2f} (r2 = {np.mean(r2s):.2f})", col='b')

        # Return as dataframe
        pred_df = pd.DataFrame({"predictions": all_preds, "sics_sorted": sics[sort_indices],
                                "target": all_testy})

        if return_single_performs:
            return pred_df, maes, r2s
        else:
            return pred_df

    def get_headmodel_cv_sort_indices(self, concat=True):
        if not self.is_multilevel_ensemble():
            cprint("This works only for multi-level ensembles", 'r')
            return None

        dataindices = self.get_headmodel(head_model_type="linear", multi_level=True,
                                         cv=True)["data_indices"]
        # dict_keys(['head_models', 'data_indices'])

        split_indices = split_in_n_bins(a=dataindices, attribute_remainder=True)
        if concat:
            sort_indices = np.concatenate(split_indices)
            return sort_indices
        else:
            return split_indices

    def get_predictions(self, subset="test", submodel_name=None, verbose=True):

        if self.active_model is None and submodel_name is None:
            cprint("Activate a submodel first OR provide the name of the sub/head-model!", col='r')
            return None

        subset = subset.lower()
        subset = "val" if "val" in subset else "test" if "test" in subset else None
        assert subset is not None, "Subset must be 'val' OR 'test'!"

        if isinstance(self.active_model, TrainedEnsembleModel) or "_ens" in submodel_name:

            # Return predictions of linear headmodel of given subensemble
            data = get_sub_ensemble_predictions(model_name=self.name, subset=subset, as_numpy=False)
            # Since order of predictions (columns) is not necessarily the order of self.list_of_submodels
            # we must extract the right prediction column

            # Define MRI (and Region) to drop from table:
            dropseq = [seq for seq in mri_sequences if seq.upper() not in submodel_name]
            dropreg = [reg.upper()[0:3] for reg in brain_regions.keys() if reg.upper()[
                                                                           0:3] not in submodel_name]

            # Throw out sequences (and regions) which do not match the given sub-ensemble
            keepcols = data.columns.to_list()
            for dropper in [dropseq, dropreg]:
                for d in dropper:
                    while True:
                        for i, c in enumerate(keepcols):
                            if d in c:
                                keepcols.pop(i)
                                break
                        else:
                            break

            if len(keepcols) > 2:
                # Remove "pred" column if required sequence(-region pair) is found in other column
                keepcols.pop(np.where(np.array(keepcols) == "pred")[0][0])

            # Sort analogous to return for basemodel predictions
            data = data[sorted(keepcols)]  # 'pred...' before 'test/val_y'
            data.reset_index(inplace=True)  # make SIC-index to column

            if verbose:
                cprint(f"Return predictions of linear head-model of sub-ensemble:\n"
                       f"{self.name}/{submodel_name}", col='b')
                print("To get predictions of all sub-ensembles, use:\n\tself.get_headmodel_data() OR ...")
                print("To get predictions of a requested base models, use:\n\t"
                      "self.active_model.get_predictions(submodel_name=BASE_MODEL_NAME)\n")

            return data

        if self.active_model is None:
            pred_file = crop_model_name(submodel_name)
        else:
            pred_file = self.active_model.name.split("/")[-1]
        pred_file += f"_{subset}_predictions.csv"

        for file in os.listdir(self.path2ensemble):
            if file == pred_file:
                file = pd.read_csv(os.path.join(self.path2ensemble, file))
                break
        else:
            file = None
            print(f"No {subset}-prediction file was found for '{self.active_model.name}'.")
            # TODO compute the file

        return file

    def get_mri_sequence(self):
        if self.active_model is None and "sequence-ens" in self.name:
            cprint("Activate a submodel first, since you operate with a sequence-ensemble!", 'r')
        elif "sequence-ens" in self.name:
            return self.active_model.name.split("_")[-2]
        else:
            for seq in mri_sequences:
                if seq.upper() in self.name:
                    return seq
            else:
                print("No indication for other than T1 MRI sequence found! Return 't1'!")
                return "t1"

    def get_region(self):

        if self.is_region_ensemble() and self.is_multilevel_ensemble():
            if self.active_model is not None:
                return [r for r in brain_regions.keys() if r[:3].upper() in self.active_model.name][0]
            else:
                cprint("Activate a submodel first to determine the region it is trained on!", 'r')
                return None
        elif self.is_region_ensemble() and not self.is_multilevel_ensemble():
            if "GrandREG" in self.name:  # in case it is a sub-ensemble of Grand-Ensemble
                return [r for r in brain_regions.keys() if r[:3].upper() in self.name][0]
            elif self.active_model is not None:
                return [r for r in brain_regions.keys() if r in self.active_model.name][0]
            else:
                cprint("Activate a submodel first to determine the region it is trained on!", 'r')
                return None
        else:
            cprint("Model was trained on whole brain data, hence there is no specific region!", 'r')
            return None

    def get_n_params(self, return_n=False, verbose=True):
        if self.is_multilevel_ensemble():
            n_subens = len(self.list_of_submodels)  # n sub-ensembles
            sub_ens = self.get_submodel(self.list_of_submodels[0])
            n_bm = len(sub_ens.list_of_submodels)  # n basemodels
            bm = sub_ens.get_submodel(sub_ens.list_of_submodels[0])  # load one basemodel to count params
            n_params_bm = bm.count_params()  # n params in a basemodel
            prt_line = f"{n_subens} sub-ensembles with each "  # extra text for printing

        else:
            n_subens = 1  # 1 for later multiplication, actually zero
            n_bm = len(self.list_of_submodels)
            bm = self.get_submodel(self.list_of_submodels[0])
            n_params_bm = bm.count_params()  # n params per basemodel
            prt_line = ""

        # Calculate all params
        all_n_params = n_params_bm * n_bm * n_subens

        if verbose:
            cprint(f"'{self.name}' has {prt_line}{n_bm} basemodels. "
                   f"Each basemodel has {n_params_bm} parameters. \n "
                   f"Hence, the whole ensemble has {all_n_params} parameters (excl. headmodel(s)).", 'b')

        if return_n:
            return all_n_params


def load_trained_model(model_name=None):
    if model_name:
        if os.path.isdir(p2models + model_name):
            # Load & return ensemble model
            return TrainedEnsembleModel(model_name=model_name)

        else:
            # Load single model
            # e.g., 2020-01-13_14-05_MRIkerasNetAGE_MNI_BinClassi_final.h5
            if ".h5" not in model_name:
                if "_final" not in model_name:
                    model_name += "_final.h5"
            return keras.models.load_model(p2models + model_name)

    else:

        cprint("Note: If you browse for an ensemble model, just choose a random submodel from the "
               "respective ensemble model folder", 'y')

        path2model = browse_files(p2models, "H5")

        if "_ens" in path2model or "region-ens" in path2model:
            _model_name = path2model.split("/")[-2]

        else:
            _model_name = path2model.split("/")[-1]

        return load_trained_model(model_name=_model_name)


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Model ensembles
def prep_headmodel_data(model_name, verbose=False):
    """For training on base model predictions"""

    # Set path do predictions of base models
    p2pred = os.path.join(p2models, model_name)

    # Check first whether data is there already or can be computed at all.
    if 'train_data_ensemble_head.sav' in os.listdir(
            p2pred) and "test_data_ensemble_head.sav" in os.listdir(p2pred):
        if verbose:
            cprint(f"Train & Test data for headmodel of {model_name} already computed.", "b")
        return None

    # Head model train data are val predictions
    p2traindf = os.path.join(p2models, model_name, "train_data_ensemble_head.sav")
    p2testdf = os.path.join(p2models, model_name, "test_data_ensemble_head.sav")

    for sset, p2sav in zip(["val", "test"], [p2traindf, p2testdf]):

        pred_files = find(fname=f"_{sset}_predictions.csv", folder=p2pred, exclusive=False,
                          fullname=False, verbose=verbose)

        if pred_files is not None:
            pred_files.sort()
        else:
            if verbose:
                cprint(f"No prediction files found for base models of {model_name}!", 'y')
            return None

        # Load single prediction tables and join them in one bigger table
        pred_tabs = None
        for i, file in enumerate(pred_files):
            train_tab = pd.read_csv(filepath_or_buffer=file, index_col="sic")
            if pred_tabs is None:
                pred_tabs = train_tab
            else:
                pred_tabs = pred_tabs.join(other=train_tab, on="sic", rsuffix=i)

        # Find most complete y-col
        max_col = None
        for col in pred_tabs.columns:
            if "_y" in col:
                if max_col is None:
                    max_col = col
                else:
                    max_col = col if pred_tabs[col].count() > pred_tabs[max_col].count() else max_col

                if col != max_col:  # Remove y-data from df which is less complete
                    pred_tabs.drop(columns=col, inplace=True)

        # Pull ydata
        ydata = pred_tabs.pop(max_col).to_numpy().astype(np.float32)  # (N, )
        x_ensemble = pred_tabs.to_numpy().astype(np.float32)  # (N, M)

        # Save training/test data for ensemble head
        joblib.dump(value=(x_ensemble, ydata), filename=p2sav)
        if verbose:
            print(f"\nSaved head-model {sset} data in {p2sav}\n")


def get_sub_ensemble_predictions(model_name: str, subset: str, as_numpy: bool, verbose=False):
    """Compute the predictions of all sub-ensembles of grand-ensemble in given sub-set"""

    subset = subset.lower()
    assert subset in ["val", "validation", "test"], \
        "sub-ensemble predictions are only available on the validation or test set!"

    if subset == "validation":
        subset = "val"
    p2pred = os.path.join(p2models, model_name)
    p2sav = os.path.join(p2pred, f"all_subens_preds_on_{subset}set.sav")
    max_col = f"{subset}_y"

    if os.path.isfile(p2sav):
        pred_tabs = joblib.load(filename=p2sav)

    else:
        # Find sub-ensembles
        pred_tabs = None  # init
        i = -1  # indexer
        _sset = subset if subset == "test" else "train"

        for sub_dir in os.listdir(p2pred):
            p2sub_ens = os.path.join(p2pred, sub_dir)
            if os.path.isdir(p2sub_ens):
                i += 1

                # Load sub-ensemble
                sub_ens = TrainedEnsembleModel(model_name=p2sub_ens.lstrip(p2models))

                # Use only linear headmodel
                sub_ens_hm = sub_ens.get_headmodel(head_model_type="linear", multi_level=False, cv=False)

                # Load train or test data of subensemble
                sub_hm_data = sub_ens.get_headmodel_data(multi_level=False, subset=_sset, dropnan=False,
                                                         verbose=False)
                # There are no NaNs in this data

                # Get also the corresponding SICs, since data-length between sub-ensembles can differ
                sics_data = pd.read_csv(
                    os.path.join(sub_ens.path2ensemble,
                                 f"0_model_{subset}_predictions.csv"),  # for all base models the same
                    index_col="sic")
                sub_ens_pred_tab = sics_data.drop(columns=f"{subset}_pred", inplace=False)

                assert np.all(sub_hm_data[1] == sics_data[f"{subset}_y"].to_numpy()), \
                    "must be the same ..."  # [TESTED]

                # Get sub-ensemble predictions on subset
                sub_ens_pred = sub_ens_hm.predict(sub_hm_data[0])
                sub_ens_pred_tab["pred"] = sub_ens_pred
                # This will become training data for top-headmodel

                if verbose:
                    # TODO if following will be kept, do also for classification
                    cprint(f'{subset.title()}-performance of sub-ensemble {sub_ens.name.split("/")[-1]}: '
                           f'{np.mean(np.abs(sub_hm_data[1] - sub_ens_pred)):.3f}', 'y')  # TEST

                # Save predictions of all sub-ensembles in common df
                if pred_tabs is None:
                    pred_tabs = sub_ens_pred_tab
                else:
                    sfx = f"_{sub_ens.get_mri_sequence()}"
                    sfx += sub_ens.get_region().upper()[:3] if sub_ens.is_region_ensemble() else ""
                    pred_tabs = pred_tabs.join(other=sub_ens_pred_tab, on="sic", rsuffix=sfx)

        # Find most complete y-col
        for col in pred_tabs.columns:
            if "_y" in col:
                max_col = col if pred_tabs[col].count() > pred_tabs[max_col].count() else max_col

                if col != max_col:  # Remove y-data from df which is less complete
                    pred_tabs.drop(columns=col, inplace=True)

    # Pull ydata
    if as_numpy:
        ydata_ = pred_tabs.pop(max_col).to_numpy().astype(np.float32)  # (N, )
        x_ensemble_ = pred_tabs.to_numpy().astype(np.float32)  # (N, M)

        return x_ensemble_, ydata_

    else:
        return pred_tabs


def prep_grand_headmodel_data(model_name, verbose=True):
    """
    Instead of training the top-headmodel with all single basemodel predictions, we train it with the
    predictions of its sub-ensembles' on either:
        * the validation set and validate it with the predictions on the test set (might overfit)
        * or via cross-validation with the predictions of its sub-ensembles' on the test-set only

    Here we use only the linear headmodels.
    :param model_name: name of grand-ensemble (i.e., multi-level ensemble)
    :param verbose: bool
    """

    # Set path to grand-ensemble folder
    p2pred = os.path.join(p2models, model_name)

    # Top-headmodel train & test data are predictions of sub-ensembles
    # Training on val data could lead to slight overfit, but we keep both
    for sset in ["val", "test"]:
        data = get_sub_ensemble_predictions(model_name=model_name, subset=sset, as_numpy=False,
                                            verbose=verbose)

        # Save training/test data for ensemble head
        p2sav = os.path.join(p2pred, f"all_subens_preds_on_{sset}set.sav")
        joblib.dump(value=data, filename=p2sav)
        if verbose:
            print(f"\nSaved top-headmodel data in {p2sav}\n")


def pred_per_basemodel(model_name, xdata):
    cprint("\nPreparing data for ensemble head ...", 'b')

    # Get paths to all sub-models
    sub_model_names = [mo for mo in os.listdir(f"{p2models}{model_name}/") if "final.h5" in mo]
    sub_model_names.sort()

    is_sequence_ens = "sequence-ens" in model_name
    xdata_all = xdata.copy() if is_sequence_ens else None
    n_data = len(list(xdata.values())[0]) if is_sequence_ens else len(xdata)

    # Gather predictions of each sub-model
    stack_x = None  # init
    for m_name in sub_model_names:
        sub_model = load_trained_model(model_name=f"{model_name}/{m_name}")
        # sub_models.append(sub_model)

        # Check whether region-ensemble
        if "region-ens" in model_name:
            region = get_region(_model_name=m_name)
            if region is None:
                raise ValueError(f"'region' for sub-model '{m_name}' not found!")

            # Adapt data transformer
            region_mask = create_region_mask(region=region, reduce_overlap=True)

            def dt(_x):
                """dt = lambda _x: stamp_region(dataset=_x, region_mask=region_mask)"""
                _xc = _x.copy()
                return stamp_region(dataset=_xc, region_mask=region_mask)

        else:
            def dt(_x):
                """Placeholder function for region-ensemble (dt = lambda _x: _x  # identity mapping)"""
                return _x

        nan_idx = None
        if is_sequence_ens:
            seq = m_name.split("_")[1]  # e.g. 't1', or 'flair'
            assert seq in mri_sequences, f"Unknown sequence '{seq}' for submodel '{m_name}'!"
            xdata = xdata_all.pop(seq)

            if xdata.ndim == 2:  # e.g. shape=(201, 1)
                # This is when there is NaN data within the training set
                nan_idx = []
                for i in range(xdata.shape[0]):
                    if xdata[i, -1] is None:
                        nan_idx.append(i)
                xdata = np.delete(arr=xdata, obj=nan_idx)
                xdata = np.stack(xdata).astype(xdata[0].dtype)
                xdata = np.expand_dims(xdata, -1)

        # Predict on data
        sub_m_pred = sub_model.predict(x=dt(xdata), batch_size=1)  # verbose=0  # y-hat

        if isinstance(nan_idx, list):
            for inan in nan_idx:
                sub_m_pred = np.insert(arr=sub_m_pred, obj=inan, values=np.nan)
            sub_m_pred = sub_m_pred.reshape((len(sub_m_pred), 1))

        if stack_x is None:
            stack_x = sub_m_pred

        else:
            stack_x = np.dstack((stack_x, sub_m_pred))

    # Flatten predictions to [n_samples, n_submodels x probabilities/predictions]
    stack_x = stack_x.reshape((stack_x.shape[0], stack_x.shape[1] * stack_x.shape[2]))

    return stack_x


def remove_nan_ensemble_data(_xdata, _ydata, return_nan_index=False):
    nan_idx = None
    if np.isnan(_xdata).any():
        nan_idx = np.where(np.isnan(_xdata).any(axis=1))[0]
        # Delete rows
        _ydata = _ydata[~np.isnan(_xdata).any(axis=1)]  # start with y-values first
        _xdata = _xdata[~np.isnan(_xdata).any(axis=1)]

    if return_nan_index:
        return (_xdata, _ydata), nan_idx
    else:
        return _xdata, _ydata


def build_keras_ensemble(model_name):
    # Check: https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

    sub_model_names = find(fname="final.h5", folder=f"{p2models}{model_name}/", exclusive=False,
                           fullname=False, verbose=True)

    sub_model_names.sort()

    n_models = len(sub_model_names)

    cl = "BinClass" in model_name  # classification task or not

    merge = visible = keras.layers.Input(shape=(n_models * (2 if cl else 1),),
                                         name="stacked_submodel_predictions")
    # == input (i.e. prediction of submodels)

    # Define head model
    hidden = keras.layers.Dense(units=n_models, activation='relu')(merge)
    output = keras.layers.Dense(units=2 if cl else 1,
                                activation='softmax' if cl else 'linear')(hidden)

    ensemble_model = keras.models.Model(inputs=visible, outputs=output)

    # # Plot graph of ensemble
    # keras.utils.plot_model(ensemble_model, show_shapes=True,
    #                        to_file=f'{p2models}{model_name}/{model_name}_ensemble_graph_nonlin_.png')

    # Compile
    ensemble_model.compile(loss='mse',  # 'categorical_crossentropy'
                           optimizer='adam',
                           metrics=["accuracy"] if cl else ["mae"])

    # Print model summary
    ensemble_model.summary()

    return ensemble_model


def fit_grand_ensemble(model_name, cross_val=True, save=False):
    """
    Fit the grand head-model (linear) on predictions of sub-ensembles.
    :param model_name: Name of grand ensemble
    :param cross_val: True: Cross-validation to train & evaluate the grand head-model on test-set only
                      False: Train on sub-ensemble predictions on validation set and test on test-set
    :param save: Save head-model(s), and data shuffle indices (in case of cross_val=True)
    :return: head model(s) and data shuffle indices (in case of cross_val=True)
    """

    # Load data
    xdata, ydata = get_sub_ensemble_predictions(model_name=model_name,
                                                subset="test" if cross_val else "val",
                                                as_numpy=True,
                                                verbose=False)

    # Check wether data is complete
    if np.isnan(xdata).any():
        (xdata, ydata), nan_idx = remove_nan_ensemble_data(_xdata=xdata, _ydata=ydata,
                                                           return_nan_index=True)

        cprint(f"Data for grand-head-model training in '{model_name}' has {len(nan_idx)} rows with "
               f"NaN's. These rows with (partially) missing data were deleted ...", 'y')

    if cross_val:

        # Fit head-model
        cprint(f"\nFit linear ensemble head via cross-validation on test-set predictions ...", 'b')
        mae_best = np.inf  # init
        dataindices_best = None  # init
        # Repeat to check stability of result and to find optimal fit
        for i in range(5000):

            # Split data in 5 folds: create corresponding indices
            dataindices = np.arange(len(ydata))
            np.random.shuffle(dataindices)
            split_indices = split_in_n_bins(a=dataindices, n=5, attribute_remainder=True)
            # Seed important to reconstruct the split for unequal subset sizes

            # Init savers
            cv_headmodels = []
            results2_mae = []
            results2_r2 = []

            # Run trough all splits
            for sp in range(len(split_indices)):

                # Split data: create test-data
                test_indices = split_indices[sp].copy()
                _testx = xdata[test_indices].copy()
                _testy = ydata[test_indices].copy()
                # print("len test:", len(test_indices))

                # Pull train-data
                try:
                    train_indices = np.concatenate(np.delete(split_indices, sp)).copy()
                except ValueError:
                    train_indices = np.delete(split_indices, sp).copy()
                _trainx = xdata[train_indices].copy()
                _trainy = ydata[train_indices].copy()
                # print("len train:", len(train_indices))

                # Define headmodel and fit
                cv_headmodel = Ridge(alpha=1.)
                cv_headmodel.fit(X=_trainx, y=_trainy)
                cv_headmodels.append(cv_headmodel)
                mae = np.mean(np.abs(cv_headmodel.predict(X=_testx) - _testy))  # MAE
                results2_mae.append(mae)
                r2 = cv_headmodel.score(X=_testx, y=_testy)  # R2
                results2_r2.append(r2)

            # Save best performing split
            if np.mean(results2_mae) < mae_best:
                mae_best = np.mean(results2_mae)
                r2_best = np.mean(results2_r2)
                dataindices_best = copy.copy(dataindices)
                models_best = copy.copy(cv_headmodels)
            print(f"MAE = {mae_best:.4f} | r2 = {r2_best:.4f} | {(i + 1) * 100 / 5000:.2f} %", end="\r")

        # # Alternative:
        # from sklearn.model_selection import cross_validate
        # cv_headmodel = Ridge()
        # results = cross_validate(cv_headmodel, X=xdata, y=ydata,
        #                          scoring=('r2', 'neg_mean_absolute_error'),
        #                          cv=5,  # number of splits
        #                          return_train_score=True, return_estimator=True)
        # # print(-1*results["train_neg_mean_absolute_error"].mean())
        # print("mae =", -1 * results["test_neg_mean_absolute_error"].mean())
        # # print(results['train_r2'].mean())
        # print("r2 =", results['test_r2'].mean())

        # Prep to save
        on_top_model = {"head_models": models_best,
                        "data_indices": dataindices_best}  # split_in_n_bins(a=dataindices_best, 5, True)

    else:  # sub-ens predictions on val-set: training data | predictions on test-set: test data

        # Fit head-model
        cprint(f"\nFit linear headmodel of grand ensemble ...", 'b')

        # Head model
        on_top_model = Ridge(alpha=1.)  # LinearRegression()
        # cl = "BinClass" in model_name  # classification task or not
        on_top_model.fit(X=xdata, y=ydata)

    # Save ensemble head-model
    if save:
        cprint(f"\nSave linear grand-ensemble headmodel ...", 'b')
        joblib.dump(value=on_top_model,
                    filename=f"{p2models}{model_name}/"
                             f"ensemble_multilevel_{'cv_' if cross_val else ''}headmodel_lin.sav")

    return on_top_model


def fit_ensemble(model_name, xdata=None, ydata=None, valdata=None, head_model_type="linear",
                 multi_level=False, save=False):
    is_seq_ens = "sequence-ens" in model_name
    head_m_type = ["linear", "nonlinear"] if head_model_type.lower() == "both" else [head_model_type]
    head_m = []  # init
    xtrain_ensemble = None  # init

    # Paths to ensemble head data
    if multi_level:
        assert "Grand" in model_name, "'multi_level' asserts fitting top-head-model of grand-ensmble!"
        prep_grand_headmodel_data(model_name=model_name, verbose=True)
        return fit_grand_ensemble(model_name=model_name, cross_val=True, save=save)

    else:
        p2traindf = os.path.join(p2models, model_name, "train_data_ensemble_head.sav")
        p2testdf = os.path.join(p2models, model_name, "test_data_ensemble_head.sav")

        prep_headmodel_data(model_name=model_name, verbose=True)
        # Computes *.sav files quickly if possible

        for idx, mh in enumerate(head_m_type):

            # Get on-top / ensemble head model
            if mh.lower() == "linear":
                on_top_model = Ridge(alpha=1.)  # LinearRegression()

            elif mh.lower() == "nonlinear":
                on_top_model = build_keras_ensemble(model_name=model_name)

            else:
                raise ValueError("'head_model_type' unknown. Must be 'linear' OR 'nonlinear' OR 'both'.")

            # cl = "BinClass" in model_name  # classification task or not

            # Load training data if not given
            if not os.path.isfile(p2traindf) or not os.path.isfile(p2testdf):
                if xdata is None and ydata is None and idx == 0:
                    if is_seq_ens:
                        # For sequence ensemble
                        train_seq_ens = {}  # init
                        val_seq_ens = {}  # init
                        ydatall = None
                        ytestall = None
                        for seq in mri_sequences:
                            xdata, ydata, xtest, ytest = get_model_data(
                                model_or_name=model_name, lrp_prep=False, only_data=True,
                                # use val-data to train head-model, since 'unseen' by sub-models:
                                all_data=True, mri_sequence=seq, return_nones=True)[2:]

                            ydatall = ydata if (ydatall is None or len(ydata) > len(ydatall)) else ydatall
                            ytestall = ytest if (ytestall is None or len(ytest) > len(ytestall)) else ytestall

                            # Collect data in dicts
                            train_seq_ens[seq] = xdata
                            val_seq_ens[seq] = xtest  # valdata

                        xdata, ydata = train_seq_ens, ydatall
                        valdata = (val_seq_ens, ytestall)

                    else:
                        xdata, ydata, xtest, ytest = get_model_data(
                            model_or_name=model_name, lrp_prep=False, only_data=True,
                            # use val-data to train head-model, since 'unseen' by sub-models:
                            all_data=True, mri_sequence=None)[2:]  # train: 0:2, val: 2:4, test: 4:
                        valdata = (xtest, ytest)

                # Prepare training data (i.e. output of sub-models) for ensemble head-model
                if xtrain_ensemble is None:
                    xtrain_ensemble = pred_per_basemodel(model_name=model_name, xdata=xdata)
                    # Save training data for ensemble head
                    joblib.dump(value=(xtrain_ensemble, ydata), filename=p2traindf)

                if (valdata is not None) and mh == "nonlinear":
                    valdata = (pred_per_basemodel(model_name=model_name, xdata=valdata[0]), valdata[1])
                    # Save validation/test data for ensemble head
                    joblib.dump(value=valdata, filename=p2testdf)

            else:
                # Load the 'ensembled' train & test data
                xtrain_ensemble, ydata = joblib.load(filename=p2traindf)
                valdata = joblib.load(filename=p2testdf)

            # Check wether data is complete
            if np.isnan(xtrain_ensemble).any():
                (xtrain_ensemble, ydata), nan_idx = remove_nan_ensemble_data(_xdata=xtrain_ensemble,
                                                                             _ydata=ydata,
                                                                             return_nan_index=True)

                cprint(f"Training data of head-model in '{model_name}' has {len(nan_idx)} (of "
                       f"{xtrain_ensemble.shape[0]}) rows with NaN's. These rows with (partially) "
                       f"missing data were deleted ...", 'y')

            # Fit head-model
            cprint(f"\nFit {mh} ensemble head ...", 'b')

            if mh == "linear":
                on_top_model.fit(X=xtrain_ensemble, y=ydata)

            else:  # nonlinear

                # Check valdata
                if np.isnan(valdata[0]).any():
                    valdata, nan_idx = remove_nan_ensemble_data(_xdata=valdata[0],
                                                                _ydata=valdata[1],
                                                                return_nan_index=True)

                    cprint(f"Validation data of head-model in '{model_name}' has {len(nan_idx)} (of "
                           f"{valdata[0].shape[0]}) rows with NaN's. These rows with (partially) missing "
                           f"data were deleted ...", 'y')

                callbacks = [keras.callbacks.ModelCheckpoint(
                    filepath=f"{p2models}{model_name}/ensemble_headmodel_nonlin.h5",
                    save_best_only=True,
                    save_weights_only=False,
                    monitor="val_loss",
                    period=5,
                    verbose=False),
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)]

                on_top_model.fit(x=xtrain_ensemble,
                                 y=ydata,
                                 batch_size=int(len(ydata) / 20),
                                 epochs=2000,
                                 verbose=False,
                                 callbacks=callbacks,
                                 validation_data=valdata)  # since unseen

            # Save ensemble head-model
            if save:
                cprint(f"\nSave {mh} ensemble model ...", 'b')
                if mh == "linear":
                    joblib.dump(value=on_top_model,
                                filename=f"{p2models}{model_name}/"
                                         f"ensemble_headmodel_lin.sav")
                else:  # nonlinear
                    on_top_model.save(
                        filepath=f"{p2models}{model_name}/"
                                 f"ensemble_headmodel_nonlin.h5")  # HDF5

            head_m.append(on_top_model)

        return head_m


def eval_ensemble(model_name, head_model_list=None, head_type="both", multi_level=False):
    # Evaluate on test set
    htypes = ["linear", "nonlinear"] if head_type.lower() == "both" else [head_type.lower()]
    class_task = is_binary_classification(model_name)

    if head_model_list is None:
        head_model_list = []
        for ht in htypes:
            head_model_list.append(load_ensemble(model_name=model_name, head_model_type=ht,
                                                 multi_level=multi_level))

    performs = []  # init

    if not isinstance(head_model_list, dict):
        if multi_level:
            p2testdf = f"{p2models}{model_name}/test_data_ensemble_multilevel.sav"  # this must be there
        else:
            p2testdf = f"{p2models}{model_name}/test_data_ensemble_head.sav"  # this must be there
        xtest_ensemble, ytest_ensemble = joblib.load(filename=p2testdf)  # _ == y_test

        xtest_ensemble, y_test = remove_nan_ensemble_data(xtest_ensemble, ytest_ensemble)

        for ht, ens_m in zip(htypes, head_model_list):
            ens_preds = ens_m.predict(xtest_ensemble)
            # r2 = ens_m.score(X=pred_per_basemodel(model_name=netname, xdata=x_test),
            #                  y=y_test)  # R^2 of the prediction

            if class_task:
                correct = (np.argmax(ens_preds, axis=1) == np.argmax(y_test, axis=1)) * 1
                performance = sum(correct) / len(correct)  # accuracy

            else:
                performance = np.abs(ens_preds.reshape(y_test.shape) - y_test).mean()  # MAE

            performs.append(performance)

            cprint(f"Test performance of ensemble '{model_name}' with {ht} head-model: "
                   f"MAE = {performance:.3f}", col='y')

    else:
        # For multi-level ensemble trained via cross_val

        if class_task:
            raise NotImplementedError("Not implemented for classification task yet.")

        # Load data
        xtest_ensemble, ytest_ensemble = get_sub_ensemble_predictions(model_name=model_name,
                                                                      subset="test", as_numpy=True,
                                                                      verbose=False)
        xtest_ensemble, y_test = remove_nan_ensemble_data(xtest_ensemble, ytest_ensemble)
        data_indices = head_model_list['data_indices']
        head_model_list = head_model_list['head_models']
        split_indices = split_in_n_bins(a=data_indices, n=5, attribute_remainder=True)

        results_mae = []
        results_r2 = []
        for sp in range(len(split_indices)):

            # Get model
            cv_headmodel = head_model_list[sp]

            # Split data: create test-data
            test_indices = split_indices[sp].copy()
            _testx = xtest_ensemble[test_indices].copy()
            _testy = y_test[test_indices].copy()

            # Pull train-data
            try:
                train_indices = np.concatenate(np.delete(split_indices, sp)).copy()
            except ValueError:
                train_indices = np.delete(split_indices, sp).copy()
            _trainx = xtest_ensemble[train_indices].copy()
            _trainy = y_test[train_indices].copy()
            # print("len train:", len(train_indices))

            # Performance
            mae = np.mean(np.abs(cv_headmodel.predict(X=_testx) - _testy))  # MAE
            results_mae.append(mae)
            r2 = cv_headmodel.score(X=_testx, y=_testy)  # R2
            results_r2.append(r2)

        # Aggregate performance across the splits
        performance_mae = np.mean(results_mae)
        performance_r2 = np.mean(results_r2)
        performs.append(performance_mae)

        cprint(f"Aggregated performance across cross-val-splits of ensemble '{model_name}' with "
               f"{htypes[0]} head-model MAE={performance_mae:.3f} (r2={performance_r2:.2f})", col='y')

    return performs


def load_ensemble(model_name, head_model_type="linear", multi_level=False, cv=False):
    if cv:
        if not multi_level or head_model_type != "linear":
            cprint("All cross-validated models are always linear (multi-level) ensemble headmodels", 'y')
            # Set args correctly:
            multi_level = True
            head_model_type = "linear"

    if head_model_type.lower() == "linear":
        ensemble_model = joblib.load(filename=f"{p2models}{model_name}/"
                                              f"ensemble_{'multilevel_' if multi_level else ''}"
                                              f"{'cv_' if cv else ''}headmodel_lin.sav")

    elif head_model_type.lower() == "nonlinear":
        ensemble_model = load_trained_model(
            model_name=f"{model_name}/ensemble_{'multilevel_' if multi_level else ''}headmodel_nonlin.h5")

    else:
        raise ValueError("'head_model_type' unknown. Must be 'linear' OR 'nonlinear'.")

    return ensemble_model


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Read & write results & model information

# TODO update to MRI-space-style
def extract_params_from_name(model_name):
    date, target = model_name.split("_MRIkerasNet")
    target = target.split("_")[0].lower()
    mri_type = "T1"  # TODO UPDATE
    mni = "MNI" in model_name
    intensity_scale = "(-1, 1)" if "N-11" in model_name else "(0, 1)"
    if "BGn." in model_name:
        background_noise = float(".".join(model_name.split("BGn")[-1].split("_")[0].split(".")[0:2]))
    else:
        background_noise = "BGn" in model_name
    augmentation = "augm" in model_name
    transformation_types = model_name.split("augm-")[-1].split("_")[0] if "augm-" in model_name else None
    classification_task = "BinClassi" in model_name
    corrected_for = model_name.split("CF")[-1] if "CF" in model_name else False
    ensemble = int(model_name.split("_ens")[-1]) if "_ens" in model_name else False

    return [date, model_name, target, mri_type, mni, intensity_scale, background_noise, augmentation,
            transformation_types, classification_task, corrected_for, ensemble]


def open_performance_table_html():
    # Check table out via
    from pivottablejs import pivot_ui
    from IPython.core.display import HTML
    import webbrowser

    perform_tab = pd.read_csv(p2resulttab)
    perform_tab_html = p2resulttab.split("csv")[0] + "html"
    pivot_ui(perform_tab, outfile_path=perform_tab_html)
    HTML(perform_tab_html)
    webbrowser.open("file://d/" + os.path.abspath(perform_tab_html), new=2)
    # os.system(f"open {p2resulttab.split('csv')[0] + 'html'}")  # only MacOs


def write_performance_table(model_name=None):
    perform_tab = pd.read_csv(p2resulttab)

    # Write model performance(s) in table
    for file in os.listdir(p2logs):
        if (model_name is None or model_name in file) and "results" in file:
            with open(p2logs + file, "r") as perform_file:
                perform = perform_file.readline()

            if "ens" in file:
                perform = perform.split("'linear': ")[1].split(", 'nonlinear': ")
                perform = str([float(perform[0]), float(perform[1].rstrip('\n'))])
            else:
                perform = ast.literal_eval(perform[perform.find("["):])[-1]

            varis = extract_params_from_name(model_name if model_name else file.split("_results")[0])
            varis.append(perform)
            perform_tab = perform_tab.append(dict(zip(perform_tab.columns.to_list(), varis)),
                                             ignore_index=True)

    # Save table
    perform_tab.drop_duplicates().to_csv(p2resulttab, index=False)  # (remove duplicates before)


def write_basemodel_performance_table(model_name, verbose=True):
    model_dir = os.path.join(p2logs, model_name)
    results = [fn for fn in os.listdir(model_dir) if fn.endswith(".txt")]
    results.sort()
    metric = "acc" if is_binary_classification(_model_name=model_name) else "MAE"

    # Init table
    result_tab = pd.DataFrame(data=np.zeros(len(results)),
                              index=[mn.rstrip("_results.txt") for mn in results],
                              columns=[metric])

    # Write results in table
    for i, bm_fn in enumerate(results):
        with open(os.path.join(model_dir, bm_fn), "r") as file:
            bm_result = file.readline()
            bm_result = float(bm_result.split(",")[-1].rstrip("]\n"))

        result_tab.iloc[i] = bm_result

    # Save table
    result_tab.to_csv(os.path.join(model_dir, "base_model_performance_table.csv"))

    # Summary
    if verbose:
        cprint(f"\n{model_name} base-models performance summary:", fm="ul")
        print(result_tab[metric].describe())


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Get data specific to a (trained) model
def is_binary_classification(_model_name):
    return "BinClass" in _model_name


def get_target(_model_name):
    return _model_name.split("Net")[1].split("_")[0].lower()


def get_classes(_model_name):
    return pred_classes(get_target(_model_name)) if is_binary_classification(_model_name) else None


def get_region(_model_name):
    for reg in brain_regions.keys():
        if f"_{reg.upper()[0:3]}_" in _model_name or f"_{reg}_" in _model_name:
            region = reg
            break
    else:
        region = None

    return region


def get_model_data(model_or_name, bg_noise=False, only_data=False, all_data=False, lrp_prep=True,
                   mri_sequence=None, return_sics=False, **kwargs):
    if isinstance(model_or_name, str):
        _model_name = model_or_name
        if lrp_prep: \
                _model = load_trained_model(model_name=model_or_name)
    else:
        _model = model_or_name
        _model_name = model_or_name.name

    if mri_sequence is None:
        _mri_type = "swi" if "SWI" in _model_name else "flair" if "FLAIR" in _model_name else "t1"
    else:
        _mri_type = mri_sequence

    _mri_space = "mni" if "MNI" in _model_name else "raw" if "RAW" in _model_name else "fs"
    _renorm_11 = True if "N-11" in _model_name else False  # norm -1,1
    norm = (0, 1)
    try:
        with open(os.path.join(p2logs, _model_name) + "_results.txt") as file:
            for line in file:
                if "norm=None" in line:
                    norm = None  # Currently, for raw data, norm could be None [STILL IN TESTING]
    except FileNotFoundError:
        if "ens" in _model_name:  # works for model-ensembles
            for dir_file in os.listdir(os.path.join(p2logs, _model_name)):
                if dir_file.endswith("_results.txt"):
                    with open(os.path.join(p2logs, _model_name, dir_file)) as file:
                        for line in file:
                            if "norm=None" in line:
                                norm = None
        else:
            raise NotImplementedError("'norm' of model data can't be inferred.")  # TEST
            # This shouldn't be necessary, since for non-ensembles, upper file should be there already

    _target = get_target(_model_name)
    _binary_cls = is_binary_classification(_model_name)
    _classes = get_classes(_model_name)
    _region = get_region(_model_name)

    if _binary_cls and lrp_prep:
        # TODO does not work for ensembles yet
        import innvestigate
        # Remove softmax layer for LRP
        # print("Last Acti. Fct:",_model.get_layer(name=_model.output_names[0]).activation.__name__)  # sm
        try:
            model_wo = innvestigate.utils.model_wo_softmax(_model)  # for neuron_selection_mode="index"
        except ValueError:
            for lyr in _model.layers:
                # Necessary due to naming issues of keras/iNNvestigate
                if "dense" in lyr.name:
                    lyr.name = lyr.name + "_w_sftmx"

            model_wo = innvestigate.utils.model_wo_softmax(_model)
        _model = model_wo
        # _model = innvestigate.utils.model_wo_softmax(_model)  # for neuron_selection_mode="index"
        # print("Last Acti. Fct:",_model.get_layer(name=_model.output_names[0]).activation.__name__)  #lin

    # # Load Data
    _sics_split = load_datasplit(model_name=_model_name.split("/")[0])  # split for potential grand ens

    # Compute target bias
    all_sics = [sic for sublist in _sics_split.values() for sic in sublist]

    # # Get _target_bias via study-table
    interim_t = _target if _target != "age" else "AGE_FS"
    study_tab = load_study_table(exclusion=True, specify_vars=[interim_t])
    study_tab = study_tab.loc[study_tab.SIC_FS.isin(all_sics)].reset_index(drop=True)
    _target_bias = study_tab[interim_t].mean()

    if not all_data:
        # Reduce to test-set:
        _sics_split.update({"train": [None], "validation": [None]})  # remove other data than "test"

        test_data = get_life_data(mri_sequence=_mri_type,
                                  target=_target, target_scale="softmax" if _binary_cls else "linear",
                                  region=_region,
                                  augment=None,
                                  mri_space=_mri_space, mri_scale=(-1, 1) if _renorm_11 else norm,
                                  split_dict=_sics_split, **kwargs)["test"]
        if bg_noise:
            test_data.add_bg_noise()

        _x_test, _y_test = test_data.to_keras(verbose=True)

        if only_data:
            if return_sics:
                return _x_test, _y_test, test_data.sics
            else:
                return _x_test, _y_test

        else:
            if return_sics:
                return _model, _x_test, _y_test, _target, _target_bias, _binary_cls, _classes, \
                       _mri_type, _mri_space, test_data.sics

            else:
                return _model, _x_test, _y_test, _target, _target_bias, _binary_cls, _classes, \
                       _mri_type, _mri_space

    else:

        train_data, val_data, test_data = get_life_data(
            mri_sequence=_mri_type, target=_target, target_scale="softmax" if _binary_cls else "linear",
            region=_region, augment=None, mri_space=_mri_space, mri_scale=(-1, 1) if _renorm_11 else norm,
            split_dict=_sics_split, **kwargs).values()

        if bg_noise:
            train_data.add_bg_noise()
            val_data.add_bg_noise()
            test_data.add_bg_noise()

        _x_train, _y_train = train_data.to_keras(verbose=True)
        _x_val, _y_val = val_data.to_keras(verbose=True)
        _x_test, _y_test = test_data.to_keras(verbose=True)

        sic_dict = {"train": train_data.sics,
                    "validation": val_data.sics,
                    "test": test_data.sics}

        if only_data:
            if return_sics:
                return _x_train, _y_train, _x_val, _y_val, _x_test, _y_test, sic_dict
            else:
                return _x_train, _y_train, _x_val, _y_val, _x_test, _y_test

        else:
            if return_sics:
                return _model, (_x_train, _y_train, _x_val, _y_val, _x_test, _y_test), \
                       _target, _target_bias, _binary_cls, _classes, _mri_type, _mri_space, sic_dict
            else:
                return _model, (_x_train, _y_train, _x_val, _y_val, _x_test, _y_test), \
                       _target, _target_bias, _binary_cls, _classes, _mri_type, _mri_space


# %% Copy heatmap data o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def prep_copy_heatmap_data(list_of_sics, model_name, only_aggregated_heatmaps=False, original_mri=False):
    """Prepare data for easy transfer, specifically for heatmap analysis"""
    from apply_LRP import p2intrprt_rslts
    from distutils.dir_util import copy_tree

    if original_mri:
        raise NotImplementedError("Copying of original corresponding MRI data not implemented yet!")

    p2rootsave = "./TEMP/transfer"
    if not os.path.isdir(p2rootsave):
        os.mkdir(p2rootsave)
    p2hms = os.path.join(p2intrprt_rslts, model_name)
    model = load_trained_model(model_name=model_name)

    if model.is_multilevel_ensemble():
        for subens_name in model.list_of_submodels:
            prep_copy_heatmap_data(list_of_sics=list_of_sics,
                                   model_name=os.path.join(model.name, subens_name),
                                   only_aggregated_heatmaps=only_aggregated_heatmaps,
                                   original_mri=original_mri)

    else:
        for sic in list_of_sics:
            sicpaths = find(fname=sic, folder=p2hms, typ="folder", exclusive=False, abs_path=True,
                            verbose=False)
            if sicpaths is None:
                continue
            for sicpath in sicpaths:
                if only_aggregated_heatmaps and "aggregated" not in sicpath:
                    continue
                p2sicsave = os.path.join(p2rootsave,
                                         sicpath.split(os.path.abspath(p2intrprt_rslts))[-1][1:])
                if not os.path.isdir(p2sicsave):
                    os.makedirs(p2sicsave)
                copy_tree(src=sicpath, dst=p2sicsave)

        cprint(f"\nData to copy for model '{model_name}' can be found in {p2rootsave}.", "b")


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


if __name__ == "__main__":
    # %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    # # # Run several model tests
    # good_datasplit = load_datasplit("good_split_n2016")  # now: N=2016

    # # Train on AGE (regression)
    # train_and_test_keras_model(target="age", mri_space="fs")
    # train_and_test_keras_model(target="age", mri_space="mni")

    # # Train on AGE (regression) with ReLUs (check then sum-relevance of LRP)
    # train_and_test_keras_model(target="age", mri_space="fs", split_dict=good_datasplit,
    #                            leaky_relu=False, bnorm=True)
    # train_and_test_keras_model(target="age", mri_space="mni", split_dict=good_datasplit,
    #                            leaky_relu=False)

    # # Train on AGE (regression): with other sequences and raw images (must be implemented)
    # train_and_test_keras_model(target="age", mri_type="flair", mri_space="raw", batch_size=1,prune=True)
    # train_and_test_keras_model(target="age", mri_type="flair", mri_space="fs")
    # train_and_test_keras_model(target="age", mri_type="swi", mri_space="raw", prune=True)
    # train_and_test_keras_model(target="age", mri_type="swi", mri_space="fs")
    # train_and_test_keras_model(target="age", mri_type="t1", mri_space="raw", prune=False)

    # # Train on AGE (regression): with different intensity scale (norm)
    # train_and_test_keras_model(target="age", mri_space="fs", norm=(-1, 1))
    # train_and_test_keras_model(target="age", mri_space="mni", norm=(-1, 1))

    # # Train on AGE (regression): with background noise
    # train_and_test_keras_model(target="age", mri_space="fs", bg_noise=True)
    # train_and_test_keras_model(target="age", mri_space="mni", bg_noise=True)
    # train_and_test_keras_model(target="age", mri_space="fs", bg_noise=.02)  # more noise
    # train_and_test_keras_model(target="age", mri_space="mni", bg_noise=.02)
    # train_and_test_keras_model(target="age", mri_space="mni", norm=(-1, 1), bg_noise=True)

    # # TODO Train on AGE (regression): with augmentation
    # train_and_test_keras_model(target="age", mri_space="fs", augment_data=True)
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True)
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, bg_noise=True)

    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=1)
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=100,
    #                            transform_types=["none"])
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=100)
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=500)
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=500,
    #                            transform_types=["translation", "rotation"])
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True, n_augment=1000)

    # # Train on AGE (regression): with augmentation, translation only
    # (if more transformations required, do, e.g. transform_types=["translation", "noise", "none"])
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True,
    #                            transform_types="translation", bg_noise=False)  # augment True
    # train_and_test_keras_model(target="age", mri_space="mni", augment_data=True,
    #                            transform_types="translation", bg_noise=True)

    # Train on AGE (regression): regions specific training
    # train_and_test_keras_model(target="age", mri_space="mni", region="cerebellum")
    # train_and_test_keras_model(target="age", mri_space="mni", region="subcortical")
    # train_and_test_keras_model(target="age", mri_space="mni", region="cortical")
    # train_and_test_keras_model(target="age", mri_space="mni", region="all")  # region ensemble

    # # Train on AGE (regression): Sequence-Ensemble
    # train_and_test_keras_model(target="age", mri_space="fs", mri_type="all", ensemble_head='both')
    # train_and_test_keras_model(target="age", mri_space="mni", mri_type="all", ensemble_head='both')
    # train_and_test_keras_model(target="age", mri_space="raw", mri_type="all", ensemble_head='both',
    #                            norm=None,  # prob.better due to huge FLAIR (works now with float16)
    #                            # [Works, but performance seems to be lower (maybe numerical reasons)]
    #                            batch_size=1)  # b-size necessary since FLAIR too big for GPUs
    # train_and_test_keras_model(target="age", mri_space="raw", mri_type="all", ensemble_head='both',
    #                            norm=(0, 1), batch_size=1)  # b-size necessary since FLAIR too big 4 GPU

    # # Run below Train on AGE (regression): with ensemble model
    # train_and_test_keras_model(target="age", mri_space="mni", ensemble=10, ensemble_head='both')
    # train_and_test_keras_model(target="age", mri_space="fs", ensemble=10, ensemble_head='both')
    # train_and_test_keras_model(target="age", mri_space="mni", ensemble=10, class_task=True,
    #                            ensemble_head='both')
    # train_and_test_keras_model(target="age", mri_type="T1", mri_space="fs", ensemble=10,
    #                            ensemble_head='both', split_dict=good_datasplit,
    #                            leaky_relu=False)

    # # TODO Run Grand Ensemble (use same datasplit for both ensembles)
    # This is an extended version of "2020-07-23_17-33_MRIkerasNetAGE_MNI_region-ens"  # MAE=3.6, N=1954
    # good_datasplit = load_datasplit("good_split_n2016")  # now: N=2016
    #
    # train_grand_ensemble(target="age", regional=False,
    #                      data_split=good_datasplit,  # = None  # uncomment 'Find model_name' below
    #                      n_basemodels=10)
    # train_grand_ensemble(target="age", regional=False,
    #                      data_split=good_datasplit,  # = None  # uncomment 'Find model_name' below
    #                      n_basemodels=10, leaky_relu=False)

    #
    # # # Find model_name of computed model
    # # prev_ens_name = find(fname="Grand_ens10", folder=p2logs, exclusive=True, fullname=False,
    # #                           verbose=True).split("/")[-1].rstrip("_results.txt")
    # #
    # # print("prev_ens_name:", prev_ens_name)
    #
    # # Train the grand regional ensemble with the same datasplit
    # train_grand_ensemble(target="age", regional=True,
    #                      data_split=good_datasplit,  # load_datasplit(model_name=prev_ens_name)
    #                      n_basemodels=5)

    # train_grand_ensemble(target="age", regional=True,
    #                      data_split=good_datasplit,  # load_datasplit(model_name=prev_ens_name)
    #                      n_basemodels=5, leaky_relu=False)

    # # Train on AGE (binary classification): old vs. young
    # train_and_test_keras_model(target="age", mri_space="fs", class_task=True)
    # train_and_test_keras_model(target="age", mri_space="mni", class_task=True)
    # train_and_test_keras_model(target="age", mri_space="mni", class_task=True,
    #                            bg_noise=True)  # with BG-noise

    # %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # # # Train on BMI (regression)
    # train_and_test_keras_model(target="BMI", mri_space="fs")
    # train_and_test_keras_model(target="BMI", mri_space="mni")

    # # Train on BMI (regression): with different intensity scale (norm)
    # train_and_test_keras_model(target="BMI", mri_space="fs", norm=(-1, 1))
    # train_and_test_keras_model(target="BMI", mri_space="mni", norm=(-1, 1))

    # # Train on BMI (regression): with background noise
    # train_and_test_keras_model(target="BMI", mri_space="mni", bg_noise=True)
    # train_and_test_keras_model(target="BMI", mri_space="mni", bg_noise=.02)

    # # Train on BMI (regression): with augmentation
    # ...

    # # Train on BMI (binary classification): normal vs. obese
    # train_and_test_keras_model(target="BMI", mri_space="fs", class_task=True)
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True)
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True,
    #                            norm=(-1, 1))  # (-1,1) i-scale
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True,
    #                            bg_noise=True)  # with BG-noise

    # # Train on AGE (regression): with augmentation, translation only
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True, augment_data=True)
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True, augment_data=True,
    #                            transform_types="translation")

    # # Train on BMI (binary classification): correct for age
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True, correct_for="age")
    # train_and_test_keras_model(target="BMI", mri_space="mni", class_task=True, correct_for="age",
    #                            bg_noise=True)  # with BG-noise

    # %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # # Copy heatmaps to analyse
    # prep_copy_heatmap_data(list_of_sics=['LI02818617', 'LI00033113', 'LI00611396', 'LI01172878',
    #                                      'LI02820234',  'LI01096077',  'LI02792030', 'LI04856096',
    #                                      'LI00603356'],
    #                        model_name="2020-10-13_17-22_AGE_GrandREG_ens5",
    #                        only_aggregated_heatmaps=False, original_mri=False)

    end()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
