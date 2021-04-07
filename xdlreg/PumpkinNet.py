"""
Create deep convolutional neural network.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)  # primarily for tensorflow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xdlreg import utils
from xdlreg.utils import cprint, browse_files, function_timed
from xdlreg.SimulationData import split_simulation_data, get_pumpkin_set


# %% Set global params << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

def p2models():
    return os.path.join(utils.root_path, "Results/model")


# %% ConvNet << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

# TODO ADD CITATION / DOI
def create_simulation_model(name: str = "PumpkinNet", output_bias: float = None,
                            input_shape: tuple = (98, 98), batch_norm: bool = False,
                            class_task: bool = False):
    """

    This is a 2D adaptation of the model reported in Hofmann et al. (2021)

    :param name: name of model
    :param output_bias: for regression tasks setting the output bias to target mean can ease the training
                        process. Also, for relevance analysis, this can do improved imterpretability of
                        sign of relevance values.
    :param input_shape: shape of single image sample
    :param batch_norm: whether to include batch norm layers
    :param class_task: whether model will be applied on classification problem
    :return: build model
    """

    name += "BiCL" if class_task else ""
    kmodel = keras.Sequential(name=name)  # OR: Sequential([keras.layer.Conv2d(....), layer...])

    actfct = "relu"

    # # 2D-convolutional neural network (CNN)
    # Can be written more compact, but this comes with a bit more flexibility w.r.t. adaptations.
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization(input_shape=input_shape + (1,)))
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct))
    else:
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct, input_shape=input_shape + (1,)))
        # auto-add batch:None, last: channels
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME", activation=actfct))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation=actfct))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation=actfct))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    # 2D-Conv (1x1x1)
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="SAME", activation=actfct))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())

    # FC
    kmodel.add(keras.layers.Flatten())
    kmodel.add(keras.layers.Dropout(rate=.5))
    kmodel.add(keras.layers.Dense(units=64, activation=actfct))

    # Output
    if not class_task:
        kmodel.add(keras.layers.Dense(
            units=1, activation='linear', use_bias=True,
            bias_initializer="zeros" if output_bias is None else keras.initializers.Constant(
                value=output_bias)))

    else:
        kmodel.add(keras.layers.Dense(units=2, use_bias=False,
                                      activation='softmax'))  # in binary case. also: 'sigmoid'

    # Compile
    kmodel.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse",
                   metrics=["accuracy"] if class_task else ["mae"])

    # Summary
    kmodel.summary()

    return kmodel


def is_binary_classification(model_name: str) -> bool:
    """
    Report whether model was used for a binary classification task.
    :param model_name: name of model
    :return: is model for binary classification [bool]
    """
    return "BiCL" in model_name


# %% Plotting << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><

def plot_training_process(_model, save: bool = True):
    """
    Plot the training process of given model.
    :param _model: model
    :param save: whether to save plot
    """

    history_fn = os.path.join(p2models(), _model.name, f"{_model.name}_history.npy")
    if os.path.isfile(history_fn) and not os.path.isfile(history_fn.replace(".npy", ".png")):
        _binary_cls = is_binary_classification(_model.name)
        model_history = np.load(history_fn, allow_pickle=True).item()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
        fig.suptitle(f"{_model.name}")
        ax1.plot(model_history['acc' if _binary_cls else 'mean_absolute_error'])
        ax1.plot(model_history['val_acc' if _binary_cls else 'val_mean_absolute_error'])
        ax1.set_title('Performance')
        ax1.set_ylabel('accuracy' if _binary_cls else 'mean absolute error')
        ax1.set_xlabel('training epoch')
        ax1.legend(['train', 'val'], loc='upper left')
        # summarize history for loss
        ax2.plot(model_history['loss'])
        ax2.plot(model_history['val_loss'])
        ax2.set_title('Loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('training epoch')
        ax2.legend(['train', 'val'], loc='upper left')
        plt.tight_layout()
        if save:
            fig.savefig(history_fn.replace(".npy", ".png"))
            plt.close()
            return history_fn.replace(".npy", ".png")
        else:
            plt.show()
    elif os.path.isfile(history_fn) and os.path.isfile(history_fn.replace(".npy", ".png")):
        return history_fn.replace(".npy", ".png")
    else:
        cprint(f"No history file found for given model '{_model.name}'!", 'r')


def plot_prediction(_model, xdata, ydata, save: bool = True):
    """
    Plot predictions of model for given data.
    :param _model: model
    :param xdata: input data
    :param ydata: target variable
    :param save: whether to save plot
    """
    _target = "age"

    # # Calculate model-performance
    if is_binary_classification(_model.name):  # TODO consider multi-class case

        from sklearn.metrics import classification_report, confusion_matrix
        import pandas as pd

        # # Get model predictions
        _classes = ["young", "old"]
        m_pred = _model.predict(x=xdata)  # takes a while
        correct = (np.argmax(m_pred, axis=1) == np.argmax(ydata, axis=1)) * 1
        accuracy = sum(correct) / len(correct)

        # # Classification report
        y_true = np.argmax(ydata, axis=1)
        y_pred = np.argmax(m_pred, axis=1)
        cprint(classification_report(y_true, y_pred, target_names=_classes), 'y')
        # Precision: tp/(tp+fp) | Recall (=sensitivity): tp/(tp+fn) | Specificity: tn/(tn+fp))

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
        _ax.set_ylim([0, 2])  # labelling is off otherwise, OR downgrade to matplotlib==3.1.0

        plot_paths = [os.path.join(p2models(), _model.name,
                                   f"{_model.name}_confusion-matrix_(acc={accuracy:.2f}).png")]

    else:
        # # Get model predictions
        m_pred = _model.predict(x=xdata) # takes a while
        mae = np.absolute(ydata - m_pred[:, 0]).mean()

        # # Jointplot
        plot_paths = []
        plot_paths.append(os.path.join(p2models(), _model.name,
                                       f"{_model.name}_predictions_MAE={mae:.2f}.png"))

        sns.jointplot(x=m_pred[:, 0], y=ydata, kind="reg", height=10,
                      marginal_kws=dict(bins=int(round((ydata.max() - ydata.min()) / 3))),
                      xlim=(np.min(m_pred) - 10, np.max(m_pred) + 10),
                      ylim=(np.min(ydata) - 10, np.max(ydata) + 10)).plot_joint(
            sns.kdeplot, zorder=0, n_levels=6).set_axis_labels("Predictions",
                                                               f"True-{_target.upper()}")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f"Predicted {_target.lower()}", fontsize=20)
        plt.ylabel(f"Chronological {_target.lower()}", fontsize=20)

        plt.tight_layout()
        if save:
            plt.savefig(plot_paths[0])  # Save plot
            plt.close()
        else:
            plt.show()

        # # Residuals
        plot_paths.append(os.path.join(p2models(), _model.name,
                                       f"{_model.name}_residuals_MAE={mae:.2f}.png"))

        _fig = plt.figure(f"{_target.title()} Prediction Model Residuals MAE={mae:.2f}",
                          figsize=(10, 8))
        _ax2 = _fig.add_subplot(1, 1, 1)
        _ax2.set_title(f"Residuals w.r.t. {_target.upper()} (MAE={mae:.2f})")
        rg_ = sns.regplot(x=ydata, y=m_pred[:, 0] - ydata, order=3)
        plt.hlines(y=0, xmin=min(ydata) - 3, xmax=max(ydata) + 3, linestyles="dashed", alpha=.5)
        plt.vlines(x=np.median(ydata), ymin=min(m_pred[:, 0] - ydata) - 2,
                   ymax=max(m_pred[:, 0] - ydata) + 2,
                   linestyles="dotted", color="red", alpha=.5, label="median test set")
        _ax2.legend()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f"True-{_target.upper()}", fontsize=20)
        plt.ylabel(f"Prediction Error (pred-t)", fontsize=20)

        plt.tight_layout()

    if save:
        _fig.savefig(plot_paths[-1])
        plt.close()
    else:
        plt.show()

    return plot_paths


# %% Train model << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def model_for_data_exists(dataset_name: str, epochs: int):
    """
    Check whether there is a model which was trained on the given dataset already, with the same number of
    given epochs.
    :param dataset_name: name of dataset
    :param epochs: number of epochs
    :return: name of model if model found else None
    """
    model_name = f"PumpkinNet_{dataset_name.split('N-')[1]}"
    model = load_trained_model(model_name=model_name, verbose=False)

    if model is not None:
        history_fn = os.path.join(p2models(), model.name, f"{model.name}_history.npy")
        model_epochs = len(np.load(history_fn, allow_pickle=True).item()["loss"])

        if epochs == model_epochs:
            cprint(f"Found model '{model.name}' which was trained on given dataset with the "
                   f"requested number of epochs (N={epochs}).", 'b')
            return model.name
        else:
            cprint(f"Found model '{model.name}' which was trained on given dataset, but not with the "
                   f"requested number of epochs ({model_epochs}!={epochs}).", 'y')


@function_timed
def train_simulation_model(pumpkin_set, epochs: int = 80, batch_size: int = 4) -> str:
    """
    Train model on simulated dataset.

    :param pumpkin_set: dataset containing simulated images (ageing 'pumpkins')
    :param epochs: number of training epochs
    :param batch_size: size of batch
    :return: name of model
    """

    already_trained = model_for_data_exists(dataset_name=pumpkin_set.name, epochs=epochs)
    if isinstance(already_trained, str):  # ... is not None
        return already_trained

    # Prep data for model
    xdata, ydata = pumpkin_set.data2numpy(for_keras=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_simulation_data(xdata=xdata, ydata=ydata,
                                                                           only_test=False)

    # Create model
    _model_name = f"PumpkinNet_{pumpkin_set.name.split('N-')[-1]}"
    model = create_simulation_model(name=_model_name, output_bias=np.mean(ydata))

    # Create folders
    if not os.path.exists(os.path.join(p2models(), model.name)):
        os.makedirs(os.path.join(p2models(), model.name))

    # # Save model progress (callbacks)
    # See also: https://www.tensorflow.org/tutorials/keras/save_and_load
    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=f"{p2models()}/{model.name}/{model.name}" + "_{epoch}.h5",
        save_best_only=True,
        save_weights_only=False,
        period=10,
        monitor="val_loss",
        verbose=0),
        keras.callbacks.TensorBoard(log_dir=f"{p2models()}/{model.name}/")]
    # , keras.callbacks.EarlyStopping()]

    # # Train the model
    cprint('Fit model on training data ...', 'b')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))

    # Save final model (weights+architecture)
    model.save(filepath=f"{p2models()}/{model.name}/{model.name}_final.h5")  # HDF5 file

    # Report training metrics
    # print('\nhistory dict:', history.history)
    np.save(file=f"{p2models()}/{model.name}/{model.name}_history", arr=history.history)

    # # Evaluate the model on the test data
    cprint(f'\nEvaluate {model.name} on test data ...', 'b')
    performs = model.evaluate(x_test, y_test, batch_size=1)  # , verbose=2)
    cprint(f'test loss, test performance: {performs}', 'y')

    # Plots for training process & predictions on test set
    plot_training_process(_model=model)
    plot_prediction(_model=model, xdata=x_test, ydata=y_test)

    return model.name


def crop_model_name(model_name: str) -> str:
    """
    Crop model name, removing file-type etc.
    :param model_name: name of model
    :return: cropped name of model
    """
    if model_name.endswith(".h5"):
        model_name = model_name[0:-len(".h5")]

    if model_name.endswith("_final"):
        model_name = model_name[0:-len("_final")]

    return model_name


def load_trained_model(model_name: str = None, verbose: bool = True):
    """
    Load a (trained) model.
    :param model_name: name of model, can be None: then browsing option will be opened.
    :param verbose: True: report if model wasn't found
    :return: trained model
    """

    if model_name:
        if os.path.isdir(os.path.join(p2models(), crop_model_name(model_name))):
            # Load & return ensemble model
            # Load single model
            # e.g., 2020-01-13_14-05_MRIkerasNetAGE_MNI_BinClassi_final.h5
            if ".h5" not in model_name:
                if "_final" not in model_name:
                    model_name += "_final.h5"
            return keras.models.load_model(os.path.join(p2models(),
                                                        crop_model_name(model_name),
                                                        model_name))
        else:
            if verbose:
                cprint(f"No model directory found for given model '{model_name}'", col='r')
            return None

    else:
        path2model = browse_files(p2models(), "H5")
        _model_name = path2model.split("/")[-1]

        return load_trained_model(model_name=_model_name)


def get_model_data(model_name: str, for_keras: bool = True):
    """
    Get data for given model. That is, the data the model was trained on.
    :param model_name: name of model
    :param for_keras: True: prepare data, such that they can directly used for the given model
    :return: dataset for model training and evaluation
    """
    n_samples = int(model_name.split("_")[-2])
    uniform = "non-uni" not in model_name
    age_bias = None if uniform else float(model_name.split("uniform")[-1])
    if for_keras:
        return get_pumpkin_set(n_samples=n_samples, uniform=uniform,
                                 age_bias=age_bias).data2numpy(for_keras=True)
    else:
        return get_pumpkin_set(n_samples=n_samples, uniform=uniform,
                               age_bias=age_bias)

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
