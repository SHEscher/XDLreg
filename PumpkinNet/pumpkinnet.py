"""
Create deep convolutional neural network.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import

import os
import keras
import numpy as np
from utils import cprint, p2results
from PumpkinNet.simulation_data import split_simulation_data


# %% ConvNet ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><

def create_simulation_model(name="PumpkinNet", target_bias=None, input_shape=(98, 98), batch_norm=False,
                            class_task=False):
    # TODO ADD CITATION / DOI
    """
    This is a 2D adaptation of the model reported in Hofmann et al. (2021)
    """

    if target_bias is not None:
        cprint(f"\nGiven target bias is {target_bias:.3f}\n", "y")

    actfct = "relu"

    kmodel = keras.Sequential(name=name)  # OR: Sequential([keras.layer.Conv2d(....), layer...])

    # 3D-Conv
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

    # 3D-Conv (1x1x1)
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
            units=1, activation='linear',
            # add target bias == 57.317 (for age), or others
            use_bias=True,
            bias_initializer="zeros" if target_bias is None else keras.initializers.Constant(
                value=target_bias)))

    else:
        kmodel.add(keras.layers.Dense(units=2,
                                      activation='softmax',  # in binary case. also: 'sigmoid'
                                      use_bias=False))  # default: True

    # Compile
    kmodel.compile(optimizer=keras.optimizers.Adam(5e-4),  # ="adam",
                   loss="mse",
                   metrics=["accuracy"] if class_task else ["mae"])

    # Summary
    kmodel.summary()

    return kmodel


def train_simulation_model(pumpkin_set, epochs=80, batch_size=4):
    # Prep data for model
    xdata, ydata = pumpkin_set.data2numpy(for_keras=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_simulation_data(xdata=xdata, ydata=ydata,
                                                                           only_test=False)

    # Create model
    _model_name = f"PumpkinNet_{pumpkin_set.name.split('N-')[-1]}"
    model = create_simulation_model(name=_model_name, target_bias=np.mean(ydata))

    # Create folders
    if not os.path.exists(os.path.join(p2results, "model", model.name)):
        os.makedirs(os.path.join(p2results, "model", model.name))

    # # Save model progress (callbacks)
    # See also: https://www.tensorflow.org/tutorials/keras/save_and_load
    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=f"{p2results}/model/{model.name}" + "_{epoch}.h5",
        save_best_only=True,
        save_weights_only=False,
        period=10,
        monitor="val_loss",
        verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"{p2results}/model/{model.name}/")]
    # , keras.callbacks.EarlyStopping()]

    # # Train the model
    cprint('Fit model on training data ...', 'b')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))

    # Save final model (weights+architecture)
    model.save(filepath=f"{p2results}/model/{model.name}_final.h5")  # HDF5 file

    # Report training metrics
    # print('\nhistory dict:', history.history)
    np.save(file=f"{p2results}/model/{model.name}_history", arr=history.history)

    # # Evaluate the model on the test data
    cprint(f'\nEvaluate {model.name} on test data ...', 'b')
    performs = model.evaluate(x_test, y_test, batch_size=1)  # , verbose=2)
    cprint(f'test loss, test performance: {performs}', 'y')

    return model.name


def crop_model_name(model_name):
    if model_name.endswith(".h5"):
        model_name = model_name[0:-len(".h5")]

    if model_name.endswith("_final"):
        model_name = model_name[0:-len("_final")]

    return model_name
