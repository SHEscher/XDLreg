"""
Create deep convolutional neural network.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import

import keras
from utils import cprint

# %% ConvNet ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><


def create_simulation_model(name="PumpkinNet", target_bias=None, input_shape=(98, 98), class_task=False,
                            leaky_relu=True, batch_norm=False):
    # TODO ADD CITATION / DOI
    """
    This is a 2D adaptation of the model reported in Hofmann et al. (2021)
    """

    if target_bias is not None:
        cprint(f"\nGiven target bias is {target_bias:.3f}\n", "y")

    if leaky_relu and not batch_norm:
        actfct = None
    else:
        actfct = "relu"

    kmodel = keras.Sequential(name=name)  # OR: Sequential([keras.layer.Conv3d(....), layer...])

    # 3D-Conv
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization(input_shape=input_shape + (1,)))
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct))
    else:
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct, input_shape=input_shape + (1,)))
        # auto-add batch:None, last: channels
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))  # lrelu
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    # 3D-Conv (1x1x1)
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())

    # FC
    kmodel.add(keras.layers.Flatten())
    kmodel.add(keras.layers.Dropout(rate=.5))
    kmodel.add(keras.layers.Dense(units=64, activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))

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
