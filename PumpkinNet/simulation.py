"""
Main script running simulation.
"""

# %% Import

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # due to BigSur issue with "PyQt5" / "MacOSX" backend

import keras

from utils import root_path, cprint, save_obj, load_obj
from PumpkinNet.train_kerasMRInet import crop_model_name
from PumpkinNet.pumpkinnet import create_simulation_model
from PumpkinNet.simulation_data import split_simulation_data, get_pumpkin_set
from LRP.LRP import apply_colormap  # , analyze_model
from LRP.apply_heatmap import create_cmap, gregoire_black_firered

# %% Set Paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

p2results = os.path.join(root_path, "Results")

# %% Run simulation << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def train_simulation_model(pumpkin_set, leaky_relu=False, epochs=80, batch_size=4):
    # Prep data for model
    xdata, ydata = pumpkin_set.data2numpy(for_keras=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_simulation_data(xdata=xdata, ydata=ydata,
                                                                           only_test=False)

    # Create model
    _model_name = f"PumpkinNet_{'leaky' if leaky_relu else ''}ReLU_{pumpkin_set.name.split('_')[-1]}"
    model = create_simulation_model(name=_model_name,
                                    target_bias=np.mean(ydata),
                                    leaky_relu=leaky_relu)

    # Create folders
    if not os.path.exists(os.path.join(p2results, model.name)):
        os.mkdir(os.path.join(p2results, model.name))

    # # Save model progress (callbacks)
    # See also: https://www.tensorflow.org/tutorials/keras/save_and_load
    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=f"{p2results}{model.name}" + "_{epoch}.h5",
        save_best_only=True,
        save_weights_only=False,
        period=10,
        monitor="val_loss",
        verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"{p2results}{model.name}/")]
    # , keras.callbacks.EarlyStopping()]

    # # Train the model
    cprint('Fit model on training data ...', 'b')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))

    # Save final model (weights+architecture)
    model.save(filepath=f"{p2results}{model.name}_final.h5")  # HDF5 file

    # Report training metrics
    # print('\nhistory dict:', history.history)
    np.save(file=f"{p2results}{model.name}_history", arr=history.history)

    # # Evaluate the model on the test data
    cprint(f'\nEvaluate {model.name} on test data ...', 'b')
    performs = model.evaluate(x_test, y_test, batch_size=1)  # , verbose=2)
    cprint(f'test loss, test performance: {performs}', 'y')

    return model.name


def plot_simulation_heatmaps(model_name, n_subjects=20, subset="test",
                             analyzer_type="lrp.sequential_preset_a", pointers=True,
                             cbar=False, true_scale=False):
    # Get model
    _model = keras.models.load_model(os.path.join(p2results, model_name + "_final.h5"))

    # Get relevance maps
    rel_obj = create_relevance_dict(model_name=model_name, subset=subset, analyzer_type=analyzer_type)

    # Prep data
    pdata = get_pumpkin_set(n_samples=2000, uniform="non-uni" not in model_name)
    _x, _y = pdata.data2numpy(for_keras=True)
    if subset == "test":
        xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        didx = split_simulation_data(xdata=_x, ydata=_y, return_idx=True, only_test=True)
    else:
        # xdata, ydata = ...
        raise NotImplementedError("Not implemented yet for other subsets than 'test'")

    for sub in range(n_subjects):
        # sub = 0
        img = xdata[sub].copy()
        img = img[np.newaxis, ...]
        a = rel_obj[sub]
        # plt.imshow(a)

        col_a = apply_colormap(R=a, inputimage=img.squeeze(), cmapname='black-firered',
                               cintensifier=5., gamma=.2, true_scale=true_scale)
        sub_y = ydata[sub]
        sub_yt = _model.predict(img).item()

        fig = plt.figure(num=f"S{sub}, age={int(sub_y)}, pred={sub_yt:.2f}")
        ax = fig.add_subplot(1, 1, 1)
        aximg = plt.imshow(col_a[0], cmap=create_cmap(gregoire_black_firered))

        if cbar:
            cbar_range = (-1, 1) if not true_scale else (-col_a[2], col_a[2])
            caxbar = fig.colorbar(aximg, ax=ax, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
            caxbar.set_ticks(np.linspace(0, 1, 7), True)
            caxbar.ax.set_yticklabels(labels=[f"{tick:.2g}" for tick in np.linspace(
                cbar_range[0], cbar_range[1], len(caxbar.get_ticks()))])

        plt.tight_layout()

        for fm in ["png", "pdf"]:
            plt.savefig(os.path.join(p2results, _model.name,
                                     f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}.{fm}"))

        if pointers:
            phead = pdata.data[didx[0] + sub]
            # phead.exhibition()
            # cntr = np.array(col_a[0].shape[:-1]) // 2  # center of image

            # Mark atrophies
            for coord in phead.atrophy_coords:
                plt.plot(coord[1], coord[0], "s", color="#D3F5D4",  # "lightgreen"
                         ms=2, alpha=.9)  # ms=4: full-pixel

            # Arrows to lesions
            for coord in phead.lesion_coords:
                # Shadow
                plt.annotate(s='', xy=coord[::-1],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             xytext=np.array(coord[::-1]) + [-4.6, 5.4],
                             arrowprops=dict(arrowstyle='simple', color="black",
                                             alpha=.5))

                # Arrow
                plt.annotate(s='', xy=coord[::-1],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             xytext=np.array(coord[::-1]) + [-5, 5],
                             arrowprops=dict(arrowstyle='simple', color="#E3E7E3",  # "lightgreen"
                                             alpha=.9))

            plt.tight_layout()

            for fm in ["png", "pdf"]:
                plt.savefig(os.path.join(p2results, _model.name,
                                         f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}_pointer.{fm}"))
        plt.close()


def create_relevance_dict(model_name, subset="test", analyzer_type="lrp.sequential_preset_a", save=True):
    try:
        rel_obj = load_obj(name=model_name + f"_relevance-maps_{subset}-set", folder=p2results)
    except FileNotFoundError:

        import innvestigate

        _model = keras.models.load_model(os.path.join(p2results, model_name + "_final.h5"))

        _x, _y = get_pumpkin_set(n_samples=2000,
                                 uniform="non-uni" not in model_name).data2numpy(for_keras=True)
        if subset == "test":
            xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        else:
            # xdata, ydata = ...
            raise NotImplementedError("Implement for other subsets if required!")

        analyzer = innvestigate.create_analyzer(analyzer_type, _model, disable_model_checks=True,
                                                neuron_selection_mode="max_activation")

        rel_obj = analyzer.analyze(xdata, neuron_selection=None).squeeze()
        # can do multiple samples in e.g. xdata.shape (200, 98, 98, 1)

        # rel_dict = {}
        # start = datetime.now()
        # for sub in range(len(ydata)):
        #     img = xdata[sub].copy()
        #     img = img[np.newaxis, ...]
        #     # Generate relevance map
        #     a = analyze_model(mri=img, analyzer_type=analyzer_type, model_=_model, norm=False)
        #     # Fill dict
        #     rel_dict.update({sub: a})
        #     loop_timer(start_time=start, loop_length=len(ydata), loop_idx=sub,
        #                loop_name="Generate Heatmaps")

        if save:
            save_obj(obj=rel_obj, name=model_name + f"_relevance-maps_{subset}-set", folder=p2results)

    return rel_obj


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Testing

if __name__ == "__main__":

    import seaborn as sns

    # # Create heatmaps for all models on testset
    for fn in []:  # os.listdir(p2results):  # TODO TEMP SWITCH OFF
        # find(fname="final.h5", folder=p2results, typ="file", exclusive=False, fullname=False,
        #      abs_path=True, verbose=False)
        if "final.h5" in fn:
            model_name = crop_model_name(model_name=fn)
            cprint(f"\nCreate heatmaps for {model_name}\n", col="p", fm="bo")
            rel_obj = create_relevance_dict(model_name=model_name, subset="test", save=True)

            # Plot heatmaps for N random tori
            plot_simulation_heatmaps(model_name=model_name, n_subjects=20, subset="test", pointers=True,
                                     true_scale=False)

            # Check sum relevance depending on model prdiction
            model = keras.models.load_model(os.path.join(p2results, fn))

            pdata = get_pumpkin_set(n_samples=2000, uniform="non-uni" not in model_name)
            x, y = pdata.data2numpy(for_keras=True)
            xtest, ytest = split_simulation_data(xdata=x, ydata=y, only_test=True)

            pred = model.predict(xtest)
            perf = np.mean(np.abs(pred - ytest[..., np.newaxis]))  # MAE
            print(f"{model_name} with MAE of {perf:.2f}")

            # Compute sum relevance
            sumR = [np.sum(rel_obj[sub]) for sub in range(len(ytest))]

            # Plot Sum.R as function of prediction
            cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
            fig, ax = plt.subplots()
            plt.title(model_name)
            ax.scatter(pred, sumR, c=np.sign(sumR), cmap=cmap)  # cmap="bwr"
            plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], alpha=.5, linestyles="dashed")
            plt.vlines(x=y.mean(),  # target bias
                       ymin=plt.ylim()[0], ymax=plt.ylim()[1], label=f"target bias = {y.mean():.1f}",
                       colors="pink",
                       alpha=.5, linestyles="dotted")
            plt.xlabel("prediction")
            plt.ylabel("sum relevance")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(p2results, model_name, "Sum_relevance_over_Prediction.png"))
            plt.savefig(os.path.join(p2results, model_name, "Sum_relevance_over_Prediction.pdf"))
            # plt.show()
            plt.close()

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  END
