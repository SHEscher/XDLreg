"""
Create LRP heatmaps for predictino model.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import
import os
import numpy as np
import keras
import matplotlib.pyplot as plt

from utils import p2results, save_obj, load_obj
from PumpkinNet.simulation_data import get_pumpkin_set, split_simulation_data
from LRP.LRP import apply_colormap, create_cmap, gregoire_black_firered


# %% Create heatmaps & plots >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

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
            phead = pdata.data[didx[0]+sub]
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