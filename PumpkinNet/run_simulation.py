"""
Run simulation on LRP for regression.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from utils import p2results, cprint, find
from PumpkinNet.pumpkinnet import crop_model_name
from PumpkinNet.simulation_data import get_pumpkin_set, split_simulation_data
from LRP.create_heatmaps import create_relevance_dict, plot_simulation_heatmaps

# %% >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Testing

if __name__ == "__main__":

    # # Create heatmaps for all models on testset

    fn = find(fname="final.h5", folder=p2results, typ="file", exclusive=False, fullname=False,
              abs_path=True, verbose=False)

    for fn in [fn.split("/model/")[-1]]:  # os.listdir(p2results):  # TODO TEMP toggled
        if "final.h5" in fn:
            model_name = crop_model_name(model_name=fn)
            cprint(f"\nCreate heatmaps for {model_name}\n", col="p", fm="bo")
            rel_obj = create_relevance_dict(model_name=model_name, subset="test", save=True)

            # Plot heatmaps for N random tori
            plot_simulation_heatmaps(model_name=model_name, n_subjects=20, subset="test", pointers=True,
                                     true_scale=False)

            break

            # Check sum relevance depending on model prdiction
            model = keras.models.load_model(os.path.join(p2results, fn))

            pdata = get_pumpkin_set(n_samples=2000, uniform="non-uni" not in model_name)
            x, y = pdata.data2numpy(for_keras=True)
            xtest, ytest = split_simulation_data(xdata=x, ydata=y, only_test=True)

            pred = model.predict(xtest)
            perf = np.mean(np.abs(pred-ytest[..., np.newaxis]))  # MAE
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
