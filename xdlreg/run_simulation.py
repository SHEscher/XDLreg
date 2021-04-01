"""
Run simulation on LRP for regression.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import
import os
from xdlreg.utils import cprint, open_folder
from xdlreg.PumpkinNet import train_simulation_model
from xdlreg.SimulationData import get_pumpkin_set
from xdlreg.LRP.create_heatmaps import create_relevance_dict, plot_simulation_heatmaps, p2relevance


# %% Run simulation << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def run_simulation(n_samples=2000, uniform=True, target_bias=None, epochs=80, plot_n_heatmaps=20):

    cprint(f"Run simulation on {n_samples} samples:\n", col="b", fm="bo")

    # Create data and train model
    cprint(f"Train PumpkinNet:\n", col="b", fm="bo")
    model_name = train_simulation_model(pumpkin_set=get_pumpkin_set(n_samples=n_samples, uniform=uniform,
                                                                    age_bias=target_bias), epochs=epochs)

    # Create relevance maps via LRP
    cprint(f"\nCreate relevance maps (LRP) for {model_name}:\n", col="p", fm="bo")
    rel_obj = create_relevance_dict(model_name=model_name, subset="test", save=True)

    # Plot some of the relevance maps (heatmaps)
    cprint(f"\nPlot heatmaps for {plot_n_heatmaps} random tori ('pumpkins'):\n", col="p", fm="bo")
    plot_simulation_heatmaps(model_name=model_name, n_subjects=plot_n_heatmaps, subset="test",
                             pointers=True, true_scale=False)

    # Open folder with plots
    open_folder(os.path.join(p2relevance, model_name, "plots"))

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
