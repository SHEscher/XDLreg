# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execute __main__.py

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

from utils import p2results, cprint, find
from PumpkinNet.pumpkinnet import train_simulation_model, crop_model_name
from PumpkinNet.simulation_data import get_pumpkin_set
from LRP.create_heatmaps import create_relevance_dict, plot_simulation_heatmaps

# TODO temporary:
def main():
    print(f"\nExecute __main__.py in {__file__}\n")
    train_simulation_model(pumpkin_set=get_pumpkin_set(n_samples=2000), epochs=80)

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

# TODO Install locally [remove later]
# python3 setup.py sdist bdist_wheel  # build
# pip install -e ./XDLreg  # install as module


if __name__ == "__main__":
    main()
