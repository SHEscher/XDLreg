# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execute __main__.py

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# TODO Install locally [remove later]
# python3 setup.py sdist bdist_wheel  # build
# pip install -e ./XDLreg  # install as module

# %% Import
import argparse
from utils import str2bool, end
from PumpkinNet.run_simulation import run_simulation


# %% Run simulation when main is called << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def get_args():

    parser = argparse.ArgumentParser(description="Arguments relevant for model training & heatmap plots.")
    # TODO consider adding project path
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of samples in simulated dataset.')
    parser.add_argument('--uniform', type=str2bool, default=True,
                        help='Whether simulated dataset is to be uniformly distributed.')
    parser.add_argument('--target_bias', type=float, default=None,
                        help='For non-uniformly distributed datasets a distribution bias can be set, '
                             'i.e., the majority of samples will be drawn from this bias.')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training iterations.')
    parser.add_argument('--plot_n_heatmaps', type=int, default=20,
                        help='Number of heatmaps which are to be plotted.')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def main():

    print(f"\nExecute __main__.py in {__file__}\n")  # TODO temporary
    print(get_args())
    run_simulation(**vars(get_args()))
    end()


if __name__ == "__main__":
    main()

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
