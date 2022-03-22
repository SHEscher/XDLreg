# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execute __main__.py

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021-2022
"""

# %% Import
import os
import argparse
from xdlreg.utils import end
from xdlreg.run_simulation import run_simulation


# %% Run simulation when main is called < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def str2bool(v):
    """Convert bool-like string into bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Parse arguments given via shell.
    :return: valid arguments for PumpkinNet.run_simulation
    """
    parser = argparse.ArgumentParser(description="Arguments relevant for model training & heatmap plots.")
    parser.add_argument('--path', type=str, default=os.getcwd(),
                        help='Number of samples in simulated dataset.')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of samples in simulated dataset.')
    parser.add_argument('--uniform', type=str2bool, default=True,
                        help='Whether simulated dataset is to be uniformly distributed.')
    parser.add_argument('--target_bias', type=float, default=None,
                        help='For non-uniformly distributed datasets a distribution bias can be set, '
                             'i.e., the majority of samples will be drawn from this bias.')
    parser.add_argument('--growth_mode', type=str, default="human",
                        help='Defines how pumpkins grow with age. Only if set to "human"'
                             'pumpkins do suffer lesions and atrophies with age.')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training iterations.')
    parser.add_argument('--plot_n_heatmaps', type=int, default=20,
                        help='Number of heatmaps which are to be plotted.')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def main():
    """
    Run the whole pipeline:
    1) creating a simulated dataset,
    2) building and training a model (ConvNet) on that dataset, and
    3) analyzing the continuous model predictions (regression) with the LRP algorithm.
    """

    print(f"Execute {__file__}:main\n")
    run_simulation(**vars(get_args()))
    end()


if __name__ == "__main__":
    main()

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
