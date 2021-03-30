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

from utils import end
from PumpkinNet.run_simulation import run_simulation


# %% Run simulation when main is called << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def main():
    print(f"\nExecute __main__.py in {__file__}\n")  # TODO temporary:
    # TODO add shell *args
    run_simulation()
    end()


if __name__ == "__main__":
    main()

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
