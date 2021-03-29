# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execute __main__.py

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

from PumpkinNet.pumpkinnet import train_simulation_model
from PumpkinNet.simulation_data import get_pumpkin_set

# TODO temporary:
def main():
    print(f"\nExecute __main__.py in {__file__}\n")
    train_simulation_model(pumpkin_set=get_pumpkin_set(n_samples=100), epochs=50)

# TODO Install locally [remove later]
# python3 setup.py sdist bdist_wheel  # build
# pip install -e ./XDLreg  # install as module


if __name__ == "__main__":
    main()
