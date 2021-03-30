# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply LRP on (trained) MRInet

Sebastian L.:
    Use LrpPreset*-Analyzer for decomposition (current best practice).
    That is, alpha-beta for conv-layer, epsilon for dense-layer,
    optional flat-LRP for invariance w.r.t. scaling in input/lowest layer

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2019, 2020
"""
# %% Import
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from LRP.LRP import (apply_analyzer_and_plot_heatmap, analyze_model, plot_heatmap)

from utils import cprint, p2results
from PumpkinNet.train_kerasMRInet import (load_trained_model, get_model_data,
                                          get_sub_ensemble_predictions,
                                          load_datasplit, get_target, get_classes,
                                          is_binary_classification,
                                          crop_model_name)

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Functions for __main__:


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
