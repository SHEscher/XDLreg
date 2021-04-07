__version__ = "1.0.0"
__doc__ = """
a package to study explainable deep learning (XDL) for regression tasks on simulated image data.
"""

# Import
from .PumpkinNet import p2models
from .run_simulation import run_simulation
from .SimulationData import p2data, PumpkinSet, get_pumpkin_set
from .LRP.create_heatmaps import p2relevance, create_relevance_dict, plot_simulation_heatmaps
