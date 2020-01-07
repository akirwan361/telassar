from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .data import DataND
from .world import Position, VelWave
from .plotter import (ImPlotter, get_plot_norm, get_plot_extent,
                      get_background_rms, get_contour_levels)

class SpecLine(DataND):

    _is_spectral = True

    def __init__(self, *args):
        return
