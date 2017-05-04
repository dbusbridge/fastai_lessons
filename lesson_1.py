# Imports #####################################################################
import json
import os

import numpy as np
import importlib as il

from matplotlib import pyplot as plt


# Display options #############################################################
np.set_printoptions(precision=4, linewidth=100)


# Code ########################################################################

# Cats vs dogs
path = "data/dogscats/"

import utils; il.reload(utils)
from utils import plots
