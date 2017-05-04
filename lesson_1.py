# Imports #####################################################################
import json
import os

import numpy as np
import importlib as il

from matplotlib import pyplot as plt


# Display options #############################################################
np.set_printoptions(precision=4, linewidth=100)


# Code ########################################################################
# Kaggle competition https://www.kaggle.com/c/dogs-vs-cats
path = "data/dogscats/"

import utils; il.reload(utils)
from utils import plots


# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size=64