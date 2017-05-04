# Imports #####################################################################
import json
import os
import utils

import vgg16

import numpy as np
import importlib as il

from matplotlib import pyplot as plt
from utils import plots

from vgg16 import Vgg16


# Display options #############################################################
np.set_printoptions(precision=4, linewidth=100)


# Code ########################################################################
# Kaggle competition https://www.kaggle.com/c/dogs-vs-cats
path = "data/dogscats/"

# Set the batch size
batch_size = 64


# Training ####################################################################
# Create an instance of the Vgg model
vgg = Vgg16()

# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
