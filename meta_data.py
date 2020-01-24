import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
# https://keras.io/preprocessing/image/
import numpy as np
import pathlib

BATCH_SIZE = 32
IMG_HEIGHT = 294
IMG_WIDTH = 776
TARGET_SIZE = (IMG_HEIGHT,IMG_WIDTH)
RESCALE=1./255


image_count = 0
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)