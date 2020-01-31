import numpy as np

BATCH_SIZE = 32
EPOCHS = 1
STEPS_PER_EPOCH = np.ceil(800/BATCH_SIZE)
VALIDATION_STEPS = 1

CONV_FILTERS = 8
POOL_SIZE = 2