import numpy as np

DATA_PATH = "data"
TRAIN_PATH = "/train" #"/moded" #
TEST_PATH = "/test"

TRAIN_FILENAME = "./train.csv"
VALID_FILENAME = "./valid.csv"
TEST_FILENAME = "./test.csv"
CSV_COLS = ("filename","class")

IMG_HEIGHT = 294
IMG_WIDTH = 776
IMAGE_TARGET_SIZE = (IMG_HEIGHT,IMG_WIDTH)
RESCALE=1./255