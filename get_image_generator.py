# https://keras.io/preprocessing/image/

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import backend as K

from pandas import read_csv
from meta_data import IMAGE_TARGET_SIZE, RESCALE, TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME, DATA_PATH, CSV_COLS, TRAIN_PATH, TEST_PATH
from meta_parameters import BATCH_SIZE

train_df = read_csv(TRAIN_FILENAME)
valid_df = read_csv(VALID_FILENAME)
test_df = read_csv(TEST_FILENAME)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=RESCALE,
        data_format=K.image_data_format())

valid_gatagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=RESCALE,
        data_format=K.image_data_format())

test_gatagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=RESCALE,
        data_format=K.image_data_format())

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_PATH + TRAIN_PATH,
        shuffle=True,
        x_col=CSV_COLS[0],
        y_col=CSV_COLS[1],
        target_size=IMAGE_TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

validation_generator = valid_gatagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=DATA_PATH + TRAIN_PATH,
        x_col=CSV_COLS[0],
        y_col=CSV_COLS[1],
        target_size=IMAGE_TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

test_generator = valid_gatagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH + TEST_PATH,
        x_col=CSV_COLS[0],
        y_col=CSV_COLS[1],
        target_size=IMAGE_TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

