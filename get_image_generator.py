# https://keras.io/preprocessing/image/

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from pandas import read_csv
from meta_data import TARGET_SIZE, BATCH_SIZE, RESCALE

train_df = read_csv("./train.csv")
valid_df = read_csv("./valid.csv")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        shuffle=True,
        x_col="filename",
        y_col="class",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

