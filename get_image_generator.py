import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from pandas import read_csv
train_df = read_csv("./train.csv")
valid_df = read_csv("./valid.csv")

print(valid_df)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        huffle=True,
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=800)