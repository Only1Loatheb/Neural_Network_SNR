# https://keras.io/examples/image_ocr/
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import backend as K
from meta_data import IMG_HEIGHT, IMG_WIDTH
from meta_parameters import CONV_FILTERS, POOL_SIZE
from tensorflow import keras

if K.image_data_format() == 'channels_first':
    input_shape = (3,IMG_HEIGHT,IMG_WIDTH)
else:
    input_shape = (IMG_HEIGHT,IMG_WIDTH, 3)

#TODO ile mam wag

model = tf.keras.models.Sequential()

model.add(
  tf.keras.layers.Conv2D(
    CONV_FILTERS, # liczba filtrów w warstwie 
    (5,5), # kernel size - rozmiar filtru
    name='conv1',
    input_shape=input_shape,
    data_format=K.image_data_format(),
    strides=(1, 1), # krok przesówania filtru
    padding='valid', # no padding => crop
    dilation_rate=(1, 1), # odstępy między próbkami w filtrze
    activation='relu',
    use_bias=False,
    kernel_initializer='glorot_uniform',
    )
  )
for i in range(5):
  model.add(
    tf.keras.layers.MaxPooling2D(name=f'max_{i+2}',
      pool_size=(POOL_SIZE, POOL_SIZE),
      strides=(POOL_SIZE, POOL_SIZE),
      )
    )

  model.add(
    tf.keras.layers.Conv2D(
      CONV_FILTERS, # liczba filtrów w warstwie 
      (5,5), # kernel size - rozmiar filtru
      name=f'conv_{i+2}',
      strides=(1, 1), # krok przesówania filtru
      padding='valid', # no padding => crop
      dilation_rate=(1, 1), # odstępy między próbkami w filtrze
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros')
    )

model.add(
  tf.keras.layers.Flatten(name='flat1')
  )

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2, activation='softmax'))



model.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

              #TODO loss inne 
              #TODO przeuczony
              # summary - jak wyglądała sięc
