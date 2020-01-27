# https://keras.io/examples/image_ocr/
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import backend as K
from meta_data import IMG_HEIGHT, IMG_WIDTH, CONV_FILTERS, POOL_SIZE

if K.image_data_format() == 'channels_first':
    input_shape = (3,IMG_HEIGHT,IMG_WIDTH)
else:
    input_shape = (IMG_HEIGHT,IMG_WIDTH, 3)



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
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros')
  )

model.add(
  tf.keras.layers.MaxPooling2D(name='max1',
    pool_size=(POOL_SIZE, POOL_SIZE))
  )

model.add(
  tf.keras.layers.Conv2D(
    CONV_FILTERS, # liczba filtrów w warstwie 
    (5,5), # kernel size - rozmiar filtru
    name='conv2',
    strides=(1, 1), # krok przesówania filtru
    padding='valid', # no padding => crop
    dilation_rate=(1, 1), # odstępy między próbkami w filtrze
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros')
  )

model.add(
  tf.keras.layers.MaxPooling2D(name='max2',
    pool_size=(POOL_SIZE, POOL_SIZE))
  )
model.add(
  tf.keras.layers.Flatten(name='flat1')
  )

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])