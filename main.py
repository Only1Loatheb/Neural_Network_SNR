import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from get_image_generator import train_generator, validation_generator
from neural_network_model import model

from meta_data import EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS

model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)