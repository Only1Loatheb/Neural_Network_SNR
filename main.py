import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from datetime import datetime
from tensorflow import keras

logdir = "./folder" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


from get_image_generator import train_generator, validation_generator, test_generator
from model import model

from meta_parameters import EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS, BATCH_SIZE

model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    use_multiprocessing=True,
    callbacks=[tensorboard_callback],
)


model.save_weights('my_model_weights.h5')
print("evaluate", model.metrics_names)
print(model.evaluate_generator(
    test_generator,
    use_multiprocessing=True,
    )
)


model.count_params()
model.summary()
#TODO numbe of parameters
# https://stackoverflow.com/questions/35792278/how-to-find-number-of-parameters-of-a-keras-model