import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from get_image_generator import train_generator, validation_generator
from model import model

from meta_data import EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS

model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    use_multiprocessing=True)

# model.save('my_model.h5')

# predictions = model.predict(testX, batch_size=32)

model.save_weights('my_model_weights.h5')


# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=32)

# #Uncomment to see the predicted probabilty for each class in every test image
# # print ("predictions---------------->",predictions)
# #Uncomment to print the predicted labels in each image
# # print("predictions.argmax(axis=1)",predictions.argmax(axis=1))

# # print the performance report of the prediction
# print(classification_report(testY.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=lb.classes_))

# # plot the training loss and accuracy for each epoch
# N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["acc"], label="train_acc")
# plt.plot(N, H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy (simple_multiclass_classifcation)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig("training_performance.png")