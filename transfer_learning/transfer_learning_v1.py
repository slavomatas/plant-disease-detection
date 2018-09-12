import os

import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from keras.initializers import TruncatedNormal
from tqdm import tqdm
from sklearn.datasets import load_files

from keras.applications import VGG16, ResNet50, Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
#from extract_bottleneck_features import *

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/home/slavo/Dev/image-classification/dogImages/train/*/"))]

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

def create_model():
    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    conv_base = Xception(include_top=False, weights='imagenet')
    #conv_base = ResNet50(include_top=False, weights='imagenet')
    conv_base.trainable = False

    """
    for layer in conv_base.layers:
        print("Layer {}".format(layer.name))
    """

    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D(input_shape=conv_base.layers[-1].output_shape[1:]))

    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax',
                    kernel_initializer=init,
                    bias_initializer='zeros'))

    model.summary()

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def train_model():
    base_dir = '/home/slavo/Dev/plant-disease-detection/images/'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')

    batch_size = 20

    # Augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # this is a generator that will read pictures found in subfolders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # this is the augmentation configuration we will use for testing: only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a similar generator, for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # ### Train the Model
    checkpointer = ModelCheckpoint(filepath='saved_models/transfer_learning_v1.hdf5',
                                   verbose=1, save_best_only=True)

    model = create_model()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=500,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[checkpointer],
        verbose=1,
        use_multiprocessing=1,
        workers=5)


def plot_accuracy_loss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def test_model():
    # Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.
    # We print the test accuracy below.

    # ### Load the Model with the Best Validation Loss
    # model.load_weights('saved_models/weights.best.rmsprop.hdf5')

    # get index of predicted dog breed for each image in test set
    Xception(weights='imagenet', include_top=False)
    model_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in testing_features]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # ### Predict Dog Breed with the Model
    def ResNet50_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]

    ### TODO: Test the performance of the leaf_detector function

    dog_files_short = train_files[:100]
    dog_targets_short = train_targets[:100]

    human_files_short = human_files[:100]

    for dog_file in dog_files_short:
        print("Prediction:", ResNet50_predict_breed(dog_file))

        for human_file in human_files_short:
            print("Prediction:", ResNet50_predict_breed(human_file))


train_model()