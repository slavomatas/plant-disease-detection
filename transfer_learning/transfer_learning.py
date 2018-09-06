import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam

from tqdm import tqdm
from glob import glob
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


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


def Xception_predict(img_path, model, dog_names):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return leaf category - healthy vs damaged
    return np.argmax(predicted_vector)]


# Model Architecture
#
# We only add a global average pooling layer and a fully connected layer,
# where the latter contains one node for each dog category and is equipped with a softmax.
def create_model(input_shape):
    # Truncated Normal weight initializer
    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))

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

    # Adam optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def train_model():

    # Load train, test, and validation datasets
    train_files, train_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/train')
    valid_files, valid_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/valid')
    test_files, test_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/test')

    # Load bottleneck features
    bottleneck_features = np.load('/home/slavo/Dev/image-classification/bottleneck_features/DogXceptionData.npz')
    training_features = bottleneck_features['train']
    validation_features = bottleneck_features['valid']
    testing_features = bottleneck_features['test']

    model = create_model(training_features.shape[1:])

    # ### Train the Model
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5',
                                   verbose=1, save_best_only=True)

    history = model.fit(training_features, train_targets,
                            validation_data=(validation_features, valid_targets),
                            epochs=500, batch_size=20, callbacks=[checkpointer], verbose=1)

    print(model.evaluate(testing_features, test_targets))

    #plot_accuracy_loss(history)


# ### Test the Model
#
# Now, we can use the CNN to test how well it identifies .
# We print the test accuracy below.
def test_model():

    # Load train, test, and validation datasets
    train_files, train_targets = load_dataset('/home/slavo/Dev/plan-disease-detection/images/train')
    valid_files, valid_targets = load_dataset('/home/slavo/Dev/plan-disease-detection/images/valid')
    test_files, test_targets = load_dataset('/home/slavo/Dev/plan-disease-detection/images/test')

    # Load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("/home/slavo/Dev/image-classification/dogImages/train/*/"))]

    # Print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.' % len(test_files))

    import random
    random.seed(8675309)

    # Load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/*/*"))
    random.shuffle(human_files)

    # Print statistics about the dataset
    print('There are %d total human images.' % len(human_files))

    bottleneck_features = np.load('/home/slavo/Dev/image-classification/bottleneck_features/DogXceptionData.npz')
    test_features = bottleneck_features['test']

    # Create model
    model = create_model(test_features.shape[1:])

    # Load the model weights with the best validation loss
    model.load_weights('saved_models/weights.best.Xception.hdf5')

    print(model.evaluate(test_features, test_targets))

    # Get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(model.predict(np.expand_dims(test_feature, axis=0))) for test_feature in test_features]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    #model_predictions = []
    #for test_file, test_target in zip(test_files, test_targets):
    #    bottleneck_feature = extract_Xception(path_to_tensor(test_file))
    #    model_prediction = np.argmax(model.predict(bottleneck_feature))
    #    model_predictions.append(model_prediction)
    #    print("Model prediction:", model_prediction, "Target:", np.argmax(test_target))

    # report test accuracy
    #test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(model_predictions)
    #print('Test accuracy: %.4f%%' % test_accuracy)

    # TODO: Test the performance of the dog_detector function
    # on the images in human_files_short and dog_files_short.

    dog_files_short = train_files[:100]
    dog_targets_short = train_targets[:100]
    human_files_short = human_files[:100]

    model_predictions = []
    for dog_file, dog_target in zip(dog_files_short, dog_targets_short):
        bottleneck_feature = extract_Xception(path_to_tensor(dog_file))
        model_prediction = np.argmax(model.predict(bottleneck_feature))
        model_predictions.append(model_prediction)
        print("Model prediction:", model_prediction, "Target:", np.argmax(dog_target))

    # Report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(dog_targets_short, axis=1)) / len(model_predictions)
    print('Dog test accuracy: %.4f%%' % test_accuracy)

    #for human_file in human_files_short:
    #    print("Prediction:", predict_breed(human_file, top_model, dog_names))


train_model()
#test_model()

#dog_file = "/home/slavo/Downloads/zara-1.jpg"
