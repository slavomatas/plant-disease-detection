import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, Xception


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


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


def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def Xception_predict_breed(img_path, top_model, dog_names):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = top_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# Plot Accuracy and Loss
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


# Extract bottleneck features using pre-trained model and datagenerator
def extract_features_with_data_augmentation(datagen, directory, sample_count, num_of_classes, batch_size):
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3))

    print("Conv Base Dims:", conv_base.output.shape)
    features = np.zeros(shape=(sample_count, conv_base.output.shape[1], conv_base.output.shape[2],
                               conv_base.output.shape[3]))  # output shape of conv base
    labels = np.zeros(shape=(sample_count, num_of_classes))  # number of classes

    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch, verbose=1)
        print("Feature batch size:", len(features_batch))
        if len(features_batch) == batch_size:
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        else:
            print("Processing last batch:", len(features_batch))
            tmp_batch_size = len(features_batch)
            features[i * batch_size: (i * batch_size) + tmp_batch_size] = features_batch
            labels[i * batch_size: (i * batch_size) + tmp_batch_size] = labels_batch
        i += 1

        if i * batch_size >= sample_count:
            break

    return features, labels


def extract_features():
    # Load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("/home/slavo/Dev/image-classification/dogImages/train/*/"))]

    # Load train, test, and validation datasets
    train_files, train_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/train')
    valid_files, valid_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/valid')
    test_files, test_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/test')

    base_dir = '/home/slavo/Dev/image-classification/dogImages/'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')

    batch_size = 100

    # Extract training bottleneck features
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        #horizontal_flip=True,
        fill_mode='nearest',
        featurewise_center=True)

    training_features, training_labels = extract_features_with_data_augmentation(train_datagen, train_dir, len(train_files),
                                                                         len(dog_names), batch_size)
    np.save('bottleneck_features/augmented_bottleneck_features_training.npy', training_features)
    np.save('bottleneck_features/augmented_training_labels.npy', training_labels)

    '''
    # Extract validation bottleneck features
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_features, validation_labels = extract_features_from_conv_base(test_datagen, validation_dir,
                                                                             len(valid_files), len(dog_names),
                                                                             batch_size)
    np.save('bottleneck_features/augmented_bottleneck_features_validation.npy', validation_features)
    np.save('bottleneck_features/augmented_validation_labels.npy', validation_labels)
    '''


# Create top model for classification
def create_model(input_shape):
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

    model.add(Dense(133, activation='softmax',
                    kernel_initializer=init,
                    bias_initializer='zeros'))

    model.summary()

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


# Train top model using bottleneck features
def train_model():
    # Load train, test, and validation datasets
    train_files, train_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/train')
    valid_files, valid_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/valid')
    test_files, test_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/test')

    # Load augmented bottleneck features
    training_features = np.load('bottleneck_features/augmented_bottleneck_features_training.npy')
    #training_labels = np.load('bottleneck_features/augmented_training_labels.npy')

    #validation_features = np.load('bottleneck_features/augmented_bottleneck_features_validation.npy')
    #validation_labels = np.load('bottleneck_features/augmented_validation_labels.npy')

    # Load and concatenate bottleneck features
    #bottleneck_features = np.load('/home/slavo/Dev/image-classification/bottleneck_features/DogXceptionData.npz')
    #training_features = np.concatenate((training_features, bottleneck_features['train']), axis=0)
    #validation_features = np.concatenate((validation_features, bottleneck_features['valid']), axis=0)
    #training_labels = np.concatenate((training_labels, training_labels), axis=0)
    #validation_labels = np.concatenate((validation_labels, validation_labels), axis=0)

    # Load bottleneck features
    bottleneck_features = np.load('/home/slavo/Dev/image-classification/bottleneck_features/DogXceptionData.npz')
    #training_features = bottleneck_features['train']
    validation_features = bottleneck_features['valid']
    testing_features = bottleneck_features['test']

    model = create_model(training_features.shape[1:])

    # ### Train the Model
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.rmsprop.hdf5',
                                   verbose=1, save_best_only=True)

    history = model.fit(training_features, train_targets,
                            validation_data=(validation_features, valid_targets),
                            epochs=500, batch_size=200, callbacks=[checkpointer], verbose=1)

    print(model.evaluate(testing_features, test_targets))

    # plot_accuracy_loss(history)


# Test top model preditions
def test_model():
    test_files, test_targets = load_dataset('/home/slavo/Dev/image-classification/dogImages/test')

    bottleneck_features = np.load('/home/slavo/Dev/image-classification/bottleneck_features/DogXceptionData.npz')
    testing_features = bottleneck_features['test']

    top_model = create_top_model(testing_features.shape[1:])

    # Load the Model with the Best Validation Loss
    top_model.load_weights('saved_models/augmented.weights.best.rmsprop.hdf5')

    print(top_model.evaluate(testing_features, test_targets))

    # Get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(top_model.predict(np.expand_dims(feature, axis=0))) for feature in testing_features]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(
        model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # TODO: Test the performance of the dog_detector function
    # on the images in human_files_short and dog_files_short.

    # dog_files_short = train_files[:100]
    # dog_targets_short = train_targets[:100]
    # human_files_short = human_files[:100]

    # for dog_file in dog_files_short:
    #   print("Prediction:", Xception_predict_breed(dog_file))

    # for human_file in human_files_short:
    #    print("Prediction:", predict_breed(human_file))


#extract_features()
train_model()
#test_model()
