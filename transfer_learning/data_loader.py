import numpy as np

from tqdm import tqdm
from glob import glob
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
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
    leaf_files = np.array(data['filenames'])
    leaf_targets = np_utils.to_categorical(np.array(data['target']), 4)
    return leaf_files, leaf_targets


# Load train, test, and validation datasets
leaf_files, leaf_targets = load_dataset('/home/slavo/Dev/plant-disease-detection/images/')

train_files = leaf_files[:-40]
train_targets = leaf_targets[:-40]

valid_files = leaf_files[-40:-20]
valid_targets = leaf_targets[-40:-20]

test_files = leaf_files[-20:]
test_targets = leaf_targets[-20:]

print("Leaf files {}".format(len(leaf_files)))
print("Train files{}".format(len(leaf_files)))
