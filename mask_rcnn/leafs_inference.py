# # Mask R-CNN Leafs Inference

import os
import sys

import skimage.io
import mrcnn.model as modellib

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/slavo/Dev/plant-disease-detection")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
LEAFS_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_leafs.h5")

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images/roztoc")
IMAGE_DIR = os.path.join(ROOT_DIR, "images/esca")

# We'll be using a model trained on the Leafs dataset.

from mask_rcnn.leafs import LeafsConfig


class InferenceConfig(LeafsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(LEAFS_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID.
class_names = ['BG', 'esca']

# Load and run object detection random samples

file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:

    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=1)

    # Check results
    #r = results[0]

    # Regions of interest
    #results[0]['rois']

    print("Image {} ROI detected {}".format(file_name, len(results[0]['rois'])))
