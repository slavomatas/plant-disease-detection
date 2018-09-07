import os
import numpy as np

from leafs import LeafsConfig, LeafsDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/slavo/Dev/plant-disease-detection")

config = LeafsConfig()
LEAFS_DIR = os.path.join(ROOT_DIR, "datasets/macik")

dataset = LeafsDataset()
dataset.load_leafs(LEAFS_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    #visualize.display_top_masks(image, mask, class_ids, dataset.class_names)