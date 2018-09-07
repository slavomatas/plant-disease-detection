import os

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