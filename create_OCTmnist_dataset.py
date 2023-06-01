import pandas as pd
import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import random

# read the dataset
paths = ['OCT/train/DME', 'OCT/train/DRUSEN', 'OCT/train/NORMAL']

# create a directory for each class in side 'OCT/'
for path in paths:
    os.makedirs(path.replace('/train/', '/') + '/images')

# resize the images to 299x299 resolution and save to a new folder
for path in paths:
    images = random.sample(os.listdir(path), 8000)
    for file in tqdm(images):
        im = Image.open(path + '/' + file)
        im = im.resize((299, 299))
        im.save(path.replace('/train/', '/') + '/images/' + file)

