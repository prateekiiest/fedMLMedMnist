import pandas as pd
import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm

# Read the dataset
train_path = 'aptos2019-blindness-detection/train_images'
train_metadata = pd.read_csv('aptos2019-blindness-detection/train.csv')

# Add the path to each image as a column in the metadata
train_metadata['path'] = train_metadata['id_code'].map(lambda x: os.path.join(train_path, '{}.png'.format(x)))

print(train_metadata.head())

# Count the number of images in each class
print(train_metadata['diagnosis'].value_counts())

# Give images of class 0 a label of 'Normal', images of class 1 and 2 a label of Mild', images of class 3 and 4 a label of 'Severe'
train_metadata['label'] = train_metadata['diagnosis'].map(lambda x: 'Normal' if x == 0 else 'Mild' if x == 1 or x == 2 else 'Severe')

print(train_metadata.head())

# Count the number of images for each label
print(train_metadata['label'].value_counts())

# Create a new directory for each label
for label in ['Normal', 'Mild', 'Severe']:
    os.makedirs('aptos2019-blindness-detection/{}/images'.format(label))

# Copy the images to the new directories and resize them to 299x299
for index, row in tqdm(train_metadata.iterrows()):
    image = Image.open(row['path'])
    image = image.resize((299, 299))
    image.save('aptos2019-blindness-detection/{}/images/{}.png'.format(row['label'], row['id_code']))