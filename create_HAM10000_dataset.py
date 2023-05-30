import pandas as pd
import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm

# Read the dataset
path1 = 'dataverse_files/HAM10000_images_part_1'
path2 = 'dataverse_files/HAM10000_images_part_2'
metadata = pd.read_csv('dataverse_files/HAM10000_metadata')

# Create a dictionary of the images
imageid_path_dict1 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path1, '*.jpg'))}
imageid_path_dict2 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path2, '*.jpg'))}

# Merge the dictionaries
imageid_path_dict = {**imageid_path_dict1, **imageid_path_dict2}

# Create a column of the images
metadata['path'] = metadata['image_id'].map(imageid_path_dict.get)

# Plot the class distribution
print(metadata['dx'].value_counts())

# Delete images that are not in the class 'bkl', 'nv', 'mel'
metadata = metadata[metadata['dx'].isin(['bkl', 'nv', 'mel'])]

# sample 1000 images from each class
metadata = pd.concat([  
                        metadata[metadata['dx'] == 'bkl'].sample(1099), 
                        metadata[metadata['dx'] == 'nv'].sample(3000),
                        metadata[metadata['dx'] == 'mel'].sample(1113)  
                    ])

# Find out the resolution of the images
widths = []
heights = []
for path in metadata['path']:
    im = Image.open(path)
    width, height = im.size
    widths.append(width)
    heights.append(height)
    break

print('The resolution of the images is: {}x{}'.format(width, height))

# Resize the images to 299x299 resolution and save to a new folder

# Create a new folder for each class in the dataset
pwd = os.getcwd()
if not os.path.exists('dataverse_files/bkl/images'):
    path = os.path.join(pwd, 'dataverse_files/bkl/images')
    os.makedirs(path)
if not os.path.exists('dataverse_files/nv/images'):
    path = os.path.join(pwd, 'dataverse_files/nv/images')
    os.makedirs(path)
if not os.path.exists('dataverse_files/mel/images'):
    path = os.path.join(pwd, 'dataverse_files/mel/images')
    os.makedirs(path)

# iterate through each row in the metadata and resize the images
for i, row in tqdm(metadata.iterrows()):
    im = Image.open(row['path'])
    im = im.resize((299, 299))
    # save it in the appropriate folder
    if row['dx'] == 'bkl':
        im.save('dataverse_files/bkl/images/{}.jpg'.format(row['image_id']))
    elif row['dx'] == 'nv':
        im.save('dataverse_files/nv/images/{}.jpg'.format(row['image_id']))
    elif row['dx'] == 'mel':
        im.save('dataverse_files/mel/images/{}.jpg'.format(row['image_id']))

# Save the metadata to a csv file in the same directory
metadata.to_csv('dataverse_files/HAM10000_metadata_resized.csv', index=False)
