# Merging three scripts into one : feature_extractor_batch.py, submodlib_and_split_data.py, merge_files.py
# Requirements to run this script :
# 1. Dataset folder should be in the same directory as this script and have the names as mentioned in dataset_folder_names
# 2. Dataset folder should have subfolders for each class and the names of the subfolders should be as mentioned in dataset_class_names
# 3. Each subfolder should have images of that class
# Here is an example of the directory structure :
# COVID-19_Radiography_Database
# ├── Normal
# │   ├── images
# │   │   ├── Normal-1.png

# aptos2019-blindness-detection
# ├── Normal
# │   ├── images
# │   │   ├── 0a4e1a29ffff.png

# OCT
# ├── NORMAL
# │   ├── images
# │   │   ├── NORMAL-1384-6.jpeg

import numpy as np
import pandas as pd
import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import random
import argparse
from submodlib import FacilityLocationFunction, DisparitySumFunction, DisparityMinFunction, LogDeterminantFunction

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COVID', help='dataset name')
parser.add_argument('--algo', type=str, default='logdet', help='algo name')
parser.add_argument('--budget', type=str, default="0.1", help='budget')
parser.add_argument('--sample', type=str, default='False', help='sample or not')
parser.add_argument('--sample_size', type=int, default=3000, help='sample size')
parser.add_argument('--lake_file_given', type=str, default='False', help='is lake file already computed or not')
args = parser.parse_args()


def extract_features(img_path: str) -> list:
    """

    :param img_path: Path to where image is stored
    :return: List of features
    """

    # Load an example image
    img = Image.open(img_path)

    # Apply the transformation and convert the image to a tensor
    img_tensor = transform(img).unsqueeze(0)

    # Extract the features using the ResNet18 model
    with torch.no_grad():
        features = resnet(img_tensor)

    # Flatten the features and convert to a 1D numpy array
    features = features.squeeze().numpy()
    features = features.flatten()

    # Print the shape of the features array
    # print(features.shape)

    return list(features)

def extract_features_resnet(img_files, img_folder_path: str):
    num_imgs = len(img_files)

    features_df = pd.DataFrame(columns=range(512))

    for i in tqdm(range(num_imgs)):
        img_path = os.path.join(img_folder_path, img_files[i])
        img_features = extract_features(img_path)

        features_df.loc[img_files[i]] = img_features

    print(features_df.head())

    return features_df

start_time = time.time()

pwd = os.getcwd()
dataset_names = ['COVID', 'OCT', 'aptos']
if args.dataset not in dataset_names:
    raise Exception('Invalid dataset name. Valid names are: ' + str(dataset_names))
dataset_folder_names = {'COVID': 'COVID-19_Radiography_Dataset',
                        'OCT': 'OCT',
                        'aptos': 'aptos2019-blindness-detection'}
dataset_class_names = {'COVID': ['Normal', 'COVID', 'Lung_Opacity'],
                       'OCT': ['NORMAL', 'DME', 'DRUSEN'],
                       'aptos': ['Normal', 'Mild', 'Severe']}

dataset_folder_name = dataset_folder_names[args.dataset]
dataset_class_name = dataset_class_names[args.dataset]
paths = [pwd + '/' + dataset_folder_name + '/' + dataset_class_name[i] + '/images' for i in range(len(dataset_class_name))]

# Load the pre-trained ResNet18 model
resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

# Remove the last layer (the classifier)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)

# Set the model to evaluation mode
resnet.eval()

# Define the transformation to be applied to each image
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

f_names = dataset_class_names[args.dataset]
groundData = pd.DataFrame()
save_file = 'img_features_lake_{}_resnet.xlsx'.format(args.dataset)

if args.lake_file_given == "True":
    print("Skipping lake file generation...")
else:
    for f_name in f_names:
        print('Extracting features for {} ...'.format(f_name))
        folder_path = os.path.join(pwd, dataset_folder_names[args.dataset], f_name, 'images')
        if args.sample == 'True':
            img_files = random.sample(os.listdir(folder_path), args.sample_size)
        else:
            img_files = os.listdir(folder_path)

        df = extract_features_resnet(img_files, folder_path)
        groundData = pd.concat([groundData, df])

    groundData.to_excel(save_file)

print('Reading data...')
groundData = pd.read_excel('img_features_lake_{}_resnet.xlsx'.format(args.dataset))
print('Finished reading data.')

image_names = groundData.iloc[:, 0]
print('Image names: ')
print(image_names[:10])

groundData.drop(columns=groundData.columns[0], axis=1, inplace=True)
print('Ground Data:')
print(groundData.head())

num_images = len(groundData)
budget = int(num_images * float(args.budget))
print(num_images, budget)

if args.algo == 'facloc':
    algo = 'FacilityLocationFunction'
elif args.algo == 'dispsum':
    algo = 'DisparitySumFunction'
elif args.algo == 'dispmin':
    algo = 'DisparityMinFunction'
elif args.algo == 'logdet':
    algo = 'LogDeterminantFunction'
else:
    raise Exception('Invalid algo name. Valid names are: facloc, dispsum, dispmin, logdet')

print('Starting selection algo ...')
if algo == 'FacilityLocationFunction':
    objFL = FacilityLocationFunction(n=num_images, data=np.array(groundData), separate_rep=False, mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'DisparitySumFunction':
    objFL = DisparitySumFunction(n=num_images, data=np.array(groundData), mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'DisparityMinFunction':
    objFL = DisparityMinFunction(n=num_images, data=np.array(groundData), mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'LogDeterminantFunction':
    lambda_value = 1
    objFL = LogDeterminantFunction(n=num_images, data=np.array(groundData), mode="dense", metric="euclidean", lambdaVal=lambda_value)
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

end_time = time.time()

print(f'Time taken - {np.round((end_time - start_time) / 60, 2)}')

greedyList = np.array(greedyList)
greedyList_file_name = 'greedyList_{}_{}.npy'.format(args.dataset, args.algo)
np.save(greedyList_file_name, greedyList)

greedyList = pd.DataFrame(greedyList, columns=['selected_image_indices', 'value'])
greedyList = greedyList.astype({'selected_image_indices': 'Int32', 'value': 'Float32'})
print(greedyList.head(), time.time() - start_time)

print("Length of greedyList: ")
print(len(greedyList))

selected_image_idxs = greedyList[greedyList.columns[0]]
print('Selected image indices: ')
print(selected_image_idxs.head(), time.time() - start_time)

selected_image_names = list(image_names.iloc[selected_image_idxs])
print('Selected image names: ')
print(selected_image_names[:10], time.time() - start_time)

greedyList['selected_image_names'] = selected_image_names
print('Greedy list with image names: ')
print(greedyList)

images = {}

for path in paths:
    img_files = os.listdir(path)
    print(img_files[0])
    num_images = len(img_files)
    for i in range(num_images):
        img_path = os.path.join(path, img_files[i])
        img = Image.open(img_path)
        images[img_files[i]] = np.array(img)
        img.close()

selected_images = []
for name in selected_image_names:
    selected_images += [images[name]]

greedyList['selected_images'] = selected_images
print('Greedy list with images: ')
print(greedyList)

if args.dataset == 'COVID':
    greedyList.to_csv('ten_percent_subset_{}_{}.csv'.format(args.dataset, args.algo), index=False)

    count = {dataset_class_names[args.dataset][i]: 0 for i in range(len(dataset_class_names[args.dataset]))}
    for index, row in greedyList.iterrows():
        count[row['selected_image_names'].split('-')[0]] += 1

    print(count)

elif args.dataset == 'OCT':
    greedyList.to_csv('ten_percent_subset_{}_{}.csv'.format(args.dataset, args.algo), index=False)

    count = {dataset_class_names[args.dataset][i]: 0 for i in range(len(dataset_class_names[args.dataset]))}
    for index, row in greedyList.iterrows():
        count[row['selected_image_names'].split('-')[0]] += 1

    print(count)

elif args.dataset == 'aptos':
    # get list of image names in each class
    image_names = {}
    for path in paths:
        img_files = os.listdir(path)
        num_images = len(img_files)
        for i in range(num_images):
            # extract class name from the path
            class_name = path.split('/')[-2]
            image_names[img_files[i]] = class_name

    # Now using this dictionary, get the class names for the selected images and add as a column to greedyList
    selected_image_classes = []
    for name in selected_image_names:
        selected_image_classes += [image_names[name]]

    greedyList['selected_image_classes'] = selected_image_classes
    print(greedyList.head())
    greedyList.to_csv('ten_percent_subset_{}_{}.csv'.format(args.dataset, args.algo), index=False)

    # count the number of images in each class
    count = {dataset_class_names[args.dataset][i]: 0 for i in range(len(dataset_class_names[args.dataset]))}
    for index, row in greedyList.iterrows():
        count[row['selected_image_classes']] += 1

    print(count)