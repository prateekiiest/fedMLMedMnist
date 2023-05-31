import numpy as np
import pandas as pd
import time
import os
from PIL import Image
import argparse

# add arguments for dataset, lake file, algo, greedyList file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COVID', help='dataset name')
parser.add_argument('--lake_file', type=str, default='img_features_lake_COVID_resnet.xlsx', help='lake file name')
parser.add_argument('--algo', type=str, default='logdet', help='algo name')
parser.add_argument('--greedyList_file', type=str, default='greedyList.npy', help='greedyList file name')
args = parser.parse_args()

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

start_time = time.time()
print('Loading greedyList...', time.time() - start_time)

greedyList = np.load(args.greedyList_file)
greedyList = pd.DataFrame(greedyList, columns=['selected_image_indices', 'value'])

greedyList = greedyList.astype({'selected_image_indices': 'Int32', 'value': 'Float32'})
print(greedyList.head(), time.time() - start_time)

print("Length of greedyList: ")
print(len(greedyList))

print('Loading ground data...', time.time() - start_time)
groundData = pd.read_excel(args.lake_file)

image_names = groundData.iloc[:, 0]
print('Image names: ')
print(image_names.head(), time.time() - start_time)

selected_image_idxs = greedyList[greedyList.columns[0]]
print('Selected image indices: ')
print(selected_image_idxs.head(), time.time() - start_time)

selected_image_names = list(image_names.iloc[selected_image_idxs])
print('Selected image names: ')
print(selected_image_names, time.time() - start_time)

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