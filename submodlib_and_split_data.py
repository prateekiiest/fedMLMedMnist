import numpy as np
import pandas as pd
from submodlib import FacilityLocationFunction, DisparitySumFunction, DisparityMinFunction, LogDeterminantFunction, FacilityLocationMutualInformationFunction
import os
import shutil
import numpy as np
import time

print('Reading data...')
groundData = pd.read_excel('img_features_lake_9000.xlsx')
print('Finished reading data.')

groundData_transpose = groundData.copy()
image_names = groundData_transpose['image_names']
print(image_names[:10])

groundData_transpose.drop(columns=groundData_transpose.columns[0], axis=1, inplace=True)

num_images = len(groundData_transpose)
num_features = 1000
budget = int(num_images * 0.1)
print(num_images, budget)

algo = 'LogDeterminantFunction'
from submodlib import LogDeterminantMutualInformationFunction
etas = [1.8]
row = 0
index = 1

supported_algos = ['FacilityLocationFunction', 'DisparitySumFunction', 'DisparityMinFunction', 'LogDeterminantFunction']

start_time = time.time()
print('Starting selection algo ...')
if algo == 'FacilityLocationFunction':
    # objFL = FacilityLocationFunction(n=len(grounds), data=groundData, separate_rep=True, n_rep=36, data_rep=repData, mode="dense", metric="euclidean")
    objFL = FacilityLocationFunction(n=num_images, data=np.array(groundData_transpose), separate_rep=False, mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'DisparitySumFunction':
    objFL = DisparitySumFunction(n=num_images, data=np.array(groundData_transpose), mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'DisparityMinFunction':
    objFL = DisparityMinFunction(n=num_images, data=np.array(groundData_transpose), mode="dense", metric="euclidean")
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
elif algo == 'LogDeterminantFunction':
    lambda_value = 1
    objFL = LogDeterminantFunction(n=num_images, data=np.array(groundData_transpose), mode="dense", metric="euclidean", lambdaVal=lambda_value)
    print('Checkpoint')
    greedyList = objFL.maximize(budget=budget ,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, show_progress=True)
# elif algo == 'FacilityLocationMutualInformationFunction':
#     obj = FacilityLocationMutualInformationFunction(n=num_images, num_queries=len(repData_transpose), data=np.array(groundData_transpose),queryData=np.array(repData_transpose), metric="euclidean", magnificationEta=etas)
#     greedyList = obj.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False,
#                               stopIfNegativeGain=False, verbose=False)

end_time = time.time()

print(f'Time taken - {np.round((end_time - start_time) / 60, 2)}')

# greedys = [[grounds[i][x[0]] for x in greedyList] for i in range(num_features)]

greedyList = np.array(greedyList)

np.save('greedyList_9000.npy', greedyList)

selected_image_idxs = [greedyList[i][0] for i in range(len(greedyList))]

selected_image_names = [image_names['image_names'][i] for i in selected_image_idxs]

print(selected_image_names)

# Splitting data for creating train_baseline
# try:
#     os.mkdir('train_w_submodlib')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('train_w_submodlib/normal')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('train_w_submodlib/lung_op')
# except FileExistsError:
#     pass

# for i in range(len(selected_image_names)):

#     # print(selected_image_names[i])
#     img_file_source = os.path.join('all_to_clean', selected_image_names[i])
#     if selected_image_names[i].split('.')[0][0] == 'L':
#         img_file_destination = os.path.join('train_w_submodlib', 'lung_op', selected_image_names[i])
#     elif selected_image_names[i].split('.')[0][0] == 'N':
#         img_file_destination = os.path.join('train_w_submodlib', 'normal', selected_image_names[i])
#     else:
#         print('Image file destination not defined')
#         img_file_destination = None

#     print(img_file_source)
#     print(img_file_destination)
#     shutil.copy(img_file_source, img_file_destination)

# Creating split for training WGAN for normal and pneumonia

# Splitting data for creating train_baseline

# try:
#     os.mkdir('gan_train')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('gan_train/normal')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('gan_train/lung_op')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('gan_train/normal/normal')
# except FileExistsError:
#     pass

# try:
#     os.mkdir('gan_train/lung_op/lung_op')
# except FileExistsError:
#     pass

# for i in range(len(selected_image_names)):

#     # print(selected_image_names[i])

#     img_file_source = os.path.join('lake', selected_image_names[i])

#     if selected_image_names[i].split('.')[0][0] == 'L':
#         img_file_destination = os.path.join('gan_train', 'lung_op', 'lung_op', selected_image_names[i])
#     elif selected_image_names[i].split('.')[0][0] == 'N':
#         img_file_destination = os.path.join('gan_train', 'normal', 'normal', selected_image_names[i])
#     else:
#         print('Image file destination not defined')
#         img_file_destination = None

#     # print(img_file_source)
#     # print(img_file_destination)

#     shutil.copy(img_file_source, img_file_destination)