import numpy as np
import pandas as pd
from submodlib import FacilityLocationFunction, DisparitySumFunction, DisparityMinFunction, LogDeterminantFunction
import numpy as np
import time
import argparse

# add arguments for dataset, lake file, algo, greedyList file, budget
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COVID', help='dataset name')
parser.add_argument('--algo', type=str, default='LogDeterminantFunction', help='algo name')
parser.add_argument('--budget', type=int, default=0.1, help='budget')

args = parser.parse_args()

print('Reading data...')
groundData = pd.read_excel('img_features_lake_{}_resnet.xlsx'.format(args.dataset))
print('Finished reading data.')

image_names = groundData.iloc[:, 0]
print('Image names: ')
print(image_names[:10])

groundData.drop(columns=groundData.columns[0], axis=1, inplace=True)

num_images = len(groundData)
num_features = 1000
budget = int(num_images * args.budget)
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

start_time = time.time()
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
np.save('greedyList_{}_{}.npy'.format(args.dataset, args.algo), greedyList)

selected_image_idxs = [greedyList[i][0] for i in range(len(greedyList))]
selected_image_names = [image_names[i] for i in selected_image_idxs]

print('Selected image names: ')
print(selected_image_names)



