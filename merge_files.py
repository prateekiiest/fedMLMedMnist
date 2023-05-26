import numpy as np
import pandas as pd
import time

start_time = time.time()
print('Loading greedyList...', time.time() - start_time)
greedyList = np.load('greedyList.npy')
greedyList = pd.DataFrame(greedyList, columns=['selected_image_indices', 'value'])

greedyList = greedyList.astype({'selected_image_indices': 'Int32', 'value': 'Float32'})
print(greedyList.head(), time.time() - start_time)
print('Loading ground data...', time.time() - start_time)
groundData = pd.read_excel('img_features_lake.xlsx')

image_names = groundData['image_names']
print(image_names.head(), time.time() - start_time)

selected_image_idxs = greedyList[greedyList.columns[0]]
print(selected_image_idxs.head(), time.time() - start_time)

selected_image_names = image_names.iloc[selected_image_idxs]
print(selected_image_names.head(), time.time() - start_time)

greedyList['selected_image_names'] = selected_image_names
print(greedyList)