
import numpy as np
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split


# TODO : Restructure the code to make it more modular
# Load the data for Covid

def getDataClient1():
    levels = ['Normal/images', 'COVID/images']
    path = 'covid19-radiography-database/COVID-19_Radiography_Dataset'
    data_dir = os.path.join(path)

    data = []
    for id, level in enumerate(levels):
        for file in os.listdir(os.path.join(data_dir, level)):
            data.append(['{}/{}'.format(level, file), level])


    data = pd.DataFrame(data, columns = ['image_file', 'corona_result'])


    data['path'] = path + '/' + data['image_file']
    data['corona_result'] = data['corona_result'].map({'Normal/images': 'Negative', 'COVID/images': 'Positive'})
    data = data.sample(frac=1).reset_index(drop=True) # Shuffle dataframe

    data['image'] = data['path'].map(lambda x: np.asarray(Image.open(x)))
    data["output"] = data["corona_result"].astype('category').cat.codes


    train, test = train_test_split(data, test_size=0.2)
    return train, test

def getDataClient2():


    # Second client data

    # Load the data for Lung Opacity

    levels2 = ['Normal/images', 'Lung_Opacity/images']
    path2 = 'covid19-radiography-database/COVID-19_Radiography_Dataset'
    data_dir2 = os.path.join(path2)

    data2 = []
    for id, level in enumerate(levels2):
        for file in os.listdir(os.path.join(data_dir2, level)):
            data2.append(['{}/{}'.format(level, file), level])

    data2 = pd.DataFrame(data2, columns = ['image_file', 'lung_opacity_result'])


    data2['path'] = path2 + '/' + data2['image_file']
    data2['lung_opacity_result'] = data2['lung_opacity_result'].map({'Normal/images': 'Negative', 'Lung_Opacity/images': 'Positive'})
    data2 = data2.sample(frac=1).reset_index(drop=True) # Shuffle dataframe

    data2['image'] = data2['path'].map(lambda x: np.asarray(Image.open(x)))
    data2["output"] = data2["lung_opacity_result"].astype('category').cat.codes
    train, test = train_test_split(data2, test_size=0.2)
    return train, test