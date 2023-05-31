
import numpy as np
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms


def getDataClient():
    
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
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

    for k in range(len(data.image)):
      data.image[k] = apply_transform(data.image[k])    

    
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

    for k in range(len(data2.image)):
      data2.image[k] = apply_transform(data2.image[k])    


    common_data= pd.DataFrame({"result": data['corona_result']+ "_" +"corona", "image": data['image'], "client_id": "1" })
    common_data2 = pd.DataFrame({"result": data2['lung_opacity_result']+"_"+"lung_opacity", "image": data2['image'], "client_id": "2" })

    finalCommon = pd.concat([common_data, common_data2])
    finalCommon["output"] = finalCommon["result"].astype('category').cat.codes
    finalCommon = finalCommon.drop("result", axis=1)



    trainDataSet, testDataSet = train_test_split(finalCommon, test_size=0.2)

    trainClient1 = trainDataSet[trainDataSet["client_id"]=="1"]
    trainClient2 = trainDataSet[trainDataSet["client_id"]=="2"]
    testClient1 = testDataSet[testDataSet["client_id"]=="1"]
    testClient2 = testDataSet[testDataSet["client_id"]=="2"]

    return trainClient1, trainClient2, testClient1, testClient2

class MyDataset(Dataset):
 
  def __init__(self,data):
  
    x=data.image.values
    y=data.output.values.astype(np.float32)
    x=np.stack(x).astype(np.float32)
    y = np.stack(y).astype(np.float32)
    self.image=torch.from_numpy(x)
    self.label=torch.from_numpy(y)
 
  def __len__(self):
    return len(self.label)
   
  def __getitem__(self,idx):
    return self.image[idx],self.label[idx]
  




def getDataClientSubset(sampleFrac):
    
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
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
    data = data.sample(frac=sampleFrac).reset_index(drop=True) # Shuffle dataframe

    data['image'] = data['path'].map(lambda x: np.asarray(Image.open(x)))

    for k in range(len(data.image)):
      data.image[k] = apply_transform(data.image[k])    

    
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
    data2 = data2.sample(frac=sampleFrac).reset_index(drop=True) # Shuffle dataframe

    data2['image'] = data2['path'].map(lambda x: np.asarray(Image.open(x)))

    for k in range(len(data2.image)):
      data2.image[k] = apply_transform(data2.image[k])    


    common_data= pd.DataFrame({"result": data['corona_result']+ "_" +"corona", "image": data['image'], "client_id": "1" })
    common_data2 = pd.DataFrame({"result": data2['lung_opacity_result']+"_"+"lung_opacity", "image": data2['image'], "client_id": "2" })

    finalCommon = pd.concat([common_data, common_data2])
    finalCommon["output"] = finalCommon["result"].astype('category').cat.codes
    finalCommon = finalCommon.drop("result", axis=1)



    trainDataSet, testDataSet = train_test_split(finalCommon, test_size=0.2)

    trainClient1 = trainDataSet[trainDataSet["client_id"]=="1"]
    trainClient2 = trainDataSet[trainDataSet["client_id"]=="2"]
    testClient1 = testDataSet[testDataSet["client_id"]=="1"]
    testClient2 = testDataSet[testDataSet["client_id"]=="2"]

    return trainClient1, trainClient2, testClient1, testClient2







def dropRedundant(subset_client1_df):
    subset_client1_df = subset_client1_df.drop("selected_image_names",axis=1)
    subset_client1_df = subset_client1_df.drop("selected_images",axis=1)
    subset_client1_df = subset_client1_df.drop("selected_image_indices",axis=1)
    return subset_client1_df


def getDataClientSubModlib():
   
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    subset_df = pd.read_csv("../ten_percent_subset_9000_logdet.csv")
    subset_df = subset_df.drop("Unnamed: 0",axis=1)

    subset_client2_df= subset_df[subset_df["selected_image_names"].str.startswith("Lung")]
    subset_client1_df= subset_df[subset_df["selected_image_names"].str.startswith("COVID")]
    subset_normal_df= subset_df[subset_df["selected_image_names"].str.startswith("Normal")]

    path='covid19-radiography-database/COVID-19_Radiography_Dataset/Lung_Opacity/images/'
    subset_client2_df['image'] = subset_client2_df['selected_image_names'].map(lambda x: np.asarray(Image.open(path+x)))

    path='covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/images/'
    subset_client1_df['image'] = subset_client1_df['selected_image_names'].map(lambda x: np.asarray(Image.open(path+x)))

    path='covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/images/'
    subset_normal_df['image'] = subset_normal_df['selected_image_names'].map(lambda x: np.asarray(Image.open(path+x)))


    subset_client1_df['image'] = subset_client1_df['image'].map(lambda x: apply_transform(x))
    subset_client2_df['image'] = subset_client2_df['image'].map(lambda x: apply_transform(x))
    subset_normal_df['image'] = subset_normal_df['image'].map(lambda x: apply_transform(x))
    subset_client2_df["result"] = "Positive_lung_opacity"
    subset_client1_df["result"] = "Positive_corona"

    subset_client1_df = dropRedundant(subset_client1_df)
    subset_client2_df = dropRedundant(subset_client2_df)
    subset_normal_df = dropRedundant(subset_normal_df)
    
    subset_normal_df_client1 = subset_normal_df.copy(deep=True)
    subset_normal_df_client2 = subset_normal_df.copy(deep=True)
    subset_normal_df_client2["result"] = "Negative_lung_opacity"
    subset_normal_df_client1["result"] = "Negative_corona"

    client1_df_subset = pd.concat([subset_client1_df,subset_normal_df_client1])
    client2_df_subset = pd.concat([subset_client2_df,subset_normal_df_client2])
    client1_df_subset["client_id"] = 1
    client2_df_subset["client_id"] = 2


    finalCommon = pd.concat([client1_df_subset, client2_df_subset])
    finalCommon["output"] = finalCommon["result"].astype('category').cat.codes
    finalCommon = finalCommon.drop("result", axis=1)
    
    trainDataSet, testDataSet = train_test_split(finalCommon, test_size=0.2)

    trainClient1 = trainDataSet[trainDataSet["client_id"]=="1"]
    trainClient2 = trainDataSet[trainDataSet["client_id"]=="2"]
    testClient1 = testDataSet[testDataSet["client_id"]=="1"]
    testClient2 = testDataSet[testDataSet["client_id"]=="2"]

    return trainClient1, trainClient2, testClient1, testClient2


