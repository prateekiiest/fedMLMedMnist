import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

folder_path = 'lake'
save_file = 'img_features_lake.xlsx'

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


def extract_features_batch(img_folder_path: str):
    img_files = os.listdir(img_folder_path)
    num_imgs = len(img_files)

    features_df = pd.DataFrame()

    for i in tqdm(range(num_imgs)):
        img_path = os.path.join(img_folder_path, img_files[i])
        img_features = extract_features(img_path)

        features_df[img_files[i]] = img_features

    features_df.to_excel(save_file, index=False)


extract_features_batch(folder_path)
