#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:26:15 2025

@author: kionahutchins
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#*******************************************************************************
# starting data preperation in q.1 
#*******************************************************************************


# loading the dataset and exploring its structure to understand the image  labeling 

data = scipy.io.loadmat('umist_cropped.mat')

print("Dataset keys/folders and types:")
for key in data.keys():
    if not key.startswith("__"):
        print(f"  {key}  | shape = {data[key].shape} | dtype = {data[key].dtype}")

#now pulling all the images and labels from the dataset with error handling 
if "facedat" not in data:
    raise KeyError("The key 'facedat' was not found in the dataset.")

facedat = data['facedat']

images_list = []
labels_list = []

num_people = facedat.shape[1]
print(f"\n Total # of people in the dataset: {num_people}")

for person_id in range(num_people):
    person_images = facedat[0, person_id]

#printing dimensions for person 1 so that there is context 
    if person_id == 0:
        print(f"\nImage structure of samples: {person_images.shape}")

#parameters so that all kinds of shapes are able to be handled 
    if person_images.ndim == 2:
        
#making columns for each image 
        for img_idx in range(person_images.shape[1]):
            img_flat = person_images[:, img_idx].flatten()
            images_list.append(img_flat)
            labels_list.append(person_id + 1)

    elif person_images.ndim == 3:
 #because its a 3d image I have to analyze height x width x # of images
        for img_idx in range(person_images.shape[2]):
            img_flat = person_images[:, :, img_idx].flatten()
            images_list.append(img_flat)
            labels_list.append(person_id + 1)

    else:
        
#single image case used just in case not all the images in the dataset are 3d, 
#so this is to detect any images that might have that rare case ( not leaving anything out)
        img_flat = person_images.flatten()
        images_list.append(img_flat)
        labels_list.append(person_id + 1)

# Convert to arrays so that 
images = np.array(images_list)
labels = np.array(labels_list)

print(f"\nAll images pulled from the dataset: {images.shape[0]}")
print(f"For each image there are {images.shape[1]} features a.k.a pixels")

#now making it into a dataframe for better analysis  
pixel_columns = [f"pixel_{i}" for i in range(images.shape[1])]
df = pd.DataFrame(images, columns=pixel_columns)
df['label'] = labels

# number of samples and number of features will be displayed
print(f"DataFrame samples & features: {df.shape}")
print(df.head())

#now starting the visualization of the images 
n_features = images.shape[1]

#height and width of each shape 
if n_features == 10304:      
    img_h, img_w = 112, 92
elif n_features == 2576:     
    img_h, img_w = 56, 46
else:
# just in case 
    side = int(np.sqrt(n_features))
    img_h = img_w = side

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("all sample images from dataset", fontsize=14, fontweight="bold")

for i in range(10):
    img = df.iloc[i, :-1].values.reshape(img_h, img_w)
    label = df.iloc[i, -1]

    r, c = i // 5, i % 5
    axes[r, c].imshow(img, cmap='gray')
    axes[r, c].set_title(f"Person {label}")
    axes[r, c].axis("off")

plt.tight_layout()
plt.show()

#*******************************************************************************
# starting data splitting portion of the assignment 
#*******************************************************************************

#splitting the data stratified 
X = df.drop("label", axis=1)
y = df["label"]

#testing on 10%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.10,
    stratify=y,
    random_state=92
)

#training 70% and validating 20%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.222,     
    stratify=y_temp,
    random_state=92
)

print(f"# of training images 70%:   {len(X_train)}")
print(f"# of validation images 20%: {len(X_val)}")
print(f"# of testing images 10%:    {len(X_test)}")

#normalizing data with standard scalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#now plotting all the distribution of the images 

fig, axes = plt.subplots(1, 3, figsize=(20, 4))

y_train.value_counts().sort_index().plot(kind='bar', ax=axes[0])
axes[0].set_title('Training Distribution')
axes[0].set_xlabel('Person Label')
axes[0].set_ylabel('# Images')

y_val.value_counts().sort_index().plot(kind='bar', ax=axes[1])
axes[1].set_title('Validation Distribution')
axes[1].set_xlabel('Person Label')

y_test.value_counts().sort_index().plot(kind='bar', ax=axes[2])
axes[2].set_title('Testing Distribution')
axes[2].set_xlabel('Person Label')

plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA

# Try different PCA dimensions
components_list = [10, 20, 50, 100]

explained_variances = []

for n in components_list:
    pca = PCA(n_components=n)
    pca.fit(X_train_scaled)
    explained_variances.append(pca.explained_variance_ratio_.sum())

# Plot how much variance each PCA model keeps
plt.figure(figsize=(8,5))
plt.plot(components_list, explained_variances, marker='o')
plt.title("Explained Variance vs PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Total Explained Variance")
plt.grid(True)
plt.show()
