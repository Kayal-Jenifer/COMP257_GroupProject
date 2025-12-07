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



#*******************************************************************************
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Fit full PCA to compute cumulative explained variance 
pca_full = PCA().fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components needed for 95% variance 
n_components_95 = np.searchsorted(cumulative_variance, 0.95) + 1
print("Number of PCA components to retain 95% variance:", n_components_95)

# Build PCA 95% model
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_train_pca_reconstructed = pca.inverse_transform(X_train_pca)

X_test_pca = pca.transform(X_test_scaled)
X_test_pca_reconstructed = pca.inverse_transform(X_test_pca)

components_list = [10, 20, 50, 100]
explained_variances = []

for n in components_list:
    pca_temp = PCA(n_components=n)
    pca_temp.fit(X_train_scaled)
    explained_variances.append(pca_temp.explained_variance_ratio_.sum())

plt.figure(figsize=(8,5))
plt.plot(components_list, explained_variances, marker='o')
plt.title("Explained Variance vs PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Total Explained Variance")
plt.grid(True)
plt.show()

#*******************************************************************************
import matplotlib.pyplot as plt

# Choose some random indices from test set
indices = np.random.choice(len(X_test_scaled), size=5, replace=False)

plt.figure(figsize=(12, 6))
plt.suptitle("PCA Reconstruction Visualization", fontsize=16)

for i, idx in enumerate(indices):

    # Original image
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(X_test_scaled[idx].reshape(112, 92), cmap='gray')
    plt.axis("off")
    ax.set_title("Original")

    # PCA reconstructed image
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(X_test_pca_reconstructed[idx].reshape(112, 92), cmap='gray')
    plt.axis("off")
    ax.set_title("PCA Recon.")

plt.tight_layout()
plt.show()

#*******************************************************************************
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # Scales each feature to [0,1]
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Size of input (number of pixels)
input_dim = X_train_scaled.shape[1]

# Latent dimension (compressed size)
latent_dim = 50

# Build the autoencoder
input_layer = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(200, activation='relu')(input_layer)
encoded = layers.Dense(100, activation='relu')(encoded)
latent = layers.Dense(latent_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(100, activation='relu')(latent)
decoded = layers.Dense(200, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Model
autoencoder = models.Model(input_layer, output_layer)
encoder = models.Model(input_layer, latent)  # To extract latent features later

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=30,
    batch_size=32,
    validation_data=(X_val_scaled, X_val_scaled)
)

# Reconstruction from autoencoder
X_train_ae_reconstructed = autoencoder.predict(X_train_scaled)
X_val_ae_reconstructed   = autoencoder.predict(X_val_scaled)
X_test_ae_reconstructed  = autoencoder.predict(X_test_scaled)

from sklearn.decomposition import PCA
import numpy as np

# --- Compute reconstruction error for autoencoder ---
train_error_ae = np.mean(np.square(X_train_scaled - X_train_ae_reconstructed))
val_error_ae = np.mean(np.square(X_val_scaled - X_val_ae_reconstructed))
test_error_ae = np.mean(np.square(X_test_scaled - X_test_ae_reconstructed))

print("Autoencoder Reconstruction Error:")
print(f"Train: {train_error_ae:.4f}, Val: {val_error_ae:.4f}, Test: {test_error_ae:.4f}")

# --- PCA with same latent dimension ---
pca = PCA(n_components=latent_dim)
X_train_pca = pca.fit_transform(X_train_scaled)
X_train_pca_reconstructed = pca.inverse_transform(X_train_pca)

X_val_pca = pca.transform(X_val_scaled)
X_val_pca_reconstructed = pca.inverse_transform(X_val_pca)

X_test_pca = pca.transform(X_test_scaled)
X_test_pca_reconstructed = pca.inverse_transform(X_test_pca)

# --- Compute reconstruction error for PCA ---
train_error_pca = np.mean(np.square(X_train_scaled - X_train_pca_reconstructed))
val_error_pca = np.mean(np.square(X_val_scaled - X_val_pca_reconstructed))
test_error_pca = np.mean(np.square(X_test_scaled - X_test_pca_reconstructed))

print("\nPCA Reconstruction Error:")
print(f"Train: {train_error_pca:.4f}, Val: {val_error_pca:.4f}, Test: {test_error_pca:.4f}")

#*******************************************************************************
import matplotlib.pyplot as plt

# pick 5 samples from validation set for visualization
n = 5
plt.figure(figsize=(10, 4))

for i in range(n):
    # original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val_scaled[i].reshape(112, 92), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # reconstructed by autoencoder
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_val_ae_reconstructed[i].reshape(112, 92), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.suptitle("Autoencoder Reconstruction Visualization")
plt.show()



# *************************
# 4. CLUSTERING 

scaler_cluster = StandardScaler()
X_train_scaled_clean = scaler_cluster.fit_transform(X_train)

# Re-run PCA specifically for clustering (StandardScaled data)
# We use 95% variance preservation instead of a fixed number
pca_cluster = PCA(n_components=0.95, random_state=92)
X_train_reduced = pca_cluster.fit_transform(X_train_scaled_clean)

print(f"Data re-scaled and PCA re-calculated.")
print(f"PCA retained {X_train_reduced.shape[1]} components.")


# --- START CLUSTERING ALGORITHMS ---
import os
# Fixes memory leak warning on Windows
os.environ['OMP_NUM_THREADS'] = '2'

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import seaborn as sns

# 1. K-Means Clustering
# *********************
print("\n" + "="*50)
print("K-MEANS CLUSTERING (OPTIMIZED)")
print("="*50)

wcss_scores = []
silhouette_scores = []

# Testing different K values
for k in range(2, 26):
    #  n_init=20 gives the algorithm more attempts to find good centers
    kmeans = KMeans(n_clusters=k, random_state=92, n_init=20)
    kmeans.fit(X_train_reduced)
    wcss_scores.append(kmeans.inertia_)
    
    if len(np.unique(kmeans.labels_)) > 1:
        silhouette_scores.append(silhouette_score(X_train_reduced, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# Pick best K automatically
best_k = np.argmax(silhouette_scores) + 2
print(f"Best number of clusters identified: {best_k}")

# Run Final K-Means with optimized K
kmeans = KMeans(n_clusters=best_k, random_state=92, n_init=50, max_iter=500)
kmeans_labels = kmeans.fit_predict(X_train_reduced)
kmeans_silhouette = silhouette_score(X_train_reduced, kmeans_labels)
print(f"K-Means quality score: {kmeans_silhouette:.4f}")


# 2. DBSCAN Clustering
# ********************
print("\n" + "="*50)
print("DBSCAN CLUSTERING")
print("="*50)

# Using optimized parameters for UMIST
best_eps = 7.0  
best_min_samples = 3

print(f"Using DBSCAN settings: eps={best_eps}, min_samples={best_min_samples}")

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(X_train_reduced)

real_clusters_count = len(np.unique(dbscan_labels[dbscan_labels != -1]))
noise_points = np.sum(dbscan_labels == -1)

if real_clusters_count > 1:
    dbscan_silhouette = silhouette_score(X_train_reduced[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    dbscan_silhouette = 0


# 3. Visualization ( t-SNE)
# **********************************
print("\n" + "="*50)
print("CREATING OPTIMIZED VISUALIZATIONS")
print("="*50)

# OPTIMIZATION: n_iter=3000 untangles the knots in the graph
tsne = TSNE(
    n_components=2, 
    random_state=92, 
    perplexity=30, 
    n_iter=3000, 
    learning_rate='auto',
    init='pca'
)
X_2d = tsne.fit_transform(X_train_reduced)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: True Labels
scatter1 = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='tab20', alpha=0.7)
axes[0, 0].set_title('True Person Labels (Ground Truth)')
plt.colorbar(scatter1, ax=axes[0, 0])

# Plot 2: K-Means
scatter2 = axes[0, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='tab20', alpha=0.7)
axes[0, 1].set_title(f'K-Means Clustering ({best_k} clusters)')
plt.colorbar(scatter2, ax=axes[0, 1])

# Plot 3: DBSCAN
unique_dbscan_labels = np.unique(dbscan_labels)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_dbscan_labels)))

for i, label in enumerate(unique_dbscan_labels):
    if label == -1:
        mask = dbscan_labels == label
        axes[1, 0].scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', alpha=0.6, label='Noise')
    else:
        mask = dbscan_labels == label
        axes[1, 0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], alpha=0.7, label=f'Cluster {label}')
axes[1, 0].set_title(f'DBSCAN Clustering')

# Plot 4: Purity Heatmap
def calculate_cluster_purity(cluster_labels, true_labels):
    unique_clusters = np.unique(cluster_labels)
    unique_people = np.unique(true_labels)
    count_table = np.zeros((len(unique_clusters), len(unique_people)))
    
    for i, cluster_id in enumerate(unique_clusters):
        points_in_cluster = cluster_labels == cluster_id
        people_in_cluster = true_labels[points_in_cluster]
        for j, person_id in enumerate(unique_people):
            count_table[i, j] = np.sum(people_in_cluster == person_id)
            
    purity = np.sum(np.max(count_table, axis=1)) / len(cluster_labels)
    return purity, count_table

kmeans_purity, cluster_table = calculate_cluster_purity(kmeans_labels, y_train.values)
print(f"K-Means cluster purity: {kmeans_purity:.4f}")

cluster_composition = cluster_table / (cluster_table.sum(axis=1, keepdims=True) + 0.0001)
im = axes[1, 1].imshow(cluster_composition, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
axes[1, 1].set_title(f'K-Means Purity Heatmap (Purity: {kmeans_purity:.2f})')
axes[1, 1].set_xlabel('Person ID')
axes[1, 1].set_ylabel('Cluster ID')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETED")
print("="*50)
