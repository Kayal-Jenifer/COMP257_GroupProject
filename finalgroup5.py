import os
import scipy.io
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras import layers, models

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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
    random_state=42
)

#training 70% and validating 20%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.222,     
    stratify=y_temp,
    random_state=42
)


print(f"# of training images 70%:   {len(X_train)}")
# Data Preparation for CNN
# Re-splitting to ensure consistency and avoid any previous mutations
print("Re-splitting data for CNN to ensure consistency...")

# We use the original dataframe 'df'
X_cnn = df.drop("label", axis=1)
y_cnn = df["label"]

# 1. Split Test (10%)
X_temp_cnn, X_test_cnn_raw, y_temp_cnn, y_test_cnn_raw = train_test_split(
    X_cnn, y_cnn,
    test_size=0.10,
    stratify=y_cnn,
    random_state=42
)

# 2. Split Train (70%) and Val (20%) from the remaining 90%
# 0.222 * 0.9 ~= 0.2
X_train_cnn_raw, X_val_cnn_raw, y_train_cnn_raw, y_val_cnn_raw = train_test_split(
    X_temp_cnn, y_temp_cnn,
    test_size=0.222,
    stratify=y_temp_cnn,
    random_state=42
)

# 3. Scale
scaler_cnn = StandardScaler()
X_train_cnn_scaled = scaler_cnn.fit_transform(X_train_cnn_raw)
X_val_cnn_scaled   = scaler_cnn.transform(X_val_cnn_raw)
X_test_cnn_scaled  = scaler_cnn.transform(X_test_cnn_raw)

# 4. Reshape
print(f"Reshaping data to ({img_h}, {img_w}, 1)...")
X_train_cnn = X_train_cnn_scaled.reshape(-1, img_h, img_w, 1)
X_val_cnn   = X_val_cnn_scaled.reshape(-1, img_h, img_w, 1)
X_test_cnn  = X_test_cnn_scaled.reshape(-1, img_h, img_w, 1)

print("Train CNN shape:", X_train_cnn.shape)
print("Val CNN shape:  ", X_val_cnn.shape)
print("Test CNN shape: ", X_test_cnn.shape)

# 5. One-hot encoding
num_classes = len(np.unique(y_cnn))
y_train_cat = keras.utils.to_categorical(y_train_cnn_raw - 1, num_classes)
y_val_cat   = keras.utils.to_categorical(y_val_cnn_raw - 1, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test_cnn_raw - 1, num_classes)

print("DEBUG: Checking shapes before CNN training")
print(f"X_train_cnn: {X_train_cnn.shape}")
print(f"y_train_cat: {y_train_cat.shape}")
print(f"X_val_cnn:   {X_val_cnn.shape}")
print(f"y_val_cat:   {y_val_cat.shape}")

print(f"# of validation images 20%: {len(X_val)}")
print(f"# of testing images 10%:    {len(X_test)}")

#normalizing data with standard scalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train_scaled.shape)
print("Val shape:", X_val_scaled.shape)
print("Test shape:", X_test_scaled.shape)

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



#********************************* Before PCA tsne visualization ************************************

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=y_train, cmap="tab20", s=8)
plt.title("Before PCA (t-SNE Visualization)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

#********************************* Apply PCA *********************************************************

components_list = [10, 20, 50, 100]

pca_models = {}
X_test_recon_dict = {}
explained_variances = {}

for n in components_list:
    print(f"\n=== PCA with {n} components ===")

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # store
    pca_models[n] = pca
    X_test_recon_dict[n] = scaler.inverse_transform(pca.inverse_transform(X_test_pca))

    # explained variance ratio
    exp_var = pca.explained_variance_ratio_.sum()
    explained_variances[n] = exp_var
    print(f"Explained Variance Ratio: {exp_var:.4f}")


plt.figure(figsize=(8,5))
plt.plot(components_list,[explained_variances[n] for n in components_list],marker='o')
plt.title("Explained Variance vs PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.show()

#**************************** Class distribution after PCA *********************************************

# AFTER PCA → labels DO NOT CHANGE, only features change
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
y_train.value_counts().sort_index().plot(kind='bar')
plt.title("Training Distribution AFTER PCA")

plt.subplot(1,3,2)
y_val.value_counts().sort_index().plot(kind='bar')
plt.title("Validation Distribution AFTER PCA")

plt.subplot(1,3,3)
y_test.value_counts().sort_index().plot(kind='bar')
plt.title("Testing Distribution AFTER PCA")

plt.tight_layout()
plt.show()

#************************* After PCA visualization ****************************************************

pca_vis = PCA(n_components=2)
X_train_pca2 = pca_vis.fit_transform(X_train_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_train_pca2[:,0], X_train_pca2[:,1], c=y_train, cmap="tab20", s=8)
plt.title("After PCA (2D PCA Visualization)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#********************************* PCA Reconstruction Visualization ************************************

indices = np.random.choice(len(X_test_scaled), size=5, replace=False)

# Store original before any transformation
X_test_original = X_test.values
X_test_pca_reconstructed_original = X_test_recon_dict[100]

plt.figure(figsize=(12, 6)) 
plt.suptitle("PCA Reconstruction (Original vs Reconstructed)", fontsize=16)

for i, idx in enumerate(indices):

    # Original image (REAL pixel space)
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(X_test_original[idx].reshape(112, 92), cmap='gray')
    plt.axis("off")
    ax.set_title("Original")

    # Reconstructed image (REAL pixel space)
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(X_test_pca_reconstructed_original[idx].reshape(112, 92), cmap='gray')
    plt.axis("off")
    ax.set_title("Reconstructed")

plt.tight_layout()
plt.show()

#************************************** Autoencoders ******************************************************

from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# For reconstruction later
X_test_original = scaler.inverse_transform(X_test_scaled)

# autoencoder model

input_dim = X_train_scaled.shape[1]
latent_dim = 100

input_layer = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(500, activation='relu')(input_layer)
encoded = layers.Dense(300, activation='relu')(encoded)
encoded = layers.Dense(150, activation='relu')(encoded)
latent  = layers.Dense(latent_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(150, activation='relu')(latent)
decoded = layers.Dense(300, activation='relu')(decoded)
decoded = layers.Dense(500, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='linear')(decoded)   # FIXED

autoencoder = models.Model(input_layer, output_layer)
encoder = models.Model(input_layer, latent)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

#*********************************************************************************************************

# Train the autoencoder
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=80,
    batch_size=32,
    validation_data=(X_val_scaled, X_val_scaled)
)

# reconstruction
X_test_ae_reconstructed = autoencoder.predict(X_test_scaled)

# Convert AE output BACK to real pixel scale
X_test_ae_reconstructed_original = scaler.inverse_transform(X_test_ae_reconstructed)


# PCA (same latent_dim)
pca = PCA(n_components=latent_dim)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_test_pca_reconstructed = pca.inverse_transform(X_test_pca)

# reconstruction error
train_error_ae = np.mean((X_train_scaled - autoencoder.predict(X_train_scaled))**2)
test_error_ae  = np.mean((X_test_scaled  - X_test_ae_reconstructed)**2)

train_error_pca = np.mean((X_train_scaled - pca.inverse_transform(X_train_pca))**2)
test_error_pca  = np.mean((X_test_scaled  - X_test_pca_reconstructed)**2)

print("Autoencoder Reconstruction Error:")
print(train_error_ae, test_error_ae)

print("\nPCA Reconstruction Error:")
print(train_error_pca, test_error_pca)

#************************* Autoencoders Reconstruction Visualization ***********************************

IMG_H, IMG_W = 112, 92
indices = [0, 5, 10, 15]

plt.figure(figsize=(12, 3))

for i, idx in enumerate(indices):

    # Original
    plt.subplot(2, len(indices), i + 1)
    plt.imshow(X_test_original[idx].reshape(IMG_H, IMG_W), cmap='gray')
    plt.title(f"Original")
    plt.axis('off')

    # Reconstructed
    plt.subplot(2, len(indices), i + 1 + len(indices))
    plt.imshow(X_test_ae_reconstructed_original[idx].reshape(IMG_H, IMG_W), cmap='gray')
    plt.title(f"Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()

#****************************************** Clustering ************************************************************

# Helper function for Purity Score
from sklearn.metrics import confusion_matrix
import numpy as np

def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# already trained AutoEncoder1 and have the encoder
X_train_ae = encoder.predict(X_train_scaled)
X_test_ae  = encoder.predict(X_test_scaled)

print("AutoEncoder1 latent shape:", X_train_ae.shape)

#********************************** K-Means Clustering *****************************************************

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_classes = len(np.unique(y_train))
print("Number of actual people (true classes):", n_classes)

print("\n--- K-Means Clustering (AE latent) ---")

kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_ae)

kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_train_ae, kmeans_labels)
kmeans_purity = purity_score(y_train, kmeans_labels)

print("Inertia:", kmeans_inertia)
print("Silhouette:", kmeans_silhouette)
print("Purity:", kmeans_purity)

#****************************************** Elbow Method (Inertia vs K) *************************************

import matplotlib.pyplot as plt

K_range = range(2, 31)
inertia_values = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_ae)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Plot (AutoEncoder Latent Space)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

#******************************** Silhouette Score vs K *******************************************************

silhouette_values = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_ae)
    silhouette_values.append(silhouette_score(X_train_ae, labels))

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_values, marker='o')
plt.title("Silhouette Score vs K (AE latent)")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

#*************************** Hierarchical (Agglomerative) Clustering ****************************************

from sklearn.cluster import AgglomerativeClustering

print("\n--- Agglomerative Clustering (AE latent) ---")

agg = AgglomerativeClustering(n_clusters=n_classes, linkage='ward')
agg_labels = agg.fit_predict(X_train_ae)

agg_silhouette = silhouette_score(X_train_ae, agg_labels)
agg_purity = purity_score(y_train, agg_labels)

print("Silhouette:", agg_silhouette)
print("Purity:", agg_purity)

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_train_ae[:500], method='ward')  # sample to avoid memory issues

plt.figure(figsize=(12, 5))
plt.title("Dendrogram (Hierarchical Clustering)")
dendrogram(Z, truncate_mode='level', p=5)
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

#******************************* 2D Visualization of Clusters *******************************************************

from sklearn.decomposition import PCA

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train_ae)

plt.figure(figsize=(14, 6))

# K-Means
plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, s=10, cmap='tab20')
plt.title(f"K-Means Clustering (Purity: {kmeans_purity:.2f})")

# Hierarchical
plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=agg_labels, s=10, cmap='tab20')
plt.title(f"Agglomerative Clustering (Purity: {agg_purity:.2f})")

plt.tight_layout()
plt.show()

#******************************* Cluster Purity Visualization ************************************************

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# Define the expected labels to force the confusion matrix shape to (20, 20)
n_classes = len(np.unique(y_train))
unique_true_labels = np.arange(1, n_classes + 1) # Person 1 to 20
unique_cluster_labels = np.arange(n_classes)     # Cluster 0 to 19


# --- K-Means Heatmap Calculation ---
# Force the confusion matrix to use only the 20 known true labels (1-20) 
# and the 20 cluster labels (0-19) to ensure a 20x20 matrix.
raw_kmeans_cm = confusion_matrix(
    y_train, 
    kmeans_labels, 
    labels=unique_true_labels,  # True labels (1-20) for rows
    sample_weight=None          # Added to simplify troubleshooting
)

# Create a DataFrame for better visualization:
# We transpose (T) the CM so rows are Clusters (0-19) and columns are True Labels (Person 1-20)
cluster_composition = pd.DataFrame(
    raw_kmeans_cm.T,
    index=[f"Cluster {i}" for i in unique_cluster_labels],
    columns=[f"Person {i}" for i in unique_true_labels]
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    cluster_composition, 
    annot=True,           # show numbers in cells
    fmt="d",              # integer format
    cmap="YlGnBu",        # color map
    cbar_kws={'label': 'Number of Images'}
)
plt.title("K-Means Cluster Composition Heatmap")
plt.xlabel("True Labels (People)")
plt.ylabel("Clusters")
plt.tight_layout()
plt.show()

# --- Agglomerative Heatmap Calculation ---
# Force the confusion matrix to use only the 20 known true labels (1-20) 
# and the 20 cluster labels (0-19)
raw_agg_cm = confusion_matrix(
    y_train, 
    agg_labels,
    labels=unique_true_labels, # True labels (1-20) for rows
    sample_weight=None
)

# Create a DataFrame for better visualization (transposed)
cluster_composition_agg = pd.DataFrame(
    raw_agg_cm.T,
    index=[f"Cluster {i}" for i in unique_cluster_labels],
    columns=[f"Person {i}" for i in unique_true_labels]
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    cluster_composition_agg, 
    annot=True, 
    fmt="d", 
    cmap="YlGnBu",
    cbar_kws={'label': 'Number of Images'}
)
plt.title("Agglomerative Cluster Composition Heatmap")
plt.xlabel("True Labels (People)")
plt.ylabel("Clusters")
plt.tight_layout()
plt.show()
