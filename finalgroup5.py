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

# Assuming you already trained AutoEncoder1 and have the encoder
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

# Build K-Means table
df_kmeans = pd.DataFrame({
    "Actual": y_train,
    "Cluster": kmeans_labels
})

cluster_composition = (
    df_kmeans.groupby(["Cluster", "Actual"])
    .size()
    .unstack(fill_value=0)
)

# Build Agglomerative table
df_agg = pd.DataFrame({
    "Actual": y_train,
    "Cluster": agg_labels
})

cluster_composition_agg = (
    df_agg.groupby(["Cluster", "Actual"])
    .size()
    .unstack(fill_value=0)
)

# K-Means Heatmap 
plt.figure(figsize=(12, 6))
sns.heatmap(
    cluster_composition,
    annot=True,           # show numbers in cells
    fmt="d",             # integer format
    cmap="YlGnBu",       # color map
    cbar_kws={'label': 'Number of Images'}
)
plt.title("K-Means Cluster Composition Heatmap")
plt.xlabel("True Labels (People)")
plt.ylabel("Clusters")
plt.tight_layout()
plt.show()

# Agglomerative Heatmap 
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

#******************************** Additional Code Snippets ***********************************************

#****************************************** CNN Classifier *********************************************

print("\n" + "="*50)
print("CNN CLASSIFIER")
print("="*50)

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

#******************************** CNN Model Architecture ***************************************

model_cnn = models.Sequential([
    layers.Input(shape=(img_h, img_w, 1)),
    
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten
    layers.Flatten(),
    
    # Dense Layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularization
    
    # Output Layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_cnn.summary()

#******************************** CNN Training with Early Stopping ************************************************

early_stopping_cnn = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\nStarting CNN training...")
history_cnn = model_cnn.fit(
    X_train_cnn, y_train_cat,
    epochs=50, # CNNs might converge faster or slower, 50 is a good start
    batch_size=32,
    validation_data=(X_val_cnn, y_val_cat),
    callbacks=[early_stopping_cnn],
    verbose=1
)

# 5. Evaluation & Visualization

# Plot Training History
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Val Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on Test Set
print("\nEvaluating CNN on Test Set...")
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"CNN Test Accuracy: {test_acc_cnn:.4f}")

# Detailed Metrics
y_pred_probs_cnn = model_cnn.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)
y_true_cnn = np.argmax(y_test_cat, axis=1)

precision_cnn, recall_cnn, f1_cnn, _ = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, average='weighted')
print(f"CNN Precision (Weighted): {precision_cnn:.4f}")
print(f"CNN Recall (Weighted):    {recall_cnn:.4f}")
print(f"CNN F1-Score (Weighted):  {f1_cnn:.4f}")

# Sample Predictions Visualization
# We want to show cases where True == Pred (as requested by user "true values should match predicted values")
# But we should show a mix or just random. The user said "at last..the true values should match predictided values"
# which implies they want to see successful classifications.
# I will filter for correct predictions to display.

correct_indices = np.where(y_pred_cnn == y_true_cnn)[0]
if len(correct_indices) >= 5:
    indices = np.random.choice(correct_indices, size=5, replace=False)
else:
    indices = correct_indices # Show all if fewer than 5

plt.figure(figsize=(15, 4))
plt.suptitle("CNN Sample Correct Predictions", fontsize=14)

for i, idx in enumerate(indices):
    ax = plt.subplot(1, 5, i + 1)
    
    # Image for display (reshape to 2D)
    img = X_test_cnn[idx].reshape(img_h, img_w)
    
    true_label = y_true_cnn[idx] + 1
    pred_label = y_pred_cnn[idx] + 1
    
    plt.imshow(img, cmap='gray')
    
    color = 'green' # Since we filtered for correct, it will always be green
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()

#************* Image Recognition using Supervised Learning – Neural Network Classifier ************************

# Feature Engineering (Preparing Cluster Features)
print("Generating cluster distance features for all subsets...")

# 1. Project all data into the Autoencoder Latent Space
lat_train = encoder.predict(X_train_scaled, verbose=0)
lat_val   = encoder.predict(X_val_scaled, verbose=0)
lat_test  = encoder.predict(X_test_scaled, verbose=0)

# 2. Get distances to all cluster centroids (fit in Part 4)
# Shape will be (n_samples, n_clusters)
dist_train = kmeans.transform(lat_train)
dist_val   = kmeans.transform(lat_val)
dist_test  = kmeans.transform(lat_test)

# Normalize these distances so they play nice with the Neural Network
# We use MinMaxScaler because distances are positive
dist_scaler = MinMaxScaler()
dist_train_norm = dist_scaler.fit_transform(dist_train)
dist_val_norm   = dist_scaler.transform(dist_val)
dist_test_norm  = dist_scaler.transform(dist_test)

print(f"Cluster Feature Shape (Train): {dist_train_norm.shape}")

# Data Formatting for CNN

# Reshape images for CNN (Height, Width, Channel)
X_train_img = X_train_scaled.reshape(-1, img_h, img_w, 1)
X_val_img   = X_val_scaled.reshape(-1, img_h, img_w, 1)
X_test_img  = X_test_scaled.reshape(-1, img_h, img_w, 1)

# One-hot encode targets
num_classes = len(np.unique(y_train))
y_train_cat = keras.utils.to_categorical(y_train - 1, num_classes)
y_val_cat   = keras.utils.to_categorical(y_val - 1, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test - 1, num_classes)

#********************************* Build the model architecture ***********************************

# Input 1: The Image
input_img = layers.Input(shape=(img_h, img_w, 1), name='image_input')
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
img_features = layers.Dense(128, activation='relu')(x)

# Input 2: The Cluster Distances (Metadata)
input_dist = layers.Input(shape=(dist_train_norm.shape[1],), name='cluster_input')
dist_features = layers.Dense(32, activation='relu')(input_dist)

# Concatenate both branches
combined = layers.concatenate([img_features, dist_features])

# Final Classification Layers
z = layers.Dense(128, activation='relu')(combined)
z = layers.Dropout(0.5)(z) # Regularization to prevent overfitting
output = layers.Dense(num_classes, activation='softmax')(z)

# Create Model
model = models.Model(inputs=[input_img, input_dist], outputs=output)

# Optimizer: Adam is standard for its adaptive learning rate
# Loss: Categorical Crossentropy because targets are one-hot encoded
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nHybrid CNN Architecture:")
model.summary()

#*************************************** Training ************************************************

# Early Stopping to prevent overfitting (stops if val_loss doesn't improve)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True,
    verbose=1
)

print("\nStarting Training...")
history = model.fit(
    x={'image_input': X_train_img, 'cluster_input': dist_train_norm},
    y=y_train_cat,
    epochs=40,
    batch_size=32,
    validation_data=(
        {'image_input': X_val_img, 'cluster_input': dist_val_norm},
        y_val_cat
    ),
    callbacks=[early_stop]
)

#*************************************** Evaluation & Metrics ************************************************

# 1. Plotting Curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Test Set Metrics
print("\nTest Set Evaluation:")
loss, acc = model.evaluate(
    {'image_input': X_test_img, 'cluster_input': dist_test_norm}, 
    y_test_cat, 
    verbose=0
)
print(f"Final Test Accuracy: {acc*100:.2f}%")

# Generate Predictions
y_pred_probs = model.predict({'image_input': X_test_img, 'cluster_input': dist_test_norm})
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Precision, Recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

#*************************************** Visualizing Sample Predictions ************************************************

# Select random samples
indices = np.random.choice(len(X_test), size=5, replace=False)

plt.figure(figsize=(15, 4))
plt.suptitle("Model Predictions (Image + Cluster Info)\n\n", fontsize=14)

for i, idx in enumerate(indices):
    ax = plt.subplot(1, 5, i + 1)
    
    # Grab original image for display
    img_display = X_test_img[idx].reshape(img_h, img_w)
    
    # Get labels (adding 1 because we subtracted 1 for one-hot encoding earlier)
    true_lbl = y_true[idx] + 1
    pred_lbl = y_pred[idx] + 1
    
    # Color code: Green if correct, Red if wrong
    color = 'green' if true_lbl == pred_lbl else 'red'
    
    plt.imshow(img_display, cmap='gray')
    plt.title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()


