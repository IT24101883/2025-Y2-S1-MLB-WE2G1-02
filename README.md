# 2025-Y2-S1-MLB-WE2G1-02

# Vehicle Image Processing Pipeline

## Overview
This project implements an image processing pipeline for a vehicle dataset containing images of cars and motorcycles. The pipeline performs the following tasks to prepare the dataset for machine learning tasks such as classification:
1. **Cleaning**: Removes corrupted and blurry images, moving blurry ones to a separate folder.
2. **Resizing/Rescaling**: Resizes images to 640x640 while preserving the dataset's folder structure (Cars and Motorcycles).
3. **Normalization**: Normalizes pixel values to the [0,1] range, saving images as PNGs.
4. **Outlier Removal**: Removes noisy or mislabeled images using Isolation Forest on VGG16 features.
5. **Class Distribution Analysis**: Visualizes and saves class distribution to an Excel file.
6. **Dataset Balancing**: Balances the dataset using augmentation to ensure equal representation of classes.
7. **Feature Extraction for EDA**: Extracts PCA-reduced features using VGG16 for exploratory data analysis.

The pipeline is implemented in a single Python script designed to run in Google Colab, leveraging libraries like OpenCV, PyTorch, TensorFlow, and scikit-learn.

## Dataset Details
- **Location**: `/content/drive/MyDrive/Vehicles`
- **Structure**: The dataset is organized into two subfolders:
  - `Cars`: Contains images of cars.
  - `Motorcycles`: Contains images of motorcycles.
- **Image Formats**: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.gif`, `.tif`, `.tiff`.
- **Initial Size**: Approximately 1590 images (as indicated in the cleaning notebook).
- **Issues Addressed**:
  - Corrupted images (e.g., unreadable files).
  - Blurry images (Laplacian variance < 500).
  - Outliers (mislabeled or anomalous images).
  - Class imbalance (uneven number of images per class).
- **Output Directories**:
  - Cleaned: `/content/drive/MyDrive/cleaned_dataset` (includes `blurred` subfolder).
  - Resized: `/content/drive/MyDrive/resized_dataset` (640x640).
  - Normalized: `/content/drive/MyDrive/normalized_dataset` ([0,1] normalized PNGs).
  - No Outliers: `/content/drive/MyDrive/no_outliers_dataset`.
  - Balanced: `/content/drive/MyDrive/balanced_dataset`.
  - EDA Features: `/content/drive/MyDrive/eda_features.csv`.
  - Class Distribution: `/content/class_distribution.xlsx`.

## Group Member Roles
The project was developed by a team of contributors, each responsible for a specific component of the pipeline. The roles are inferred from the notebook filenames provided:

- **IT24101883**: Developed the cleaning module for detecting and removing corrupted and blurry images. Implemented Laplacian variance-based blur detection and moved blurry images to a separate folder.
- **IT24101557**: Implemented the resizing/rescaling module to resize images to a uniform size (corrected to 640x640 with preserved folder structure).
- **IT24101707**: Designed the normalization module to scale pixel values to [0,1] and save images as PNGs, ensuring compatibility with machine learning models.
- **IT24101561**: Created the outlier removal module using Isolation Forest on VGG16-extracted features to eliminate noisy or mislabeled images.
- **IT24101581**: Developed the class distribution analysis module, generating visualizations and an Excel summary of class counts.
- **IT24102015**: Implemented the dataset balancing module using augmentation to equalize class sizes, ensuring no bias in model training.
- **Team Coordinator (assumed)**: Integrated all modules into a cohesive pipeline, ensuring proper chaining of inputs/outputs and maintaining folder structure.

## How to Run the Code
### Prerequisites
- **Environment**: Google Colab with Google Drive access.
- **Dataset**: Ensure the dataset is stored at `/content/drive/MyDrive/Vehicles` with `Cars` and `Motorcycles` subfolders.
- **Dependencies**: The script installs required libraries automatically. Ensure a stable internet connection for package installation.

### Steps
1. **Open Google Colab**:
   - Create a new notebook in Google Colab.
   - Ensure your Google Drive is accessible (the script mounts it automatically).

2. **Copy the Pipeline Script**:
   - Paste the following Python script into a cell in the Colab notebook. The script is provided in the project documentation or can be found in the repository as `pipeline.py`.

   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Install dependencies
   !pip install opencv-python imutils torch torchvision scikit-learn pillow pandas matplotlib tqdm numpy scipy

   # Imports
   import os
   import shutil
   import numpy as np
   import pandas as pd
   import cv2
   from imutils import paths
   from PIL import Image
   import glob
   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   from torchvision import transforms, models
   from torchvision.datasets.folder import IMG_EXTENSIONS
   from sklearn.ensemble import IsolationForest
   from sklearn.decomposition import PCA
   from sklearn.metrics.pairwise import cosine_similarity
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   import matplotlib.pyplot as plt
   from tqdm import tqdm

   # Device setup
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Define paths
   ORIGINAL_DIR = '/content/drive/MyDrive/Vehicles'
   CLEANED_DIR = '/content/drive/MyDrive/cleaned_dataset'
   RESIZED_DIR = '/content/drive/MyDrive/resized_dataset'
   NORMALIZED_DIR = '/content/drive/MyDrive/normalized_dataset'
   NO_OUTLIERS_DIR = '/content/drive/MyDrive/no_outliers_dataset'
   BALANCED_DIR = '/content/drive/MyDrive/balanced_dataset'
   EDA_FEATURES_FILE = '/content/drive/MyDrive/eda_features.csv'

   # Create output directories
   for dir_path in [CLEANED_DIR, RESIZED_DIR, NORMALIZED_DIR, NO_OUTLIERS_DIR, BALANCED_DIR]:
       os.makedirs(dir_path, exist_ok=True)

   # Helper: Get image paths
   def get_image_paths(root_dir):
       image_paths = []
       for root, _, files in os.walk(root_dir):
           for f in files:
               if f.lower().endswith(IMG_EXTENSIONS):
                   image_paths.append(os.path.join(root, f))
       return sorted(image_paths)

   # Step 1: Clean corrupted/blurry images
   def clean_images(input_dir, output_dir):
       print("Step 1: Cleaning corrupted and blurry images...")
       for subdir in ['Cars', 'Motorcycles']:
           src = os.path.join(input_dir, subdir)
           dst = os.path.join(output_dir, subdir)
           if os.path.exists(src):
               shutil.copytree(src, dst, dirs_exist_ok=True)
       image_paths = get_image_paths(output_dir)
       print(f"Found {len(image_paths)} images.")
       corrupted_images = []
       for path in image_paths:
           try:
               img = cv2.imread(path)
               if img is None:
                   corrupted_images.append(path)
           except:
               corrupted_images.append(path)
       for path in corrupted_images:
           os.remove(path)
           print(f"Removed corrupted: {path}")
       def variance_of_laplacian(image_path):
           img = cv2.imread(image_path)
           if img is not None:
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               return cv2.Laplacian(gray, cv2.CV_64F).var()
           return 0
       blur_threshold = 500.0
       blurry_images = [path for path in image_paths if variance_of_laplacian(path) < blur_threshold]
       blurred_folder = os.path.join(output_dir, 'blurred')
       os.makedirs(blurred_folder, exist_ok=True)
       for path in blurry_images:
           filename = os.path.basename(path)
           shutil.move(path, os.path.join(blurred_folder, filename))
           print(f"Moved blurry: {path} to {blurred_folder}")
       print(f"Cleaned dataset saved to: {output_dir}")
       print(f"Total blurry: {len(blurry_images)}, Corrupted: {len(corrupted_images)}")

   clean_images(ORIGINAL_DIR, CLEANED_DIR)

   # Step 2: Resize to 640x640
   def resize_images(input_dir, output_dir, target_size=(640, 640)):
       print("Step 2: Resizing images to 640x640...")
       classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls)) and cls != 'blurred']
       for cls in classes:
           cls_input = os.path.join(input_dir, cls)
           cls_output = os.path.join(output_dir, cls)
           os.makedirs(cls_output, exist_ok=True)
           image_paths = get_image_paths(cls_input)
           for path in tqdm(image_paths, desc=f"Resizing {cls}"):
               try:
                   img = Image.open(path).convert('RGB')
                   img = img.resize(target_size, Image.LANCZOS)
                   base_name = os.path.basename(path)
                   img.save(os.path.join(cls_output, base_name))
               except Exception as e:
                   print(f"Error resizing {path}: {e}")
       print(f"Resized dataset saved to: {output_dir}")

   resize_images(CLEANED_DIR, RESIZED_DIR)

   # Step 3: Normalize to [0,1]
   def normalize_images(input_dir, output_dir):
       print("Step 3: Normalizing images to [0,1]...")
       transform = transforms.Compose([
           transforms.ToTensor()
       ])
       classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls))]
       for cls in classes:
           cls_input = os.path.join(input_dir, cls)
           cls_output = os.path.join(output_dir, cls)
           os.makedirs(cls_output, exist_ok=True)
           image_paths = get_image_paths(cls_input)
           for path in tqdm(image_paths, desc=f"Normalizing {cls}"):
               try:
                   img = Image.open(path).convert('RGB')
                   tensor = transform(img)
                   pil_img = transforms.ToPILImage()(tensor)
                   base_name, ext = os.path.splitext(os.path.basename(path))
                   new_name = f"{base_name}_normalized.png"
                   pil_img.save(os.path.join(cls_output, new_name))
               except Exception as e:
                   print(f"Error normalizing {path}: {e}")
       print(f"Normalized dataset saved to: {output_dir}")

   normalize_images(RESIZED_DIR, NORMALIZED_DIR)

   # Step 4: Remove outliers
   class ImageDataset(Dataset):
       def __init__(self, root_dir, transform=None):
           self.image_paths = get_image_paths(root_dir)
           self.transform = transform
       def __len__(self):
           return len(self.image_paths)
       def __getitem__(self, idx):
           path = self.image_paths[idx]
           img = Image.open(path).convert('RGB')
           if self.transform:
               img = self.transform(img)
           return img, path

   def remove_outliers(input_dir, output_dir):
       print("Step 4: Removing outliers...")
       model = models.vgg16(pretrained=True).features.to(device)
       model.eval()
       transform = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])
       dataset = ImageDataset(input_dir, transform=transform)
       loader = DataLoader(dataset, batch_size=32, shuffle=False)
       features = []
       paths = []
       with torch.no_grad():
           for imgs, batch_paths in tqdm(loader, desc="Extracting features"):
               imgs = imgs.to(device)
               feats = model(imgs).flatten(1).cpu().numpy()
               features.extend(feats)
               paths.extend(batch_paths)
       features = np.array(features)
       iso_forest = IsolationForest(contamination=0.05, random_state=42)
       outliers = iso_forest.fit_predict(features)
       for i, path in enumerate(paths):
           if outliers[i] == 1:
               rel_path = os.path.relpath(path, input_dir)
               dst_path = os.path.join(output_dir, rel_path)
               os.makedirs(os.path.dirname(dst_path), exist_ok=True)
               shutil.copy(path, dst_path)
       print(f"Dataset without outliers saved to: {output_dir}")

   remove_outliers(NORMALIZED_DIR, NO_OUTLIERS_DIR)

   # Step 5: Visualize class distribution
   def visualize_class_distribution(input_dir):
       print("Step 5: Visualizing class distribution...")
       class_counts = {}
       classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls))]
       for cls in classes:
           class_path = os.path.join(input_dir, cls)
           class_counts[cls] = len(get_image_paths(class_path))
       df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
       df.to_excel('/content/class_distribution.xlsx', index=False)
       plt.figure(figsize=(8, 5))
       plt.bar(class_counts.keys(), class_counts.values())
       plt.title('Class Distribution')
       plt.xlabel('Classes')
       plt.ylabel('Number of Images')
       plt.show()
       print("Class distribution saved to /content/class_distribution.xlsx")
       return class_counts

   class_counts = visualize_class_distribution(NO_OUTLIERS_DIR)

   # Step 6: Balance dataset
   def balance_dataset(input_dir, output_dir, class_counts):
       print("Step 6: Balancing dataset with augmentation...")
       max_count = max(class_counts.values())
       datagen = ImageDataGenerator(
           rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
           zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
       )
       classes = list(class_counts.keys())
       for cls in classes:
           cls_input = os.path.join(input_dir, cls)
           cls_output = os.path.join(output_dir, cls)
           os.makedirs(cls_output, exist_ok=True)
           for path in get_image_paths(cls_input):
               shutil.copy(path, cls_output)
           current_count = class_counts[cls]
           to_generate = max_count - current_count
           if to_generate > 0:
               image_paths = get_image_paths(cls_input)
               generated = 0
               while generated < to_generate:
                   for path in image_paths:
                       if generated >= to_generate:
                           break
                       img = np.expand_dims(np.array(Image.open(path)), 0)
                       for batch in datagen.flow(img, batch_size=1):
                           pil_img = Image.fromarray(np.uint8(batch[0]))
                           pil_img.save(os.path.join(cls_output, f"aug_{generated}_{os.path.basename(path)}"))
                           generated += 1
                           break
       print(f"Balanced dataset saved to: {output_dir}")

   balance_dataset(NO_OUTLIERS_DIR, BALANCED_DIR, class_counts)

   # Step 7: Extract features for EDA
   def extract_features_for_eda(input_dir, output_file):
       print("Step 7: Extracting features for EDA...")
       model = models.vgg16(pretrained=True).features.to(device)
       model.eval()
       transform = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])
       dataset = ImageDataset(input_dir, transform=transform)
       loader = DataLoader(dataset, batch_size=32, shuffle=False)
       features = []
       labels = []
       with torch.no_grad():
           for imgs, batch_paths in tqdm(loader, desc="Extracting VGG features"):
               imgs = imgs.to(device)
               feats = model(imgs).flatten(1).cpu().numpy()
               features.extend(feats)
               batch_labels = [os.path.relpath(p, input_dir).split(os.sep)[0] for p in batch_paths]
               labels.extend(batch_labels)
       features = np.array(features)
       pca = PCA(n_components=50)
       reduced_features = pca.fit_transform(features)
       df = pd.DataFrame(reduced_features)
       df['label'] = labels
       df.to_csv(output_file, index=False)
       plt.figure(figsize=(8, 5))
       plt.plot(np.cumsum(pca.explained_variance_ratio_))
       plt.title('PCA Explained Variance')
       plt.xlabel('Components')
       plt.ylabel('Cumulative Variance')
       plt.show()
       print(f"PCA features saved to: {output_file}")

   extract_features_for_eda(BALANCED_DIR, EDA_FEATURES_FILE)

   print("Pipeline complete! Final balanced dataset at:", BALANCED_DIR)
   print("EDA features at:", EDA_FEATURES_FILE)
   ```

3. **Run the Script**:
   - Execute the cell. The script will:
     - Mount Google Drive.
     - Install dependencies.
     - Process the dataset step-by-step, printing progress for each step.
     - Save intermediate and final outputs to the specified directories.
   - Expected runtime: Several hours for large datasets (e.g., 1590 images), depending on GPU availability and dataset size.

4. **Monitor Outputs**:
   - Check intermediate folders (`cleaned_dataset`, `resized_dataset`, etc.) in Google Drive.
   - Download `/content/class_distribution.xlsx` for class distribution analysis.
   - Use `/content/drive/MyDrive/eda_features.csv` for further EDA (e.g., in pandas or visualization tools).

5. **Troubleshooting**:
   - Ensure the dataset is at `/content/drive/MyDrive/Vehicles`.
   - If errors occur (e.g., path not found), verify folder structure and permissions.
   - For memory issues, reduce batch sizes in `DataLoader` (e.g., `batch_size=16` in outlier removal and EDA steps).

### Outputs
- **Cleaned Dataset**: `/content/drive/MyDrive/cleaned_dataset` (corrupted removed, blurry in `blurred` subfolder).
- **Resized Dataset**: `/content/drive/MyDrive/resized_dataset` (640x640, with `Cars` and `Motorcycles` subfolders).
- **Normalized Dataset**: `/content/drive/MyDrive/normalized_dataset` ([0,1] normalized PNGs, with subfolders).
- **No Outliers Dataset**: `/content/drive/MyDrive/no_outliers_dataset` (outliers removed, with subfolders).
- **Balanced Dataset**: `/content/drive/MyDrive/balanced_dataset` (augmented, equal class sizes, with subfolders).
- **Class Distribution**: `/content/class_distribution.xlsx` (Excel file with class counts).
- **EDA Features**: `/content/drive/MyDrive/eda_features.csv` (PCA-reduced features for analysis).

### Notes
- **Performance**: Use a Colab GPU for faster processing (especially for VGG16 feature extraction).
- **Customization**: Adjust parameters like `blur_threshold=500.0`, `contamination=0.05` (outlier removal), or augmentation settings as needed.
- **Storage**: Ensure sufficient Google Drive space for intermediate and final datasets.
- **Further Analysis**: Use `eda_features.csv` for clustering, visualization, or other EDA tasks in tools like pandas or scikit-learn.