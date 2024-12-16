import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(data_dir):
    images, labels = [], []
    for label in os.listdir(data_dir):  # Each folder represents a class
        path = os.path.join(data_dir, label)
        if os.path.isdir(path):  # Ensure it's a folder
            for img_file in os.listdir(path):
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64
                    images.append(img)
                    labels.append(label)  # Use folder name as label
    return np.array(images), np.array(labels)

# Path to the dataset folder
data_dir = r"C:\Users\arqua\tumourproject\dataset\Training"  # Update with your dataset path

# Load and preprocess data
print("Loading dataset...")
X, y = load_data(data_dir)
print(f"Dataset loaded: {X.shape[0]} samples.")

# Normalize and reshape the images
X = X / 255.0  # Scale pixel values
X = X.reshape(X.shape[0], -1)  # Flatten each image into a 1D array

# Dimensionality reduction with PCA
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=300)  # Reduce to 300 principal components
X = pca.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Build and train the polynomial classifier
print("Training the model...")
degree = 2  # Polynomial degree
model = make_pipeline(PolynomialFeatures(degree), LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs'))
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model and label encoder
print("Saving the model and label encoder...")
joblib.dump(model, "polynomial_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(pca, "pca_transform.pkl")
print("Model, PCA, and label encoder saved successfully.")
