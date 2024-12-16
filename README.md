# Brain Tumor Detection using Image Processing and Machine Learning

This project provides a web-based application for **brain tumor detection** from **MRI images**. It utilizes machine learning models for tumor classification and **U-Net segmentation** to detect and highlight tumors within the MRI scans. The app is built using **Flask**, **OpenCV**, **TensorFlow**, and other Python libraries.

## Features

- Upload MRI images (in **PNG**, **JPEG**, or **JPG** format) through the web interface.
- The system uses a **polynomial classifier** to predict the **type** of the tumor.
- It applies **U-Net segmentation** to detect and highlight the tumor's **location** and **size**.
- The result includes:
  - Tumor type classification (e.g., benign or malignant).
  - Image with a **bounding box** or **circle** around the tumor area.
  - Tumor size (calculated based on pixel dimensions).

## Technologies Used

- **Flask**: Web framework for the application.
- **OpenCV**: Used for image processing and tumor detection.
- **TensorFlow/Keras**: Used to load a pre-trained **U-Net model** for tumor segmentation.
- **Scikit-learn**: For machine learning classification using a polynomial classifier.
- **Pillow (PIL)**: For image manipulation and visualization.

## Project Structure
/brain-tumor-detection │ ├── app.py # Main Flask application ├── /static │ ├── /uploads # Folder for uploaded images │ └── /outputs # Folder for result images ├── /templates │ ├── index.html # Upload page │ └── result.html # Results page ├── /models │ ├── polynomial_classifier.pkl # Trained polynomial classifier │ ├── pca_transform.pkl # PCA transformation model │ ├── label_encoder.pkl # Label encoding for tumor types │ └── unet_tumor_segmentation_model.h5 # Pre-trained U-Net model └── requirements.txt # List of required Python packages

## Setup

### Prerequisites

- Python 3.6 or higher
- **TensorFlow** (for U-Net model and ML models)
- **Flask** (for the web server)
- **OpenCV** (for image processing)
- **Scikit-learn** (for machine learning models)
- **Pillow** (for image handling)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

