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

## Dataset

This project uses the **Brain Tumor MRI Dataset** for training and testing. The dataset contains **MRI scans** of brain tumors with annotations, making it ideal for tumor segmentation and classification tasks. It includes a variety of tumor types (gliomas, meningiomas, etc.) and their different characteristics.

**Dataset Link**: (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Project Structure
![Screenshot 2024-12-16 235200](https://github.com/user-attachments/assets/97d75dcb-9d8e-4cf4-bf64-6869c42082ae)


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

