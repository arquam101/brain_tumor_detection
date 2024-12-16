Brain Tumor Detection using Image Processing and Machine Learning
This project provides a web-based application for brain tumor detection from MRI images. It utilizes machine learning models for tumor classification and U-Net segmentation to detect and highlight tumors within the MRI scans. The app is built using Flask, OpenCV, TensorFlow, and other Python libraries.

Features
Upload MRI images (in PNG, JPEG, or JPG format) through the web interface.
The system uses a polynomial classifier to predict the type of the tumor.
It applies U-Net segmentation to detect and highlight the tumor's location and size.
The result includes:
Tumor type classification (e.g., benign or malignant).
Image with a bounding box or circle around the tumor area.
Tumor size (calculated based on pixel dimensions).
Technologies Used
Flask: Web framework for the application.
OpenCV: Used for image processing and tumor detection.
TensorFlow/Keras: Used to load a pre-trained U-Net model for tumor segmentation.
Scikit-learn: For machine learning classification using a polynomial classifier.
Pillow (PIL): For image manipulation and visualization.
Project Structure
bash
Copy code
/brain-tumor-detection
│
├── app.py                  # Main Flask application
├── /static
│   ├── /uploads            # Folder for uploaded images
│   └── /outputs            # Folder for result images
├── /templates
│   ├── index.html          # Upload page
│   └── result.html         # Results page
├── /models
│   ├── polynomial_classifier.pkl  # Trained polynomial classifier
│   ├── pca_transform.pkl          # PCA transformation model
│   ├── label_encoder.pkl          # Label encoding for tumor types
│   └── unet_tumor_segmentation_model.h5  # Pre-trained U-Net model
└── requirements.txt         # List of required Python packages
Setup
Prerequisites
Python 3.6 or higher
TensorFlow (for U-Net model and ML models)
Flask (for the web server)
OpenCV (for image processing)
Scikit-learn (for machine learning models)
Pillow (for image handling)
Installation
Clone this repository:
bash
Copy code
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
Create and activate a virtual environment:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Make sure you have the trained models (e.g., polynomial_classifier.pkl, unet_tumor_segmentation_model.h5, pca_transform.pkl, label_encoder.pkl) placed in the /models directory.
Requirements
Create a requirements.txt with the following contents:
makefile
Copy code
Flask==2.0.2
opencv-python==4.5.3.56
tensorflow==2.6.0
scikit-learn==0.24.2
Pillow==8.4.0
matplotlib==3.4.3
werkzeug==2.0.1
Running the App
Start the Flask development server:
bash
Copy code
python app.py
Open your web browser and go to http://127.0.0.1:5000/ to access the application.
How It Works
Upload an Image: On the main page (index.html), upload an MRI scan image (in .jpg, .jpeg, or .png format).
Tumor Detection: After uploading, the system will:
Classify the tumor type using the polynomial classifier.
Perform segmentation using the U-Net model to locate the tumor.
Display Results: The results page (result.html) will show:
Tumor type.
Detected tumor(s) position and size.
The result image with the tumor area highlighted using a bounding box or circle.
Upload Another Image: After viewing the results, you can upload another image.
Expected Output
Tumor Type: The type of tumor detected (e.g., benign or malignant).
Tumor Position: The coordinates and size of the tumor on the MRI image.
Result Image: A copy of the uploaded image with the tumor area highlighted.
Troubleshooting
Error Loading Models: Ensure that the required models (polynomial_classifier.pkl, unet_tumor_segmentation_model.h5, pca_transform.pkl, label_encoder.pkl) are placed in the correct directory (/models).
Model Prediction Failures: Verify that the images you are uploading are in the correct format and have sufficient quality.
