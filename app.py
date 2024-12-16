import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define upload and output folders and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load trained models
try:
    model = joblib.load("polynomial_classifier.pkl")
    pca = joblib.load("pca_transform.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Model files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")

# Function to preprocess image (resize, normalize, flatten)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))  # Resize to 64x64
    img_normalized = img_resized / 255.0  # Normalize
    img_flattened = img_normalized.flatten()  # Flatten
    return img, img_flattened

# Function to predict tumor type
def predict_tumor(image_path):
    original_img, img_flattened = preprocess_image(image_path)
    img_pca = pca.transform([img_flattened])  # Apply PCA transformation
    prediction = model.predict(img_pca)  # Predict tumor type
    tumor_type = label_encoder.inverse_transform(prediction)[0]
    return tumor_type, original_img

# Function to detect tumors (using contours) and return circle parameters
def detect_tumor(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the thresholded image (make tumors white on black background)
    thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tumor_positions = []
    tumor_sizes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small regions (noise)
            # Get the minimum enclosing circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            tumor_positions.append((int(x), int(y), int(radius)))
            tumor_sizes.append(int(area))

    return tumor_positions, tumor_sizes


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict tumor type
        tumor_type, original_img = predict_tumor(filepath)

        # Detect tumor positions and sizes
        tumor_positions, tumor_sizes = detect_tumor(filepath)

        # Draw circles if tumors are detected
        result_filename = f"result_{filename}"
        result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)

        image = Image.open(filepath).convert("RGB")
        draw = ImageDraw.Draw(image)

        if tumor_positions:
            for (x, y, radius) in tumor_positions:
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline="red",
                    width=3
                )

        image.save(result_image_path)

        # Pass data to the template
        return render_template(
            "result.html",
            tumor_type=tumor_type,
            tumor_positions=tumor_positions,
            tumor_sizes=tumor_sizes,
            original_filename=filename,
            result_filename=result_filename
        )

    return 'Invalid file type or no file uploaded'




if __name__ == '__main__':
    app.run(debug=True)
