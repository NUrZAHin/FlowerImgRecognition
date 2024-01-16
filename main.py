import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
from skimage import img_as_float
from skimage import io

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load the flower classification model
flower_model = load_model("flower_classification_model.h5")

# Load VGG16 model for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess images for VGG16
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image (adjust size as needed)
    img = cv2.resize(img, (224, 224))
    
    # Apply Local Binary Pattern (LBP) for feature extraction
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp = img_as_float(lbp) 
    # Flatten the LBP features
    features = lbp.flatten()

    return features

# Function to find top N most similar images
def find_similar_images(query_features, image_folder, top_n=10):
    similar_images = []
    print("Image Folder:", image_folder)

    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                print("Extracting features from image:", filepath)

                # Extract features from the current image
                image_features = extract_features(filepath)

                # Calculate cosine similarity
                similarity = cosine_similarity([query_features], [image_features])[0][0]

                # Classify the flower in the current image
                predicted_class = classify_flower(filepath)

                similar_images.append({
                    "filename": filename,
                    "similarity": similarity,
                    "predicted_class": predicted_class
                })
                

    # Sort by similarity in descending order
    similar_images.sort(key=lambda x: x["similarity"], reverse=True)

    return similar_images[:top_n]

def find_similar_images(query_features, image_folder, top_n=10):
    similar_images = []
    print("Image Folder:", image_folder)

    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                print("Extracting features from image:", filepath)

                # Extract features from the current image
                image_features = extract_features(filepath)

                # Calculate cosine similarity
                similarity = cosine_similarity([query_features], [image_features])[0][0]

                # Classify the flower in the current image
                predicted_class = filename

                filepath_convert=filepath.replace("\\", "/")

                similar_images.append({
                    "filename": filepath_convert,
                    "similarity": similarity,
                    "predicted_class": predicted_class
                })
                

    # Sort by similarity in descending order
    similar_images.sort(key=lambda x: x["similarity"], reverse=True)

    return similar_images[:top_n]

# Function to classify the flower in an image
def classify_flower(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = flower_model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    # Replace this with your actual class names
    class_names = ["bougainvillea", "daisies", "gardenias", "gardenroses", "hibiscus",
                   "hydrangeas", "lilies", "orchids", "peonies", "tulip"]

    predicted_class = class_names[class_idx]

    return predicted_class


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("uploads/", filename)

@app.route('/flowers/<subdirectory>/<filename>')
def flower_image(subdirectory, filename):
    print("subdirectory", subdirectory)
    print("filename", filename)
    return send_from_directory("flowers", f"{subdirectory}/{filename}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            # Save the uploaded file
            file_path = "./uploads/" + file.filename
            file.save(file_path)

            # Classify the flower
            predicted_class = classify_flower(file_path)

            # Extract features from the uploaded image
            query_features = extract_features(file_path)

            # Find similar images in the flower folder
            similar_images = find_similar_images(query_features, './flowers/')

            print("Predicted Class:", predicted_class)
            print("Similar Images:", similar_images)


            return jsonify({
                "image_path": file_path,
                "predicted_class": predicted_class,
                "similar_images": similar_images
            })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
