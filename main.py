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

flower_model = load_model("flower_classification_model.h5")

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

global shape_features

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_path):

    img = cv2.imread(image_path)
    # img = cv2.imread("./flowers/daisies/daisies_00002.jpg")

    # Resize the image to a fixed size
    img = cv2.resize(img, (224, 224))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    lbp_l = local_binary_pattern(l, P=8, R=1, method='uniform')
    lbp_a = local_binary_pattern(a, P=8, R=1, method='uniform')
    lbp_b = local_binary_pattern(b, P=8, R=1, method='uniform')

    lbp_l = img_as_float(lbp_l)
    lbp_a = img_as_float(lbp_a)
    lbp_b = img_as_float(lbp_b)

    features_l = lbp_l.flatten()
    features_a = lbp_a.flatten()
    features_b = lbp_b.flatten()

    color_features = np.concatenate((features_l, features_a, features_b))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_gray = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_gray = img_as_float(lbp_gray)

    shape_features = lbp_gray.flatten()

    combined_features = np.concatenate((color_features, shape_features))

    # # Plot histogram for color features
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(color_features, bins=50, color='blue', alpha=0.7)
    # plt.title('Color Features Histogram')
    # plt.xlabel('Feature Value')
    # plt.ylabel('Frequency')

    # # Plot histogram for shape features
    # plt.subplot(1, 2, 2)
    # plt.hist(shape_features, bins=50, color='green', alpha=0.7)
    # plt.title('Shape Features Histogram')
    # plt.xlabel('Feature Value')
    # plt.ylabel('Frequency')

    # print("Color Features Length:", len(color_features))
    # print("Shape Features Length:", len(shape_features))


    # plt.tight_layout()
    # plt.show()
        
    return combined_features, shape_features

def find_similar_images(query_features,shape_features, image_folder, top_n=10):
    similar_images = []
    print("Image Folder:", image_folder)

    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                print("Extracting features from image:", filepath)

                image_features , shape_features = extract_features(filepath)

                shape_similarity = cosine_similarity([query_features[:len(shape_features)]], [image_features[:len(shape_features)]])[0][0]

                color_similarity = cosine_similarity([query_features[len(shape_features):]], [image_features[len(shape_features):]])[0][0]

                predicted_class = filename

                filepath_convert = filepath.replace("\\", "/")

                similar_images.append({
                    "filename": filepath_convert,
                    "shape_similarity": shape_similarity,
                    "color_similarity": color_similarity,
                    "predicted_class": predicted_class
                })
                
    # Sort the similar images based on shape similarity
    similar_images.sort(key=lambda x: x["shape_similarity"], reverse=True)

    return similar_images[:top_n]

def classify_flower(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = flower_model.predict(img_array)
    class_idx = np.argmax(prediction)
    
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
            file_path = "./uploads/" + file.filename
            file.save(file_path)

            predicted_class = classify_flower(file_path)

            query_features , shape_features = extract_features(file_path)

            similar_images = find_similar_images(query_features,shape_features, './flowers/')

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
