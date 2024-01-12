from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model("flower_classification_model.h5")

# Define a function to process the image
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    return img_array

# Define a function to classify the flower
def classify_flower(img_path):
    img_array = process_image(img_path)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    # For demonstration purposes, let's assume you have a list of class names
    classes=['bougainvillea', 'daisies', 'gardenias', 'gardenroses', 'hibiscus','hydrangeas','lilies','orchids','peonies','tulip']
    predicted_class = class_names[class_idx]

    return predicted_class

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part")
        
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file:
            # Save the uploaded file
            file_path = "uploads/" + file.filename
            file.save(file_path)

            # Classify the flower
            predicted_class = classify_flower(file_path)

            return render_template("result.html", image_path=file_path, predicted_class=predicted_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
