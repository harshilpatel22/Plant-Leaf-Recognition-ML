from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)


UPLOAD_FOLDER = '/Users/harshilpatel/Desktop/plantrecognition/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model_path = '/Users/harshilpatel/Desktop/plantrecognition/model/plant-recognition-model.h5'  # Replace with your actual model path
model = load_model(model_path)



class_indices_path = '/Users/harshilpatel/Desktop/plantrecognition/model/class_indices.json'
with open(class_indices_path, 'r') as json_file:
    class_indices = json.load(json_file)

class_labels = [label for label, index in sorted(class_indices.items(), key=lambda item: item[1])]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']

        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_label = predict_image(file_path)
            return f"Predicted Plant: {predicted_label}"

    return '''
    <!doctype html>
    <title>Upload a Plant Image</title>
    <h1>Upload a Plant Image for Recognition</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def predict_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    predicted_label = class_labels[class_index]

    return predicted_label

if __name__ == '__main__':
    app.run(debug=True)
