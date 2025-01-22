# Plant Leaf Recognition using MobileNetV2

This project aims to recognize and classify plant leaves into different categories using a deep learning model based on **MobileNetV2**. 
The model is trained on a dataset of plant leaf images, and a web application is built using **Flask** to deploy the model for real-time image 
classification.

---

## Project Overview

The Plant Leaf Recognition project leverages **MobileNetV2**, a pre-trained deep learning model, and fine-tunes it to classify various plant leaves. 
The project includes:
- **Model training**: The model is trained to recognize different plant species using a dataset of plant leaves.
- **Web application**: A Flask-based application is built to serve the model and allow users to upload images of plant leaves for real-time classification.

---

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
   git clone https://github.com/yourusername/Plant-Leaf-Recognition-ML.git
   cd Plant-Leaf-Recognition-ML
2. **Install required dependencies**

---

## Usage

**Training the Model**:

1. Dataset preparation:

Organize the dataset into separate folders for each class (plant species).
Use the splitfolders Python package to split the dataset into training, validation, and test sets:
import splitfolders
input_folder = '/path/to/raw/data'
output_folder = '/path/to/split/data'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

2. Model Training:

The model uses MobileNetV2 with transfer learning. The base layers are frozen, and a few dense layers are added for plant leaf classification.
To start training the model, run the notebook plant-leaf-classification.ipynb. It will:
Load the data using ImageDataGenerator for image augmentation.
Train the model for 10 epochs.
Save the trained model and class indices.
After training, the model is saved as plant-recognition-model.h5.

3. Deploying the model:

   Start the Flask Application:

    The app.py file serves as the backend for the Flask application.
    The app allows users to upload plant leaf images, which are then passed to the trained model for classification.
    Run the Flask app with the following command:
       python app.py

---

## Project Structure

Project Structure
plant-leaf-classification.ipynb: Jupyter notebook for training the model.
app.py: Flask application for serving the trained model and making predictions.
model/plant-recognition-model.h5: The trained model.
model/class_indices.json: JSON file containing the mapping of class indices to labels.
uploads/: Directory to store uploaded plant images for prediction.


