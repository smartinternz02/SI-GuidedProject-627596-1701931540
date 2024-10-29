from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
import os
import numpy as np
import pandas as pd
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input as resnet_preprocess
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input, Lambda

# Initialize Flask App
app = Flask(__name__)

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.mkdir('uploads')

# Load Labels
labels_dataframe = pd.read_csv(r'D:\ai ml\ai2\labels.csv')
dog_breeds = sorted(list(set(labels_dataframe['breed'])))
n_classes = len(dog_breeds)
class_to_num = dict(zip(dog_breeds, range(n_classes)))

# Define model1
model1 = Sequential([
    InputLayer((3072,)),
    Dropout(0.7),
    Dense(120, activation='softmax')
])
model1.load_weights(r"D:\ai ml\ai2\predict.weights.h5")

# Define feature extraction model (model2)
input_shape = (331, 331, 3)
input_layer = Input(shape=input_shape)
preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
resnet50v2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_resnet)
preprocessor_densenet = Lambda(densenet_preprocess)(input_layer)
densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_densenet)
merge = concatenate([resnet50v2, densenet])
model2 = Model(inputs=input_layer, outputs=merge)
model2.load_weights(r"D:\ai ml\ai2\extd.weights.h5")

# Helper function to get breed from predicted code
def get_key(val):
    for key, value in class_to_num.items():
        if val == value:
            return key
    return "Unknown breed"

# Define Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return "No image uploaded", 400
    
    f = request.files['images']
    filepath = os.path.join('uploads', f.filename)
    f.save(filepath)
    
    # Process image and make predictions
    try:
        img = image.load_img(filepath, target_size=(331, 331))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        extracted_features = model2.predict(x)
        y_pred = model1.predict(extracted_features)
        pred_code = np.argmax(y_pred, axis=1)[0]  # Extract single prediction code
        predicted_dog_breed = get_key(pred_code)

        return f"The classified Dog breed is: {predicted_dog_breed}"
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)

