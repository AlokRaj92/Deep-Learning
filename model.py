from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import os
import pickle

# Prompt the user for the image path
#img_path = r'D:\Deployment\CNN_image_recognition\Untitled.png'
img_path = r''
uploaded_image = Image.open(img_path)
# Check if the provided path exists
if not os.path.exists(img_path):
    print(f"Error: The file at {img_path} does not exist.")
else:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Load ResNet50 model with pre-trained weights
    model = ResNet50(weights='imagenet')

    # Make a prediction
    preds = model.predict(x)
    

import pickle
pickle.dump(model, open("model.pkl", "wb"))

import os
os.getcwd()