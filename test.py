import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from PIL import Image
from tensorflow.keras.applications import ResNet50
import faiss
from keras.applications.vgg19 import preprocess_input, VGG19

# Load the precomputed feature list and filenames
feature_list = np.array(pickle.load(open('artifacts/extracted_features/embedding.pkl','rb')))
filenames = pickle.load(open('artifacts/pickle_format_data/img_PICKLE_file.pkl','rb'))

base_model = VGG19(weights='imagenet', include_top=False)
base_model.trainable = False
model = tf.keras.Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])
# Face detection using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('download.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img, 1.3, 5)
x, y, w, h = faces[0]
face = img[y:y+h, x:x+w]

# Preprocess the detected face
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = np.asarray(image).astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()

# Normalize the feature vectors for cosine similarity
faiss.normalize_L2(feature_list)
faiss.normalize_L2(result.reshape(1, -1))

# Create a Faiss index and add the feature vectors
index = faiss.IndexFlatIP(feature_list.shape[1])  # Inner Product for cosine similarity
index.add(feature_list)

# Search for the most similar vectors
D, I = index.search(result.reshape(1, -1), k=1)  # k is the number of nearest neighbors to retrieve

# Get the index of the most similar image
index_pos = I[0][0]

# Recommend the most similar image
temp_img = cv2.imread(filenames[index_pos])
temp_img = cv2.resize(temp_img, (480, 480))
cv2.imshow('output', temp_img)
cv2.waitKey(0)
