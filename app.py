import os
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
import faiss
import streamlit as st

def create_directory(dirs):
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def save_uploaded_image(uploaded_image, path):
    try:
        create_directory([path])
        with open(os.path.join(path, uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def extract_features(img_path, models, face_cascade):
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        st.error("No face detected in the image.")
        return None
    
    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    feature_vector = []
    for model in models:
        features = model.predict(preprocessed_img)
        pooled_features = GlobalMaxPooling2D()(features)
        feature_vector.extend(pooled_features.numpy().flatten())
    
    return np.array(feature_vector)

def recommend(feature_list, features):
    similarity = [faiss.cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

def extract_name_from_path(path):
    return path.split(os.sep)[-2].split(' ')[-1]

st.set_page_config(page_title='Celebrity Look-Alike Finder', page_icon=':star:', layout='wide')

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e2e;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #03c4a1;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #ffffff;
        font-size: 1.5rem;
        text-align: center;
    }
    .upload-section {
        background-color: #252545;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .header {
        color: #03c4a1;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown('<div class="title">Which Celebrity Do You Look Like?</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your photo and find out your celebrity look-alike!</div>', unsafe_allow_html=True)

uploaded_image = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

feature_list = np.array(pickle.load(open('artifacts/extracted_features/embedding.pkl', 'rb')))
filenames = pickle.load(open('artifacts/pickle_format_data/img_PICKLE_file.pkl', 'rb'))

if len(feature_list) == 0:
    st.error("Feature list is empty. Ensure features are correctly extracted and saved.")
    st.stop()

base_model = VGG19(weights='imagenet', include_top=False)
layer_names = ['block1_conv1', 'block2_conv2', 'block3_conv3']
models = [tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(name).output) for name in layer_names]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploadn_path = 'artifacts/uploads'

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image, uploadn_path):
        display_image = Image.open(uploaded_image)
        features = extract_features(os.path.join(uploadn_path, uploaded_image.name), models, face_cascade)
        
        if features is not None:
            faiss.normalize_L2(feature_list)
            faiss.normalize_L2(features.reshape(1, -1))
            
            index = faiss.IndexFlatIP(feature_list.shape[1]) 
            index.add(feature_list)
            
            D, I = index.search(features.reshape(1, -1), k=1)
            
            index_pos = I[0][0]
            predicted_actor = extract_name_from_path(filenames[index_pos])
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="header">Your Uploaded Image</div>', unsafe_allow_html=True)
                st.image(display_image, width=360)
            with col2:
                st.markdown(f'<div class="header">You Look Like {predicted_actor}</div>', unsafe_allow_html=True)
                st.image(filenames[index_pos], width=360)
            st.markdown('</div>', unsafe_allow_html=True)
