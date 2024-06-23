import os
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from src.utils.utils import read_yaml, create_directory

def detect_face(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]
    return face

def extractor(face_img, models):
    face_img = cv2.resize(face_img, (224, 224))
    img_array = image.img_to_array(face_img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    feature_vector = []
    for model in models:
        features = model.predict(preprocessed_img)
        pooled_features = GlobalMaxPooling2D()(features)
        feature_vector.extend(pooled_features.numpy().flatten())
    return np.array(feature_vector)

def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']

    img_pickle_file = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    filenames = pickle.load(open(img_pickle_file, 'rb'))

    base_model = VGG19(weights='imagenet', include_top=False)
    base_model.trainable = False
    layer_names = ['block1_conv1', 'block2_conv2', 'block3_conv3']
    models = [tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(name).output) for name in layer_names]

    feature_extraction_dir = artifacts['feature_extraction_dir']
    extracted_features_name = artifacts['extracted_features_name']
    feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
    create_directory(dirs=[feature_extraction_path])

    features_name = os.path.join(feature_extraction_path, extracted_features_name)
    features = []

    for file in tqdm(filenames):
        face_img = detect_face(file)
        if face_img is not None:
            features.append(extractor(face_img, models))
        else:
            print(f"No face detected in image {file}")

    pickle.dump(features, open(features_name, 'wb'))

if __name__ == '__main__':
    import argparse
    import logging

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_02 started")
        feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_02 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
