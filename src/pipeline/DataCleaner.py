import os
import argparse
import cv2
from src import logger
from src.utils.utils import create_directory, read_yaml

def detect_and_resize_faces(input_dir, output_dir, target_size=(224, 224)):
    create_directory([output_dir])

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y + h, x:x + w]
                resized_face = cv2.resize(face, target_size)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                cv2.imwrite(output_path, resized_face)

def DataClean(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    upload_image_dir = artifacts['upload_image_dir']
    clean_image_dir = artifacts['clean_image_dir']

    raw_local_dir_path = os.path.join(artifacts_dir, upload_image_dir)

    for subdir, dirs, files in os.walk(raw_local_dir_path):
        for directory in dirs:
            if "close and clear front face of" in directory:
                celeb_name = directory.split(" of ")[-1]
                celeb_dir_path = os.path.join(subdir, directory)
                clean_local_dir_path = os.path.join(artifacts_dir, clean_image_dir, celeb_name.replace(" ", "_"))

                detect_and_resize_faces(celeb_dir_path, clean_local_dir_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        DataClean(parsed_args.config, parsed_args.params)
        print("Cleaning and resizing images completed successfully.")
    except Exception as e:
        print(f"Error cleaning images: {str(e)}")

