# Celebrity Look-Alike Finder
Welcome to the Celebrity Look-Alike Finder project! This application leverages state-of-the-art deep learning techniques to find and recommend celebrities that closely resemble a user's uploaded photograph. The project is built using TensorFlow, OpenCV, FAISS, and Streamlit to create a seamless, interactive, and efficient user experience.

### Project Overview
Celebrity Look-Alike Finder aims to provide users with an engaging way to discover which celebrity they most resemble. By uploading a photo, users can see the closest matching celebrity from a pre-compiled database of celebrity images. The core functionalities of this project include:

1. Face Detection: Utilizing OpenCV's Haar cascades to detect and crop faces from uploaded images.
2. Feature Extraction: Using a pre-trained VGG19 model to extract high-dimensional feature vectors from detected faces.
3. Similarity Search: Implementing FAISS (Facebook AI Similarity Search) to perform efficient and scalable similarity searches.
4. Interactive Interface: Deploying the application using Streamlit to provide a user-friendly and interactive web interface.
### Key Features
1. Accurate Face Detection: Detects faces in uploaded images with high accuracy using OpenCV.
2. Multi-Layer Feature Extraction: Extracts detailed feature vectors from multiple layers of VGG19, capturing both basic and abstract facial features.
4. Efficient Similarity Search: Utilizes FAISS for fast and accurate retrieval of the most similar celebrity image.
5. Interactive Web Interface: Easy-to-use interface built with Streamlit, allowing users to upload images and view results in real-time.

### Getting Started
Clone the Repository:

``bash
Copy code
git clone https://github.com/yourusername/celebrity-look-alike-finder.git
cd celebrity-look-alike-finder``

Install Dependencies:

```bash
pip install -r requirements.txt
```

after you have to run 
```bash
python main.py
```
for Download images, clean, create pickle file of image and feature extraction.

Download Pre-trained Models: Ensure you have the necessary pre-trained VGG19 weights and Haar cascade files.

Run the Application:

```bash
streamlit run app.py
```
