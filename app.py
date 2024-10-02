from flask import Flask, request, render_template
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import os

# Load the trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html', result=None, message=None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('upload.html', result=None, message='No file uploaded!')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('upload.html', result=None, message='No file selected!')

    # Save the uploaded file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load and preprocess the image
    image = cv2.imread(file_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    # Make the prediction
    result = model.predict(input_img)
    predicted_class = np.argmax(result)

    # Clean up the uploaded file
    os.remove(file_path)

    # Determine result message
    if predicted_class == 1:
        message = 'Tumor detected!'
    else:
        message = 'No tumor detected!'

    return render_template('upload.html', result=predicted_class, message=message)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')  # Create the uploads directory if it doesn't exist
    app.run(debug=True)
