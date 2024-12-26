#!/usr/bin/env python3

from flask import Flask, request, jsonify
from ocr import *
from model import *
from aqg import *
from PIL import Image
import io
import os
import cv2
import base64
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)

def decode_image(image_base64):
    """
    Decodes a Base64-encoded image and converts it to an OpenCV-compatible format.

    Args:
        image_base64 (str): Base64 string of the image.

    Returns:
        np.ndarray: Decoded image in OpenCV format (BGR).
    """
    try:
        # Strip the MIME type if included
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        # Decode Base64 to bytes
        image_bytes = base64.b64decode(image_base64)

        # Load the image as a NumPy array (OpenCV format)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Decoded image is None.")
        
        return image
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint to process an image and return questions.
    """
    try:
        data = request.get_json()
        image_base64 = data.get('image')

        if not isinstance(image_base64, str) or not image_base64.startswith("data:image"):
            return jsonify({'status': 'error', 'message': 'Invalid image format. Provide a Base64-encoded image.'}), 400

        # Decode the image to OpenCV format
        image = decode_image(image_base64)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image.'}), 400

        # Perform OCR
        text = do_ocr(image)

        # Handle text as a list or string
        if isinstance(text, list):
            text = " ".join(text)  # Join list into a single string
        elif not isinstance(text, str):
            return jsonify({'status': 'error', 'message': 'Unexpected OCR output format.'}), 500

        logging.info(f"Extracted text: {text}")

        if not text.strip():
            return jsonify({'status': 'error', 'message': 'No text extracted from the image.'}), 400

        # Process the text to generate keywords and questions
        kw_array = kw_extract(text)
        questions = [question for question in perform_aqg(kw_array, text)]

        return jsonify({'status': 'success', 'prediction': questions})

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while processing the image.'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
