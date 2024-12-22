#!/usr/bin/env python3
#
from PIL import Image
from matplotlib import pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline

from model import *
import cv2
import numpy as np
import torch
import re

trocr_proc = trocr_processor()
trocr_mod = trocr_model()
syntax_mod = syntax_model()
syntax_tok = syntax_tokenizer()
kw_mod = kw_model()

def do_ocr(image):
  #image convolution
  rotated_image = image #change if ever rotation is necessary
  gray_image = grayscale(rotated_image)
  inverted_image = binary_inversion(gray_image)
  denoised_image = noise_removal(inverted_image)
  blurred_image = blur_image(denoised_image)
  thresholded_image = thresholding(blurred_image)
  dilated_image = vertical_dilate(thresholded_image, 1)

  #major text block identification
  rotated_cpy = image.copy()
  thresholded_cpy = thresholded_image.copy()
  cnts = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h > 500 and w > 500:
      text_block_image = rotated_cpy[y:y+h, x:x+w]
      thresholded_block = thresholded_cpy[y:y+h, x:x+w]
      cv2.rectangle(rotated_cpy, (x, y), (x + w, y + h), (36,255,12), 2)
      break
  #line identification and isolation
  horizontal_dilated = horizontal_dilate(thresholded_block, 1)
  block_cpy = text_block_image.copy()
  words = []
  cnts = cv2.findContours(horizontal_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])  # Sort by y-coordinate

  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 10 and h > 10:
      words.append(text_block_image.copy()[y:y+h, x:x+w])
      cv2.rectangle(block_cpy, (x, y), (x + w, y + h), (36, 255, 12), 2)

  generated_text = []

  for w in words:
    text = perform_ocr(Image.fromarray(w))
    generated_text.append(text)

  separator = " "
  joined = separator.join(generated_text)
  corrected = filter_text(joined)
  final = correct_grammar(corrected, 1)

  return final

def kw_extract(text):
    arr = kw_mod.extract_keywords(text, stop_words=None)
    keyword_array = [keyword for keyword, score in arr]
    return keyword_array


def correct_grammar(input_text,num_return_sequences):
  batch = syntax_tok([input_text],truncation=True,padding='max_length', max_length=512, return_tensors="pt").to(torch)
  translated = syntax_mod.generate(**batch,max_length=512,num_beams=4, num_return_sequences=num_return_sequences, do_sample=True, temperature=0.9)
  tgt_text = syntax_tok.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def filter_text(generated_text):
  text = generated_text
  text = re.sub(r' +([.,!?;:])', r'\1', text)
  text = re.sub(r'([,!?;:])(?=[^\s])', r'\1 ', text)
  text = re.sub(r'\.([a-zA-Z])', r'. \1', text)
  text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
  return text

def perform_ocr(image):
    """
    Performs OCR on a given image using the loaded TrOCR model.

    Args:
        image: PIL Image object.

    Returns:
        generated_text: The recognized text from the image.
    """
    pixel_values = trocr_proc(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = trocr_mod.generate(pixel_values)
    generated_text = trocr_proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

#preprocessing functions
def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
  import numpy as np
  kernel = np.ones((1,1), np.uint8)
  image = cv2.dilate(image, kernel, iterations=2)
  kernel = np.ones((1,1), np.uint8)
  image = cv2.erode(image, kernel, iterations=2)
  image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  image = cv2.medianBlur(image, 5)
  return (image)

def blur_image(image):
  return cv2.GaussianBlur(image, (5,5), 0)

def binary_inversion(image):
  inverted_image = cv2.bitwise_not(image)
  return (inverted_image)

def thresholding(image):
  return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def thin_font(image):
  import numpy as np
  image = cv2.bitwise_not(image)
  kernel = np.ones((2,2), np.uint8)
  image = cv2.erode(image, kernel, iterations=1)
  image = cv2.bitwise_not(image)
  return (image)

def thick_font(image):
  import numpy as np
  image = cv2.bitwise_not(image)
  kernel = np.ones((2,2), np.uint8)
  image = cv2.dilate(image, kernel, iterations=1)
  image = cv2.bitwise_not(image)
  return (image)

def vertical_dilate(image, iters):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,200))
  dilate = cv2.dilate(image, kernel, iterations=1)
  return (dilate)

def horizontal_dilate(image, iters):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,5))
  dilate = cv2.dilate(image, kernel, iterations=iters)
  return (dilate)

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

    # Rotate the image around its center

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)
