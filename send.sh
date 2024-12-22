#!/bin/bash

# Get the image file name from command line argument
image_file=$1

# Check if an image file name was provided
if [ -z "$image_file" ]; then
  echo "Error: Please provide an image file name as an argument."
  exit 1
fi

# Use a smaller buffer size for the pipe
stdbuf -oL python encode.py "$image_file" | curl -X POST -H "Content-Type: application/json" -d '{"image": "$(cat)"}' http://localhost:8080/predict
