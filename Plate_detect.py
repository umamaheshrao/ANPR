# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 06:11:30 2024

@author: SANTOSH
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

harcascade = "model/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

@app.route('/')
def index():
    return render_template('number_plate_video.html')

@socketio.on('image')
def process_image(image_data):
    # Convert base64 image data to OpenCV format
    nparr = np.frombuffer(image_data.split(',')[1].decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    detected_plates = []
    for (x, y, w, h) in plates:
        area = w * h
        if area > 500:  # Adjust this threshold as needed
            detected_plates.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    # Emit detected plates back to the frontend
    emit('detected_plates', detected_plates)

if __name__ == '__main__':
    app.run(debug=True)
