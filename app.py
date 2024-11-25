from flask import *
import cv2
import pytesseract
import re
from random import randint
from spellchecker import SpellChecker
from ultralytics import YOLO
from functools import wraps

   #http://192.168.10.6:8080
   #https://192.168.10.7:8080 


app=Flask(__name__)



def redirect_to_https(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.scheme != 'https' and not app.debug:
            return redirect(request.url.replace('http://', 'https://'), code=301)
        return f(*args, **kwargs)
    return decorated_function
def preprocess_image(image):
    # Implement your preprocessing steps here
    # For example, you might want to convert the image to grayscale and apply some filters
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

def postprocess_text(text):
    # Implement your postprocessing steps here
    # For example, you might want to remove extra whitespace and special characters
    cleaned_text = text.strip()  # Remove leading and trailing whitespace
    cleaned_text = cleaned_text.replace('\n', ' ')  # Replace newline characters with spaces
    return cleaned_text

def Obj_gen_frames():
    dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    model = cv2.dnn_DetectionModel(dnn)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    with open('classes.txt') as f:
        classes = f.read().strip().splitlines()
    capture = cv2.VideoCapture('https://10.184.80.253:8080/video')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    color_map = {}
    while True:
        success, img = capture.read()  # read the camera frame
        frame = img.copy()  # Make a copy of the original frame
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        if not success:
            break
        else:
            try:
                class_ids, confidences, boxes = model.detect(frame)
                for id, confidence, box in zip(class_ids, confidences, boxes):
                    x, y, w, h = box
                    obj_class = classes[id]

                    if obj_class not in color_map:
                        color = (randint(0, 255), randint(0, 255), randint(0, 255))
                        color_map[obj_class] = color
                    else:
                        color = color_map[obj_class]

                    cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            except Exception as e:
                print("Error during detection:", e)

            cv2.imshow('Video Capture', frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            



def NPR_gen_frames():
    cap=cv2.VideoCapture('https://10.184.80.253:8080/video')
    harcascade = "Haarcascades/haarcascade_russian_plate_number.xml"
    cap.set(3, 640) # width
    cap.set(4, 480) #height
    min_area = 500
    count = 0
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            for (x,y,w,h) in plates:
                area = w * h

                if area > min_area:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    img_roi = img[y: y+h, x:x+w]
                    cv2.imshow("ROI", img_roi)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def prediction1(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_plate_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if aspect_ratio > 2.0 and aspect_ratio < 5.0 and area > 1000:
            number_plate_contours.append(contour)
    pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
   # pytesseract.pytesseract.tesseract_cmd = r'tesseract'
    for contour in number_plate_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = blurred[y:y+h, x:x+w]
        number_plate_text = pytesseract.image_to_string(roi, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng')
        return number_plate_text
    return ''

def solve(img):
    model = YOLO('yolov8x.pt')
    results = model(img, show=False)
    if isinstance(results, list):
        for i, img_result in enumerate(results):
            output_path = f"static\processed_image_{i}.jpg"
            img_result.save(output_path)
    print(output_path)
    return output_path

@app.route('/')
def hello():
    return render_template('main.html')

@app.route('/NPR')
def NPR():
    return render_template('number_plate.html')

@app.route('/NPRImage')
def NPRImage():
    return render_template('number_plate_image.html')

@app.route('/TextImage')
def TextImage():
    return render_template('text_extraction.html')

@app.route('/predict1',methods=["GET","POST"])
def predict1():
    file=request.files['file']
    file_path = r"static/Storage" + file.filename
    file.save(file_path)  
    k=prediction1(file_path)
    return render_template('number_plate_image.html',ans=k)

@app.route('/NPRVideoLoad')
def NPRVideoLoad():
    return render_template('number_plate_video.html')

@app.route('/NPRVideo')
def NPRVideo():
    return Response(NPR_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/textdetect',methods=["GET","POST"])
def textdetect():
     file=request.files['file']
     file_path = r"static\Storage" + file.filename
     file.save(file_path)
     image = cv2.imread(file_path)
     pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
     processed_image = preprocess_image(image)

     text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')

     processed_text = postprocess_text(text)

     k = processed_text
     return render_template('TextDetection.html',ans=k)

@app.route('/object')
def object():
    return render_template('object_detection.html')

@app.route('/ObjVideo')
def ObjVideo():
    return Response(Obj_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ObjVideoLoad')
def ObjVideoLoad():
    return render_template('object_detection_video.html')

@app.route('/ObjImage')
def ObjImage():
    return render_template('object_detection_image.html')

@app.route('/predict2',methods=["GET","POST"])
def predict2():
    file=request.files['file']
    file_path = r"static\Storage" + file.filename
    file.save(file_path)  
    k=solve(file_path)
    return render_template('object_detection_image.html',file=k[7:])

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,ssl_context='adhoc')
    
    