from flask import *
import cv2
from random import randint

app=Flask(__name__)

dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
with open('classes.txt') as f:
    classes = f.read().strip().splitlines()
capture = cv2.VideoCapture(0)   
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)             
capture.set (cv2.CAP_PROP_FRAME_HEIGHT, 720)       
color_map = {}       

def gen_frames():
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


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)