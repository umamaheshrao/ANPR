import cv2

cap=cv2.VideoCapture(0)
harcascade = "Haarcascades/haarcascade_russian_plate_number.xml"
cap.set(3, 640) # width
cap.set(4, 480) #height
min_area = 500
count = 0

def NPR_gen_frames():
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
