import cv2
# Read the image
image = cv2.imread("11.jpeg")
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)
# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Filter contours based on area and aspect ratio
number_plate_contours = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    if aspect_ratio > 2.0 and aspect_ratio < 5.0 and area > 1000:
        number_plate_contours.append(contour)
# Import the Pytesseract library
import pytesseract
# Configure Pytesseract
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'  # Specify the path to the Tesseract executable
# Iterate over potential number plate contours
for contour in number_plate_contours:
    # Extract ROI (Region of Interest)
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = blurred[y:y+h, x:x+w]
    # Use Pytesseract to extract text from the ROI
   # Use Pytesseract to extract text from the ROI with adjusted configuration
    number_plate_text = pytesseract.image_to_string(roi, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng')
 
    # Print or store the extracted text
    print("Number Plate Text:", number_plate_text)

# Draw bounding boxes around number plates
for contour in number_plate_contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Number Plates Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

