import cv2
import pytesseract
import re
from spellchecker import SpellChecker  # Make sure to install the library using pip install pyspellchecker

def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform denoising to remove noise
    denoised_image = cv2.fastNlMeansDenoising(binary_image, h=10)
    return denoised_image

def postprocess_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())  # Remove extra whitespaces
    
    # Spell checking using pyspellchecker
    spell = SpellChecker()
    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word is not None:  # Check if correction is not None
            corrected_text.append(corrected_word)
    
    return ' '.join(corrected_text)


# Load image
image = cv2.imread(r'C:\Users\NandaKishore\Downloads\text.png')
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe' 
# Preprocessing
processed_image = preprocess_image(image)

# Apply OCR using Pytesseract
text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')

# Post-processing
processed_text = postprocess_text(text)

print("Extracted Text:")
print(processed_text)
