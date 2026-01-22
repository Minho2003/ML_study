import pytesseract
from PIL import Image
# import cv2

Image = Image.open('./image.png')

text = pytesseract.image_to_string(Image, lang='eng')
print(f"--------\n{text}")
text = pytesseract.image_to_string(Image, lang='kor')
print(f"--------\n{text}")
text = pytesseract.image_to_string(Image, lang='eng+kor')
print(f"--------\n{text}")