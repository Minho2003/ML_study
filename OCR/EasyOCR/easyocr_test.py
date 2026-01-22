import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, Image, ImageDraw

image_path = './img_process/gray.jpg'

reader = easyocr.Reader(['en', 'ko'])
results = reader.readtext(image_path)

image = cv2.imread(image_path)

img = Image.fromarray(image)
font = ImageFont.truetype("AppleSDGothicNeo.ttc", 25)
draw = ImageDraw.Draw(img)

for i in results:
  x = i[0][0][0]
  y = i[0][0][1]
  w = i[0][1][0] - i[0][0][0]
  h = i[0][2][1] - i[0][1][1]

  draw.rectangle(((x, y), (x+w, y+h)), outline="blue", width=2)
  draw.text((int((x+x+w)/2), y-20), str(i[1]), font=font, fill="blue")

# plt.figure(figsize=(20,12))
plt.imshow(img)
plt.show()
print(results)