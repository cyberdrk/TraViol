'''When utilizing Tesseract, Adrian Rosebrock,PhD, recommends:

Using as an input image with as high resolution and DPI as possible.
Applying thresholding to segment the text from the background.
Ensuring the foreground is as clearly segmented from the background as possible (i.e., no pixelations or character deformations).
Applying text skew correction to the input image to ensure the text is properly aligned
'''

from PIL import Image
import pytesseract
import argparse
import cv2
import numpy as np
import os

# constructing the argparse
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
#	help="type of preprocessing to be done")
args = vars(ap.parse_args)

img = cv2.imread(args["image"])
'''

img = cv2.imread('[830 122].png')

# Pre processing
gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)

# writing the grayscale image to the disk as a temporary file

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


# load the image as a PIL or Pillow image, apply OCR, and then delete

words = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(words)

cv2.imshow("Image", img)
cv2.imshow("Output", gray)
cv2.waitKey(0)