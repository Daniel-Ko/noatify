import pytesseract
from PIL import Image
import cv2
from os import getpid, remove
import imgprocess

import os
import sys
import codecs
import argparse

"""TODO:
1: MORE PREPROCESSING ()
2: KNN ON TYPOS
3: WORD ASSOCIATION
"""

# -*- coding: utf-8 -*-

if sys.stdout.encoding != 'cp850':
  sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'cp850':
  sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')


parser = argparse.ArgumentParser()
parser.add_argument("image", help="Image filename")
args = parser.parse_args()


if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

img_name = args.image


modimg = cv2.imread(img_name)
gray = cv2.cvtColor(modimg, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

tempfname = "{}.png".format(getpid())
cv2.imwrite(tempfname, gray)

gray = imgprocess.process(gray)

with open("output.txt", "w", encoding='utf-8') as f:
    try:
        img = Image.open(tempfname)
        f.write(pytesseract.image_to_string(img).encode('utf-8').decode('utf-8'))
    except IOError as e:
        print("Error occurred during OCR")
    finally:
        img.close()
        remove(tempfname)
