import pytesseract
from PIL import Image
import cv2
from os import getpid, remove
import imgprocess

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


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

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
        f.write("============================\nIMAGE_STR\n============================\n")
        f.write(pytesseract.image_to_string(img).encode('utf-8').decode('utf-8'))
        f.write("\n\n============================\nBOUNDING BOXES\n============================\n")
        f.write(pytesseract.image_to_boxes(img).encode('utf-8').decode('utf-8'))
        f.write("\n\n============================\nIMAGE_DATA\n============================\n")
        f.write(pytesseract.image_to_data(img).encode('utf-8').decode('utf-8'))

        # print("============================\nSCRIPT INFO\n============================")
        # print(pytesseract.image_to_osd(img).encode('utf-8').decode('utf-8'))
    except IOError:
        print("file couldn't be opened")
    finally:
        img.close()
        remove(tempfname)

        cv2.imshow("Image", modimg)
        cv2.imshow("Output", gray)
        cv2.waitKey(0)