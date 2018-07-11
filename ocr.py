# -*- coding: utf-8 -*-

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



if sys.stdout.encoding != 'cp850':
  sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'cp850':
  sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')



def runOCR(img_name):
    # Set tesseract path manually and the config
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    config = ('-l eng --oem 1 --psm 3')
    
    # Open image 
    modimg = cv2.imread(img_name)
    
    # Preprocess image
    # modimg = imgprocess.process(modimg)

    # Temporarily write image to file (remove in finally clause)
    tempfname = "{}.png".format(getpid())
    cv2.imwrite(tempfname, modimg)

    try:
        with Image.open(tempfname) as img:
            return pytesseract.image_to_string(img)
        # f.write("============================\nIMAGE_STR\n============================\n")
        # f.write(pytesseract.image_to_string(img).encode('utf-8').decode('utf-8'))
        # f.write("\n\n============================\nBOUNDING BOXES\n============================\n")
        # f.write(pytesseract.image_to_boxes(img).encode('utf-8').decode('utf-8'))
        # f.write("\n\n============================\nIMAGE_DATA\n============================\n")
        # f.write(pytesseract.image_to_data(img).encode('utf-8').decode('utf-8'))

        # print("============================\nSCRIPT INFO\n============================")
        # print(pytesseract.image_to_osd(img).encode('utf-8').decode('utf-8'))
    except IOError as e:
        print("file couldn't be opened")
    finally:
        img.close()
        remove(tempfname)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', help="Image filename")
    args = parser.parse_args()

    runOCR(args.images)
