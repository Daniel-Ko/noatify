# -*- coding: utf-8 -*-
from rake_nltk import Rake
from ocr import runOCR
import argparse
import io

def main(img_names):

    image_to_ocrtext = {}

    for img_name in img_names:
        string = runOCR(img_name)

        # Output OCR'd string is mapped to image
        image_to_ocrtext[img_name] = string

        r = Rake(max_length=2)

        
        r.extract_keywords_from_text(string)
        keywords = r.get_ranked_phrases()

        with io.open("keywords_output.txt", "a", encoding='utf8') as f:
            for keyword in keywords:
                f.write("{}\n".format(keyword.encode('utf-8').decode('utf-8')))

if __name__ == "__main__":
    import time
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', help="Image filename")
    args = parser.parse_args()
    
    main(args.images)

    end = time.time()
    print("[RUNTIME] {} sec".format(end - start))