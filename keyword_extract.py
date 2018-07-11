# -*- coding: utf-8 -*-
from rake_nltk import Rake
from ocr import runOCR
import argparse
import io

def main(img_name):
    string = runOCR(img_name)
    r = Rake()

    r.extract_keywords_from_text(string)
    keywords = r.get_ranked_phrases()

    with io.open("keywords_output.txt", "w", encoding='utf8') as f:
        for keyword in keywords:
            f.write("{}\n".format(keyword.encode('utf-8').decode('utf-8')))
            # f.write('\n')

if __name__ == "__main__":
    import time
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image filename")
    args = parser.parse_args()

    main(args.image)
    end = time.time()
    print(end - start)