import sys
import os
import os.path
import ocr

import sklearn.pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

N_FEATURES = 10000

def process(text_strings, filenames, true_k):
    print('Extracting features using a sparse vectoriser...');
    vectorizer = TfidfVectorizer(
        max_df=0.5, max_features=N_FEATURES,
        min_df=2, stop_words='english',
        use_idf=True
    )
    X = vectorizer.fit_transform(text_strings)

    km = KMeans(
        n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
        verbose=True
    )

    print('Performing K-means clustering...')
    km.fit(X)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for cluster_idx in range(true_k):
        print('Cluster #{0}: '.format(cluster_idx + 1))
        print('\tKey words: {0}'.format(' '.join(terms[ind] for ind in order_centroids[cluster_idx, :10])))
        print('\tFiles: #{0}: {1}'.format(cluster_idx + 1, ', '.join(
            fname for j, fname in enumerate(filenames) if km.labels_[j] == cluster_idx
        )))

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    filenames = [fname for fname in os.listdir(input_dir) if fname.lower().endswith('.png')]
    text_strings = []
    print('Performing OCR...')
    for fname in filenames:
        print('\tProcessing {0}'.format(fname))
        text = ocr.ocr(os.path.join(input_dir, fname))
        text_strings.append(text)
    process(text_strings, filenames, 5)

if __name__ == '__main__':
    main()
