import sys
import os
import os.path
import shutil
import ocr

import sklearn.pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

N_FEATURES = 10000
N_COMPONENTS = 10

def process(text_strings, filenames, true_k):
    print('Extracting features using a sparse vectoriser...');
    vectorizer = TfidfVectorizer(
        max_df=0.5, max_features=N_FEATURES,
        min_df=2, stop_words='english',
        use_idf=True
    )
    X = vectorizer.fit_transform(text_strings)

    print('Performing dimensionality reduction using latent semantic analysis...')
    svd = TruncatedSVD(N_COMPONENTS)
    normalizer = Normalizer(copy=False)
    lsa = sklearn.pipeline.make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()

    km = KMeans(
        n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
        verbose=True
    )

    print('Performing K-means clustering...')
    km.fit(X)

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()

    result = []
    for cluster_idx in range(true_k):
        print('Cluster #{0}: '.format(cluster_idx + 1))
        print('\tKey words: {0}'.format(' '.join(terms[ind] for ind in order_centroids[cluster_idx, :10])))
        cluster_fnames = [fname for j, fname in enumerate(filenames) if km.labels_[j] == cluster_idx]
        print('\tFiles: #{0}: {1}'.format(cluster_idx + 1, ', '.join(cluster_fnames)))
        result.append(cluster_fnames)

    return result

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_clusters = int(sys.argv[3])

    filenames = [fname for fname in os.listdir(input_dir) if fname.lower().endswith('.png')]
    text_strings = []
    print('Performing OCR...')
    for fname in filenames:
        print('\tProcessing {0}'.format(fname))
        text = ocr.runOCR(os.path.join(input_dir, fname))
        text_strings.append(text)

    result = process(text_strings, filenames, num_clusters)

    print('Copying files...')
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    for cluster_idx, cluster_fnames in enumerate(result):
        cluster_dir = os.path.join(output_dir, 'Cluster{0}'.format(cluster_idx + 1))
        try:
            os.mkdir(cluster_dir)
        except OSError:
            pass
        for fname in cluster_fnames:
            src_path = os.path.join(input_dir, fname)
            dst_path = os.path.join(cluster_dir, fname)
            shutil.copyfile(src_path, dst_path)
    print('Done.')

if __name__ == '__main__':
    main()
