from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

# this code trains the model on the placeholder data in the below array (too small)
""" 
# define training data (placeholder)
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# train the model
model = Word2Vec(sentences, min_count=1)
"""

# this block outputs an array of vector dimensions after training
""" 
# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

# access vector for one word
print(model['sentence'])

# save model
model.save('model.bin')

# load model
new_model = Word2Vec.load('model.bin')
print(new_model) 
"""

# shows graphic representation of resulting vectors after training
"""
# retrieve all vectors from the trained model
X = model[model.wv.vocab]

# create a 2D PCA model of the word vectors
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# plot the projection
pyplot.scatter(result[:, 0], result[:, 1])

# annotate the points on the graph with the actual words
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show() 
"""


# this was to load the stanford GLoVe embedding into a file
# must be run before using the GloVe model.
""" # using stanford's pre-trained GLoVe embedding
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B/glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file) """

# this block is for using the GloVe model to find associated words
from gensim.models import KeyedVectors
# load the GLoVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#calculate example - should be queen
result = model.most_similar(positive=['king', 'woman'], negative=['male'], topn=1)
print(result)