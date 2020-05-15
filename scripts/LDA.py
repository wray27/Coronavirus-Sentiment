import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

# 1. Tokenisation
# 2. Words that have fewer than 3 characters are removed.
# 3. Remove stopwords
# 4. Lemmatise words
# 5. Stem words

class LDA:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.dictionary = None
        try:
#             nltk.download('wordnet')
            dler = nltk.downloader.Downloader()
            dler._update_index()
            dler.download('wordnet')
            
            print("Downloaded")
        except Exception as e:
            print("Error")
            print(e)

    def lemmatise_stemming(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatise_stemming(token))
        return result

    def dict_from_vocab(self, doc):
        dictionary = gensim.corpora.Dictionary(doc)
#         dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.dictionary = dictionary

    def tf_idf(self, bow_corpus):
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]

        return corpus_tfidf

    def train(self, bow_corpus, no_topics, alpha, passes=2, workers=2):
        return gensim.models.LdaMulticore(bow_corpus, num_topics=no_topics, id2word=self.dictionary, passes=passes, workers=workers, alpha=alpha)

    #generate panic prediction from column (tweets)

    def generate_labels_from_data_frame(self, df, column='tweet', plot=True, vocab=None):
        if(vocab):
            vocabulary = createVocabulary(vocab)
            bow = bagOfWords(df, column, vocab=vocabulary)
        else:
            bow = bagOfWords(df, column)

        panic = self.modelFromBow(bow, plot)
        return panic

    #bag of words from a column (tweets)
    def generateBagOfWords(self, docs):
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        return bow_corpus

    def modelFromBow(self, bow, plot=True):
        #normalise each axis for pca

        lda_model = gensim.models.LdaMulticore(
            bow, num_topics=10, id2word=dictionary, passes=2, workers=2)

        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))

        x = StandardScaler().fit_transform(bow)

        pc = pca(x)

        return panic


def bagOfWords(df, col, vocab=None):
    #get tweets
    data = df[col].to_numpy()

    count = CountVectorizer(vocabulary=vocab)

    #bag of words with unlimited dictionary
    bag_of_words = count.fit_transform(data)
    bow = bag_of_words.toarray()

    return bow


def pca(x):
    pca = PCA(n_components=3)
    pc = pca.fit_transform(x)
    return pc


#create vocabulary for bag of words
def createVocabulary(words):
    vocab = dict()
    count = 0
    for word in words:
        if(not word in vocab):
            vocab[word] = count
            count += 1
    return vocab
