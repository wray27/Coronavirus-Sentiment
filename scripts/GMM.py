from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GMM:
    def __init__(self):
        pass

    #generate panic prediction from column (tweets)
    def generateLabelsFromDataframe(self, df, column='tweet', plot=True, vocab=None):
        if(vocab):
            vocabulary = createVocabulary(vocab)
            bow = bagOfWords(df, column, vocab=vocabulary)
        else:
            bow = bagOfWords(df, column)


        panic = self.modelFromBow(bow, plot)
        return panic

    #bag of words from a column (tweets)
    def generateBagOfWords(self, df, column, vocab=None):
        if(vocab):
            vocabulary = createVocabulary(vocab)
            bow = bagOfWords(df, column, vocab=vocabulary)
        else:
            bow = bagOfWords(df, column)
        return bow




    def modelFromBow(self, bow, plot=True, pca_com=2):
        #normalise each axis for pca
        x = StandardScaler().fit_transform(bow)

        x = pca(x, no_components=pca_com)

        gmm = GaussianMixture(n_components=2).fit(x)
        labels = gmm.predict(x)
        #probability of each class (e.g. panic and non-panic)
        panic = gmm.predict_proba(x)

        if(plot):
            #plot predictions
            plt.scatter(x[:, 0], x[:, 1], c=labels, s=40, cmap='viridis')
            plt.show()

        return panic



def bagOfWords(df, col, vocab=None):
    #get tweets
    data = df[col].to_numpy()

    count = CountVectorizer(vocabulary=vocab)

    #bag of words with unlimited dictionary
    bag_of_words = count.fit_transform(data)
    bow = bag_of_words.toarray()

    return bow


def pca(x, no_components=3):
    pca = PCA(n_components=no_components)
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
