import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


class Visualisation:
    def __init__(self):
        self.defaultStopwords = set(STOPWORDS)
        self.defaultStopwords.add("https")
        self.defaultStopwords.add("xa0")
        self.defaultStopwords.add("xa0'")
        self.defaultStopwords.add("bitly")
        self.defaultStopwords.add("bit")
        self.defaultStopwords.add("ly")
        self.defaultStopwords.add("twitter")
        self.defaultStopwords.add("pic")
        self.defaultStopwords.add("coronavirus")
        self.defaultStopwords.add("corona")
        self.defaultStopwords.add("virus")
        self.defaultStopwords.add("covid")
        self.defaultStopwords.add("19")
        self.defaultStopwords.add("covid19")
        self.defaultStopwords.add("coronavirus'")

    #plot distrubution of att
    def showDistribution(self, df, att):
        sns.distplot(df[att])
        sns.set(rc={'figure.figsize':(11.7,8.27)})


    #stopwords needs to be set
    def showWordCloud(self, df, attribute, default_stopwords=True, stopwords=set()):
        text = df[attribute].tolist()

        if(default_stopwords):
            stopwords = stopwords | self.defaultStopwords

        wordcloud = WordCloud(
            background_color = 'black',
            width = 1000,
            height = 500,
            stopwords = stopwords).generate(str(text))

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.rcParams['figure.figsize'] = [10, 10]
