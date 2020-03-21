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


    #stopwords needs to be a set
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

    #plot count of unique attribute in dataframe (e.g. amount of each date)
    def plotCountOfUnique(self, df, attribute, xlabelrotation='vertical', xlabelsize='6', xlabel=None, ylabel=None):
        count = list(map(lambda x: x-1, list(df.groupby(attribute).count()['id'])))
        unique = list(df[attribute].unique())
        plt.plot(unique, count)
        plt.ylabel(ylabel)
        plt.xticks(unique, rotation=xlabelrotation)
        plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
        plt.xlabel(xlabel)

    def categoricalScatterPlot(self, df, category, datacat, rotation=0, height=5, fontsize= 10):
        plot = sns.catplot(x=category, y=datacat, data=df, height=height, aspect=11.7/8.27)
        plot.set_xticklabels(rotation=rotation, fontsize=fontsize)
