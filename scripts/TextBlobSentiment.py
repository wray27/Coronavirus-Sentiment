from textblob import TextBlob
from IPython.display import Markdown, display

class TextBlobSentiment:
    def __init__(self):
        pass

    def addSentimentToDataframe(self, df, tweetCol='tweet', sentimentCol='sentiment', display=True):
        tweets = df[tweetCol].tolist()

        #add empty sentiment column
        if(not sentimentCol in df):
            df["sentiment"] = ""

        for idx,tweet in enumerate(tweets):
            analysis = TextBlob(tweet)
            if(display):
                print(tweet)
                print(analysis.sentiment)
                if analysis.sentiment[0]>0:
                    self.printmd("Positive", color="green")
                elif analysis.sentiment[0]<0:
                    self.printmd("Negative", color="red")
                else:
                    self.printmd("No result", color="grey")
                    print("")

            df["sentiment"][idx] = analysis.sentiment[0]

        return df

    #pformat print result with colours
    def printmd(self, string, color=None):
        colorstr = "<span style='color:{}'>{}</span>".format(color, string)
        display(Markdown(colorstr))
