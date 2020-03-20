from googletrans import Translator
from langdetect import detect
import re

class PandasDataCleaner:
    def __init__(self):
        pass

    #delete all non ascii characters
    def deEmojify(self, input):
        return input.encode('ascii', 'ignore').decode('ascii')

    #translate tweets- CHECK IF WORKING, JSONDecodeError because of request limits (I think)
    def translate(self, df, inputRow, outputRow):
        if(not outputRow in df):
            df[outputRow] = ""
        for i in range(df.shape[0]):
            translator = Translator()
            df[outputRow][i] = translator.translate(df[inputRow][i]).text
        return df

    def detectLanguage(self, df, inputRow, outputRow):
        if(not outputRow in df):
            df[outputRow] = ""
        for i in range(df.shape[0]):
            try:
                df[outputRow][i] = detect(df[inputRow][i])
            except:
                df[outputRow][i] = ''
        return df

    def cleanTweet(self, df, inputRow):
        df[inputRow] = df[inputRow].apply(lambda x: self.removeUrl(self.removeCharacters(x)))
        return df

    def removeUrl(self, input):
        return re.sub(r"http\S+", "", input)

    def removeWhitespace(self, input):
        return input.replace("\n", "")


    def removeCharacters(self, input):
        input = self.deEmojify(input)
        input = self.removeWhitespace(input)
        return input
