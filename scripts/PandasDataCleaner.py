from googletrans import Translator
from langdetect import detect
import re, sys

class PandasDataCleaner:
    def __init__(self):
        pass

    #delete all non ascii characters
    def removeEmojis(self, input):
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
        length = str(df.shape[0])
        for i in range(df.shape[0]):
            sys.stdout.write("\rDetecting language: " + str(i) + "/" + length)
            sys.stdout.flush()
            try:
                df[outputRow][i] = detect(df[inputRow][i])
            except:
                df[outputRow][i] = 'en'
        return df

    def cleanTweets(self, df, inputRow):
        df[inputRow] = df[inputRow].apply(lambda x: self.removeUrl(self.removeCharacters(x)))
        return df

    def removeUrl(self, input):
        return re.sub(r"http\S+", "", input)

    def removeWhitespace(self, input):
        return input.replace("\n", "")


    def removeCharacters(self, input):
        input = self.removeEmojis(input)
        input = self.removeWhitespace(input)
        return input

    #add new column in dataframe with general location
    def addGeneralLocation(self, df):
        if(not "genplace" in df):
            df["genplace"] = ""
        df["genplace"] = df['place'].apply(lambda x: self.getGenLocFromString(x))
        return df

    #get general location e.g. Bristol, England -> England
    def getGenLocFromString(self, string):
        splitStr = string.split(",")
        return splitStr[0].strip() if len(splitStr) == 1 else splitStr[1].strip()