from googletrans import Translator

class DataCleaner:
    def __init__(self):
        pass

    #delete all non ascii characters
    def deEmojify(self, df, row):
        df[row].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        return df

    #translate tweets- CHECK IF WORKING, JSONDecodeError because of request limits (I think)
    def translate(self, df, inputRow, outputRow):
        for i in range(df.shape[0]):
            translator = Translator()
            df[outputRow][i] = translator.translate(df[inputRow][i]).text
