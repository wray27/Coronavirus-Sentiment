import csv, os
from TwintScraper import TwintScraper
import numpy as np
import pandas as pd
from datetime import date, timedelta
import asyncio



class TwintDataMiner:
    def __init__(self):
        #place in list of outputted CSV file from TwintScraper
        self.twintCSVDict = {
            "id":0,
            "conversation_id":1,
            "created_at":2,
            "date":3,
            "time":4,
            "timezone":5,
            "user_id":6,
            "username":7,
            "name":8,
            "place":9,
            "tweet":10
            #more can be added- see first line of csv
        }

    #use twint scraper to get tweets, use kwargs to define arguments (specified in TwintScraper)
    def scrape(self,**kwargs):
        ts = TwintScraper()
        ts.scrape(**kwargs)


    #get specific attribute of the tweets e.g. date
    def getAttribute(self, info, file):
        outputList = []
        with open(file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for idx,line in enumerate(reader):
                outputList.append(line[self.twintCSVDict[info]])
        return outputList


    #turn csv file into pandas dataframe with columns specified in attributes
    def getPandasDataFrame(self, attributes, file):
        pandasDict = dict()
        for att in attributes:
            pandasDict[att] = self.getAttribute(att, file)
        df = pd.DataFrame(pandasDict)
        return df
