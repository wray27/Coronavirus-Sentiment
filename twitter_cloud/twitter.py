import twint
from datetime import date, timedelta
from os.path import dirname, abspath
from os import system
from datetime import datetime
import pandas as pd
import sys
import csv
import boto3



class TwintScraper:
    def __init__(self):
        pass

    def scrape(self, output_file="tweets.csv", get_location=True, search="coronavirus", amount=None, store_csv=True, dont_print=False, since=None, until=None):
        c = twint.Config()
        c.Location = get_location
        c.Search = search
        c.Limit = amount
        c.Store_csv = store_csv
        c.Hide_output = dont_print
        c.Output = output_file
        if(since):
            c.Since = since
        if(until):
            c.Until = until
        c.Format = "Username: {username} | Tweet: {tweet} | Location: {geo} | Time: {time} | Date: {date}"
        
        twint.run.Search(c)
        

class TwintDataMiner:
    def __init__(self):
        #place in list of outputted CSV file from TwintScraper
        self.twintCSVDict = {
            "id": 0,
            "conversation_id": 1,
            "created_at": 2,
            "date": 3,
            "time": 4,
            "timezone": 5,
            "user_id": 6,
            "username": 7,
            "name": 8,
            "place": 9,
            "tweet": 10
            #more can be added- see first line of csv
        }

    #use twint scraper to get tweets, use kwargs to define arguments (specified in TwintScraper)
    def scrape(self, **kwargs):
        ts = TwintScraper()
        ts.scrape(**kwargs)

    #get specific attribute of the tweets e.g. date

    def getAttribute(self, info, file):
        outputList = []
        with open(file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for idx, line in enumerate(reader):
                outputList.append(line[self.twintCSVDict[info]])
        return outputList

    #turn csv file into pandas dataframe with columns specified in attributes

    def getPandasDataFrame(self, attributes, file):
        pandasDict = dict()
        for att in attributes:
            pandasDict[att] = self.getAttribute(att, file)
        df = pd.DataFrame(pandasDict)
        return df


if __name__ == '__main__':
    tdm = TwintDataMiner()
  
    
    #  month of march
    month = "03"
    start = f"2020-{month}-01"
    end = f"2020-{month}-31"
    
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    days = pd.date_range(start=start, end=end).to_pydatetime().tolist()
    day_list = [datetime.strftime(d, '%Y-%m-%d') for d in days]

    # one day of tweets is given to an instance
    day = int(sys.argv[1])
    afternoon = ['12:00:00', '12:10:00']
    evening = ['21:00:00', '21:10:00']

    # afternoon tweets for that day
    since = day_list[day] + " " + afternoon[0]
    until = day_list[day] + " " + afternoon[1]
    # print(since)
    # print(until)
    tweetsFile = f"./tweets-march_{day}_afternoon.csv"
    # print(tweetsFile)
    tdm.scrape(since=since, until=until, output_file=tweetsFile, dont_print=True, amount=1)
    print(f"Done. {day}")
    

    
    # evening tweets for that daybet
    # since = day_list[day] + " " + evening[0]
    # until = day_list[day] + " " + evening[1]
    # tweetsFile = dirname(abspath(__file__)) + f"{month}_{day}_evening.csv"
    # tdm.scrape(since=None, until=None, output_file=tweetsFile,
    #           get_location=True, dont_print=True, amount=15000)

    # filename = f"tweets{month}_{day}"
    # system(f"touch {filename}")
    # upload_file(filename)

    # print(day)
    # print

    # since = "2020-01-01 12:00:00"
    # until= "2020-01-01 12:50:00"
  
    # print(since)
    # print(until)

    # tweetsFile = "testing.csv"
    # print(tweetsFile)
    
    # tdm.scrape(since=since, until=until, output_file=tweetsFile,
    #         amount=100, dont_print=True, get_location=False)
    # tdm.scrape(amount=100, output_file='./testing.csv', dont_print=True, since='2020-01-01 12:00:00', until='2020-01-01 12:50:00')
    # upload_file("testing.csv")