from TwintDataMiner import TwintDataMiner
from os.path import dirname, abspath
import sys

#tweetsFile = "../data/tweets.csv"

tdm = TwintDataMiner()


def getMinuteTweets(tdm):
    #find all tweets in minute after argument

    #argument in form  YYYY-MM-DD HH:MM:SS
    since = sys.argv[1]
    oneMin = int(since.split(" ")[1].split(":")[1]) + 1
    if(oneMin < 10):
     oneMin = "0" + str(oneMin)
    else:
     oneMin = str(oneMin)
    until =  since.split(" ")[0] + " " + since.split(" ")[1].split(":")[0]+ ':' + str(oneMin) + ":" + since.split(" ")[1].split(":")[2]

    tweetsFile = dirname(dirname(abspath(__file__))) + "/data/timeSeriesTweets.csv"
    tdm.scrape(since=since, until=until, output_file=tweetsFile, amount = 5000, get_location=False)


def getXTweetsPerDay(tdm, x):
    since = sys.argv[1] + " 00:00:00"
    until = sys.argv[1] + " 23:59:00"
    tweetsFile = dirname(dirname(abspath(__file__))) + "/data/" + str(x) + "TweetsPerDay.csv"
    tdm.scrape(since=since, until =until, output_file=tweetsFile, amount=x, get_location=False)

getXTweetsPerDay(tdm, 1000)
