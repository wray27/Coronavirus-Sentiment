import twint
import threading
from datetime import date, timedelta

class TwintScraper:
    def __init__(self):
        pass

    def scrape(self, get_location=True, search="coronavirus", amount=100, store_csv=True, output_file= "data/tweets.csv", dont_print=False, since=None, until=None):
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
