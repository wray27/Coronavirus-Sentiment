import twint


class TwintScraper:
    def __init__(self):
        pass

    def scrape(self, get_location=True, search="coronavirus", amount=100, store_csv=True, output_folder= "data/tweets.csv", dont_print=False):
        c = twint.Config()
        c.Location = get_location
        c.Search = search
        c.Limit = amount
        c.Store_csv = store_csv
        c.Hide_output = dont_print
        c.Output = output_folder
        c.Format = "Username: {username} | Tweet: {tweet} | Location: {geo} | Time: {time}"
        twint.run.Search(c)
