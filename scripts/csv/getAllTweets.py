import sys, csv, os


#this script collates all the location tweets from aarons march-afternoon folder

tweets_folder = '/home/dominic/Downloads/march-afternoon/'

output_csv = '../../data/location_tweets1.csv'

def get_tweets(dir, out_csv):
    with open(out_csv, 'w') as outcsv:
        csvwriter = csv.writer(outcsv)
        for file in os.listdir(dir):
            if(file.endswith(".csv")):
                with open(dir + file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for line in csvreader:
                        if(line[9]):
                            csvwriter.writerow(line)


get_tweets(tweets_folder, output_csv)
