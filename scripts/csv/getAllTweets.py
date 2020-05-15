import sys, csv, os


#this script collates all the location tweets from aarons march-afternoon folder

tweets_folder = '/media/dominic/New Volume/applied_data_science/OneDrive_1_08-05-2020/'

output_csv = '/media/dominic/New Volume/applied_data_science/final_location_tweets.csv'

def get_tweets(dir, out_csv):
    with open(out_csv, 'w') as outcsv:
        csvwriter = csv.writer(outcsv)
        for file in os.listdir(dir):
            print(file)
            if(file.endswith(".csv")):
                with open(dir + file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for line in csvreader:
                        if(line[9]):
                            csvwriter.writerow(line)


get_tweets(tweets_folder, output_csv)
