import pickle
import csv
from progress.bar import IncrementalBar as Bar
from twitter import TwintDataMiner

class SlowBar(Bar):
    suffix = '%(index)d/%(max)d, %(percent).1f%%, %(remaining_hours)d seconds remaining'
    @property
    def remaining_hours(self):
        return self.eta 

def read_data_from_file(filename):
    data = [[]]

    with open(filename, "r") as f:
        
        f_data = csv.reader(f)
        for row in f_data:
            data.append([list(row)])

    print(data[1])


def pickle_files(pkl_path, no_files):

    file_list = []
    bar = SlowBar('Pickling Data', max=no_files)
    
    # retrieving data from multiple files
    for i in range(no_files):
        filename = f"./tweets-march_{i}_afternoon.csv"
        # file_list.append(read_all_data(filename))
        bar.next()
    bar.finish()

    with open(pkl_path, "wb") as fileobj:
        pickle.dump(file_list, fileobj)


def main():
    

    file = f"./tweets-march_1_afternoon.csv"
    read_data_from_file(file)
     


if __name__ == "__main__":
    main()
