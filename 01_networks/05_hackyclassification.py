
from os import listdir
import pandas as pd

import sys
sys.path.insert(0, '/home/dlrc1/Desktop/00_introstuff')
from utils import *

from home.dlrc1.Desktop.00_introstuff.utils import *

# concateante the batches of data
while (False):
    directory_data = 'data_robot/'
    files_data = [f for f in listdir(directory_data) if 'datawcs_robot' in f]

    data_all = pd.read_pickle(directory_data + files_data[0])

    for e,file_data in enumerate(files_data[1:]):
        data_all = pd.concat([data_all, pd.read_pickle(directory_data + file_data)])

    print(data_all.shape)
    input('..')

    data_all.to_pickle(directory_data + 'datawcs_robot_batches0020.pkl')



directory_data = 'data_robot/'
database = pd.read_pickle(directory_data + 'datawcs_robot_batches0020.pkl')


# set up program mode and data import
argparser = argparse.ArgumentParser()
argparser.add_argument("-mr", "--mode_realtime",
                       help="Generate WCS model in realtime",
                       action="store_true")
argparser.add_argument("-md", "--mode_dataset",
                       help="Generate WCS model from a dataset",
                       action="store_true")
argparser.add_argument("-d", "--detail",
                       help="Plot camera reconstruction with additional detail of histograms",
                       action="store_true")
args = argparser.parse_args()
assert(args.mode_realtime + args.mode_dataset == 1)



# processing one frame at a time, replotting the wcp but with class colors


