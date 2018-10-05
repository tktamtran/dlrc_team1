

# add label column to datasets
import numpy as np
import pandas as pd
from os import listdir

directory_data = 'measurements/'
files_data = [f for f in listdir(directory_data) if 'datawcs_nonrobot_batch_rt' in f]
print(files_data)

for f in files_data:
    data = pd.read_pickle(directory_data + f)
    data['label'] = np.zeros((data.shape[0],1))
    #data.reset_index(inplace=True)
    #print(data.iloc[0:2])
    data.to_pickle(directory_data + f)