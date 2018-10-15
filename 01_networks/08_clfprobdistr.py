
from Network00 import *
import math, os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pytorch.set_default_tensor_type('torch.FloatTensor')

from os import listdir
from os.path import isfile, join




net = Network01(9,6)
criterion = LogProb_Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-03, weight_decay=1e-2)

directory_data = 'datapixelflat/'
files_data = sorted([f for f in listdir(directory_data) if 'datapixelcustom' in f])
batch_size = 500

# loop over datasets
loss_history = []
for e,file_data in enumerate(files_data):

    print(file_data)
    data = pd.read_pickle(directory_data + file_data)
    dataset = data[['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'px', 'py', 'wcc_x', 'wcc_y', 'wcc_z']]
    #dataset = torch.from_numpy(dataset.values)

    n_batches = math.floor(dataset.shape[0]/batch_size)
    batch_indices = [np.random.choice(dataset.shape[0], batch_size) for b in range(n_batches)]

    for b in range(n_batches):

        batch = dataset.iloc[batch_indices[b]]
        batch_pred = net(torch.from_numpy(batch[['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'px', 'py']].values).float())
        if b%200==0:
            print('batch_pred \n', batch_pred)
            input('..')
        batch_wcp = torch.from_numpy(batch[['wcc_x', 'wcc_y', 'wcc_z']].values).float()
        if batch_size==1: loss = criterion(batch_pred[:3], batch_pred[3:], batch_wcp)
        else: loss = criterion(batch_pred[:,:3].reshape(-1,1), batch_pred[:,3:].reshape(-1,1), batch_wcp.reshape(-1,1))
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%1000==0:print('epoch', e, 'batch/iteration', str(b) + '/' + str(n_batches), 'loss', loss.item())

torch.save(net, 'models/Network01_trainedbinary.tar')
with open('models/Network01_trainedbinary_losshistory', 'wb') as fp:
    pickle.dump(loss_history, fp)

#plt.plot(loss_history)
#plt.show()





