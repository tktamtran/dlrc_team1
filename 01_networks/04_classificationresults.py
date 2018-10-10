from Network00 import *
import math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pytorch.set_default_tensor_type('torch.FloatTensor')

from os import listdir

model_name = 'Network00_trainedoneclass'
net = torch.load('models/' + model_name + '.tar')
criterion = nn.MSELoss(reduction = 'sum')
batch_size = None
n_batches = 50
directory_data = 'data_robot/'

def net_loss_values(net, files_data, in_batch_size, in_n_batches):

    assert (bool(in_batch_size) + bool(in_n_batches) == 1)

    loss_history = []
    nn_batches = []
    for e,file_data in enumerate(files_data): # epoch
        data = pd.read_pickle(directory_data + file_data)
        data = data[['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'co_x', 'co_y', 'co_z', 'co_w', 'ct_x', 'ct_y', 'ct_z', 'ct_w']]
        dataset = torch.from_numpy(data.values)

        if in_batch_size:
            batch_size = in_batch_size
            n_batches = math.floor(dataset.shape[0] / batch_size)
        if in_n_batches:
            n_batches = in_n_batches
            batch_size = math.floor(dataset.shape[0] / n_batches)
        nn_batches.append(n_batches)
        batch_indices = [np.random.choice(dataset.shape[0], batch_size) for b in range(n_batches)]

        for b in range(n_batches): # batches
            batch = dataset[batch_indices[b]]
            batch_pred = net(batch.float())
            loss = criterion(batch_pred, torch.ones(batch_size)) # comparing against positive one-class
            loss_history.append(loss)

    return loss_history, nn_batches


files_data = sorted([f for f in listdir(directory_data) if 'datawcs_robot' in f])
loss_robot, nn_batches = net_loss_values(net, files_data, in_batch_size=batch_size, in_n_batches=n_batches)
files_data = sorted([f for f in listdir(directory_data) if 'datawcs_nonrobot' in f])
loss_nonrobot, nn_batches = net_loss_values(net, files_data, in_batch_size=batch_size, in_n_batches=n_batches)

nn_batches = [0] + list(np.cumsum(nn_batches))
print(nn_batches)
ax = plt.gca()
ax.plot(loss_robot, label='robot')
ax.plot(loss_nonrobot, label='nonrobot')
ax.set_ylim(0, 0.50)
ax.set(xticks=nn_batches, xticklabels=['rt'+str(e) for e in range(22)])
ax.set_title(model_name)
ax.legend()
plt.show()


# collect data of objects in bounding box
# list loss values of wcs_otheragent data