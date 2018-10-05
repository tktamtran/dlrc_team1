
from Network00 import *
import math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pytorch.set_default_tensor_type('torch.FloatTensor')

from os import listdir
from os.path import isfile, join

directory_data = 'data_robot/'
files_data = [f for f in listdir(directory_data) if isfile(join(directory_data, f))]
batch_size = 200

net = Network00()
criterion = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-04, weight_decay=1e-2)


# loop over datasets
loss_history = []
for e,file_data in enumerate(files_data):

    print(type(file_data), file_data)
    data = pd.read_pickle(directory_data + file_data)
    dataset = torch.from_numpy(data.values)

    n_batches = math.floor(dataset.shape[0]/batch_size)
    batch_indices = [np.random.choice(dataset.shape[0], batch_size) for b in range(n_batches)]

    for b in range(n_batches):

        batch = dataset[batch_indices[b]]
        batch_pred = net(batch.float())
        loss = criterion(batch_pred, torch.ones(batch_size))
        print('epoch', e, 'batch/iteration', str(b)+'/'+str(n_batches), 'loss', loss.item())
        loss_history.append(math.log(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#torch.save(net.state_dict(), 'models/trained_Network00.pkl')

plt.plot(loss_history)
plt.show()