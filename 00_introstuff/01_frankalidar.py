import py_at_broker as pab
import matplotlib.pyplot as plt
import time
import numpy as np

d = pab.broker("tcp://10.250.144.21:51468")
d.request_signal("franka_lidar", pab.MsgType.franka_lidar)
# d.request_signal("franka_images", pab.MsgType.franka_images)

print('Starting LIDAR measurements')
data = d.recv_msg("franka_lidar", -1).get_data()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(range(len(data)), data, 'r-')
datastore = np.full((data.shape[0], 100), np.nan)
iter = 0

# while(True):
for i in range(100):
    data = d.recv_msg("franka_lidar", -1).get_data()
    line1.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    datastore = np.roll(datastore, (1,0), 1)
    datastore[:,0] = data
    #print(f'New Data at {time.time()}: {data}')
    #recheck the statistics
    print(f'{iter}\t{np.nanmean(datastore,1)}')
    # print(f'{np.nanstd(datastore, 1)}')
    #print(datastore)
    #print(data)
    #print(data.shape)
    #print(datastore[:,1].shape)
    iter += 1