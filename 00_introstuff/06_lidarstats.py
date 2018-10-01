
import py_at_broker as pab
import matplotlib.pyplot as plt
import time
import numpy as np

d = pab.broker("tcp://10.250.144.21:51468")
d.request_signal("franka_lidar", pab.MsgType.franka_lidar)


def gen_lidarstats(broker, n_samples, n_lidars=9):

    print('collecting signal values')
    list_msmt = []
    for i in range(n_samples):
        msmt = broker.recv_msg("franka_lidar", -1).get_data()
        list_msmt.append(msmt)
        if i % 10 == 0: print(i)

    list_msmt = np.array(list_msmt)
    means = list_msmt.mean(axis=0)
    stds = list_msmt.std(axis=0)
    maxs = list_msmt.max(axis=0)
    mins = list_msmt.min(axis=0)

    print('plotting lidar stats')
    plt.errorbar(np.arange(n_lidars), means, stds, fmt='ok', lw=3)
    plt.errorbar(np.arange(n_lidars), means, [means-mins, maxs-means], fmt='.k', ecolor='gray', lw=1)
    plt.xlim(-1, n_lidars)
    plt.ylim(0, 10000)
    plt.title('lidar signal stats based on ' + str(n_samples) + ' samples' + '\n' + str(means))
    plt.xlabel('lidar index')
    plt.ylabel('distance in mm')
    plt.show()


for i in range(100):
    gen_lidarstats(d,100)
    input('block next lidar...')