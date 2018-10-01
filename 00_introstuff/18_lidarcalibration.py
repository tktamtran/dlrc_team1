import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time
import numpy as np
import pandas as pd

# This script reads the robots joint positions from a txt file and
# anaimates the resulting scatter plot.

# script should be run in parallel with a simulating file which constantly
# creates new joint positions

with open("LidarViz.pkl", "rb") as f:
    vizdata = pd.read_pickle(f)

fig = plt.figure()
ax1 = p3.Axes3D(fig)
startframe = 200  # choose the frame where the action starts
tracelength = 25  # number of real world points to keep as a trace

# px = vizdata['xy'][vizdata['xidx']]
# py = vizdata['xy'][vizdata['yidx']]
# pz = np.zeros(len(px))
# p = np.column_stack((px, py, pz))
# pxy =np.column_stack((px, py))
realworld = np.zeros((3, tracelength))
lxy = np.array([[l[0], l[1]] for l in vizdata['base_origin']])
lz = np.array([l[2] for l in vizdata['base_origin']])
lp = np.column_stack((lxy, lz)).T
l_ax = vizdata['base_z']
l_reading = vizdata['lidar_readings']
data = dict()


# TODO: it would be much faster just to change the data of the plots
#       instead of redrawing them completely
# TODO: make it more general so that it simply accepts some points for scatter
#       and some arrows with origins and directions for quiver
# data is a data structure that contains the keys:
# lidar_xyz:  the position of the lidar in real world coordinates
# lidar_axis: the direction the lidar z (reading) axis points to
# lidar_reading: the current sensor reading
# realworld:  a buffer of previously seen real world points (3xbufsize)
def redraw(data):
    if not hasattr(redraw, "counter"):
        redraw.counter = 0  # it doesn't exist yet, so initialize it
    redraw.counter += 1
    ax1.clear()
    ax1.axis('equal')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    # draw the sensor axis, calculate the length first

    l_sens = data['lidar_axis'] * data['lidar_reading']
    lp = data['lidar_xyz']
    ax1.scatter(lp[0], lp[1], lp[2], s=50, c='r')
    # draw the real world plot using an ugly workaround for the trace
    # executed in the most unpythonic way ever
    # use the 'realworld' as a ringbuffer, indexing cols via modulo
    if not hasattr(redraw, "realworld"):
        redraw.realworld = np.zeros((3, 25))
    realpoint = (lp[:].reshape((3, 1)) + l_sens).reshape((3))
    replacecol = np.mod(redraw.counter, realworld.shape[1])
    redraw.realworld[:, replacecol] = realpoint
    ax1.scatter(redraw.realworld[0, :],
                redraw.realworld[1, :],
                redraw.realworld[2, :], s=50, c='b', alpha=0.1)
    ax1.quiver(lp[0], lp[1], lp[2],
               l_sens[0], l_sens[1], l_sens[2],
               color='r')
    # print(f"World: { p[i,:]}")
    # print(f"Lidar: {lp[i,:]}")


for i in range(200, len(vizdata['base_lidar_T'])):
    data['lidar_xyz'] = lp[:, i]
    data['lidar_axis'] = vizdata['base_z'][i]
    data['lidar_reading'] = vizdata['lidar_readings'][i]
    redraw(data)
    fig.canvas.draw()
    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()

