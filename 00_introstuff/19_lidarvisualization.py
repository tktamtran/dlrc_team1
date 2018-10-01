# visualize lidar readings in real-time

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import time
import numpy as np
import dlrc_control as ctrl
import py_at_broker as pab
import utils

lidar_idx = 0
joint_idx = 6

fig = plt.figure()
ax1 = p3.Axes3D(fig)
broker = ctrl.initialize()
broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)
broker.request_signal("franka_state", pab.MsgType.franka_state)

# the Transformation matrix we got from regression from the joint to lidar
# T_j_l
params = np.array([ 0.01807252,  0.00734784,  0.03238632, -0.09107123, -0.02269616,  0.93559049])
T_j_l = np.zeros((4,4))
T_j_l[0:3,2] = params[3:6]
T_j_l[0:3,3] = params[:3]
T_j_l[3,3] = 1

# TODO: it would be much faster just to change the data of the plots
#       instead of redrawing them completely
# TODO: make it more general so that it simply accepts some points for scatter
#       and some arrows with origins and directions for quiver
# data is a data structure that contains the keys:
# lidar_xyz:  the position of the lidar in real world coordinates
# lidar_axis: the direction the lidar z (reading) axis points to
# lidar_reading: the current sensor reading
def redraw(data):
    redraw.bufsize = 25 # hard coded for now
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
        redraw.realworld = np.zeros((3, redraw.bufsize))
    realpoint = (lp.reshape((3, 1)) + l_sens.reshape((3,1))).reshape((3))
    replacecol = np.mod(redraw.counter, redraw.realworld.shape[1])
    redraw.realworld[:, replacecol] = realpoint
    ax1.scatter(redraw.realworld[0, :],
                redraw.realworld[1, :],
                redraw.realworld[2, :], s=50, c='b', alpha=0.1)
    ax1.quiver(lp[0], lp[1], lp[2],
               l_sens[0], l_sens[1], l_sens[2],
               color='r')
    # print(f"World: { p[i,:]}")
    # print(f"Lidar: {lp[i,:]}")
plt.ion()

while (True):
    state_msg = broker.recv_msg("franka_state", -1)
    j_pos = state_msg.get_j_pos()
    Tproduct, Tlist, T_b_j, EE_coord = \
        utils.get_jointToCoordinates(j_pos, untilJoint=joint_idx)
    lidar = broker.recv_msg("franka_lidar", -1).get_data()[lidar_idx]/1000
    print(lidar)
    data = dict()
    data['lidar_xyz'] = np.matmul(T_b_j, T_j_l)[:3,3] # translation
    data['lidar_axis'] = np.matmul(T_b_j, T_j_l)[:3, 2] # rotation
    data['lidar_reading'] = lidar # point touched
    redraw(data)
    fig.canvas.draw()
    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()