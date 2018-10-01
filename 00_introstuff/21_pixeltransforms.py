import dlrc_control as ctrl
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import time

Teecamera = np.eye(4)
# bc the DH transformation matrix only accounts for the translation of the flange from j7 and not the rotation of 45 degrees
xy_angle = 45 * np.pi/180 # simply tried by intuition
Teecamera[:2,:2] = [[np.cos(xy_angle), -np.sin(xy_angle)],
                    [np.sin(xy_angle), np.cos(xy_angle)]]
Teecamera[:3,3] = [ 0.0145966,  -0.05737133, -0.03899948]
#Teecamera[:3,2] = [0.07378408, 0.02544135, 1.00632009]

camera_resolution = (240,320)
principal_point = (240//2, 320//2)
focal_length = 0.00193
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180927T1204050000.pkl")) # flat image, x should be 0.1 to edge 0.35, y should be 0.02 to 0.32 in flat img
data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180927T1541170000.pkl")) # looking at itself, base at center, heart markings
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180926T0812290000.pkl"))[-21:-20] # realsense and joint data
depth_images = data.realsense_depthdata
rgb_images = data.realsense_rgbdata
joint_configs = data.state_j_pos

bufsize = 5



subplot_row, subplot_column = 4,2
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')

ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column,4)
ax4 = fig.add_subplot(subplot_row, subplot_column,5)
ax5 = fig.add_subplot(subplot_row, subplot_column,6)
ax6 = fig.add_subplot(subplot_row, subplot_column,7)
ax7 = fig.add_subplot(subplot_row, subplot_column,8)
box = ax7.get_position()

i=0
n_points = 475
points_in_buffer = np.empty((bufsize, n_points, 4))
colors_in_buffer = np.empty((bufsize, n_points, 3))
for joint,depth,rgb in zip(joint_configs, depth_images, rgb_images):

    #if i == 0: time.sleep(2)
    T0ee, _,_,_ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    # K = np.array([[focal_length,0,principal_point[1]/ camera_resolution[1]],
    #              [0, focal_length, principal_point[0]/ camera_resolution[0]],
    #              [0,0,1]])
    # T0caminternal = np.dot(T0cam, np.linalg.inv(K))
    ccs_points, point_colors = img_to_ccs(depth, principal_point, camera_resolution, skip=13, rgb_image=rgb)
    wcs_points = [np.dot(T0cam, ccs_point) for ccs_point in ccs_points]
    points_in_buffer[i % bufsize, :,:] = wcs_points
    colors_in_buffer[i % bufsize, :,:] = np.array(point_colors)/255
    colors_in_buffer = colors_in_buffer.reshape(-1,3)
    if colors_in_buffer.shape[0] == 1: colors_in_buffer = np.squeeze(colors_in_buffer, axis=0)
    #ax1.scatter(list(zip(*wcs_points))[0], list(zip(*wcs_points))[1], list(zip(*wcs_points))[2], s=50, c='b', alpha=0.1)
    ax1.clear()
    ax1.scatter(points_in_buffer[:,:,0], points_in_buffer[:,:,1], points_in_buffer[:,:,2], s=50, c=colors_in_buffer, alpha=0.5)
    colors_in_buffer = colors_in_buffer.reshape((bufsize, -1, 3))
    #ax1.axis('equal')
    #ax1.set_xlim(0, 0.5)
    # ax1.set_ylim(-0.5,0.5)
    #ax1.set_zlim(-0.1, 0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    camera_origin = np.dot(T0cam, np.array([0,0,0,1]))
    wcs_pp = np.dot(T0cam, np.array([0,0,depth[principal_point[1], principal_point[0]]/1000,1]))
    ax1.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
               wcs_pp[0], wcs_pp[1], wcs_pp[2],
               color='r')

    im = ax2.imshow(depth, origin='lower')
    plt.colorbar(im, cax=ax2)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.clear()
    ax3.imshow(rgb)
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_ylim(ax3.get_ylim()[::-1])


    ax7.set_position([box.x0, box.y0, box.width * 0.5, box.height * 1.0])

    ax4.clear()
    ax4.hist(list(zip(*wcs_points))[0], bins=80)
    ax4.set_title('x values in wcs')

    ax5.clear()
    ax5.hist(list(zip(*wcs_points))[1], bins=80)
    ax5.set_title('y values in wcs')

    ax6.clear()
    ax6.hist(list(zip(*wcs_points))[2], bins=80)
    ax6.set_title('z values in wcs')

    ax7.clear()
    ax7.scatter(list(zip(*wcs_points))[0], list(zip(*wcs_points))[1])
    ax7.set_xlim(ax7.get_xlim()[::-1])
    ax7.set_ylim(ax7.get_ylim()[::-1])

    if i == len(joint_configs)-1:
        input('enter to close window')

    plt.pause(1)
    plt.show()
    i+=1