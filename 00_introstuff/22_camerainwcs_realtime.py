import dlrc_control as ctrl
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import time, sys, argparse
import py_at_broker as pab


argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--detail",
                       help="Plot camera reconstruction with additional detail of histograms",
                       action="store_true")
args = argparser.parse_args()

def redraw_camera(Teecamera, joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer, colors_in_buffer, box=None, detailed=False):

    T0ee, _,_,_ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    # K = np.array([[focal_length,0,principal_point[1]/ camera_resolution[1]],
    #              [0, focal_length, principal_point[0]/ camera_resolution[0]],
    #              [0,0,1]])
    # T0caminternal = np.dot(T0cam, np.linalg.inv(K))
    ccs_points, point_colors, pp_ccs = img_to_ccs(depth, principal_point, camera_resolution, skip=13, rgb_image=rgb)
    wcs_points = [np.dot(T0cam, ccs_point) for ccs_point in ccs_points]
    points_in_buffer[i % bufsize, :,:] = wcs_points
    colors_in_buffer[i % bufsize, :,:] = np.array(point_colors).reshape(-1,3)/255
    colors_in_buffer = colors_in_buffer.reshape(-1,3)
    if colors_in_buffer.shape[0] == 1: colors_in_buffer = np.squeeze(colors_in_buffer, axis=0)


    #ax1.scatter(list(zip(*wcs_points))[0], list(zip(*wcs_points))[1], list(zip(*wcs_points))[2], s=50, c='b', alpha=0.1)
    ax1.clear()
    ax1.scatter(points_in_buffer[:,:,0], points_in_buffer[:,:,1], points_in_buffer[:,:,2], s=50, c=colors_in_buffer, alpha=0.5)
    colors_in_buffer = colors_in_buffer.reshape((bufsize, -1, 3))
    #ax1.axis('equal')
    # ax1.set_xlim(-1, 1)
    # ax1.set_ylim(-0.5,0.5)
    # ax1.set_zlim(-0.1, 1.5)
    # for equal aspect ratio: all limits have a span of 2 meters
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-0.1, 1.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    camera_origin = np.dot(T0cam, np.array([0,0,0,1]))
    # camera_target = np.dot(T0cam, np.array([0,0,1,1])*depth[principal_point[1], principal_point[0]]/1000)
    # find the point the camera is looking at. The principal point is structured
    # (y,x), so the order should be p_p[0], p_p[1]
    # also, it should be calculated just like the other image points just like
    # in img_to_ccs
    camera_target = np.dot(T0cam, np.array([0, 0, 1, 1]) * depth[principal_point[0], principal_point[1]] / 1000)
    pp_wcs = np.dot(T0cam, pp_ccs)
    # DEBUG CAMERA ARROWS
    # END DEBUG CAMERA ARROWS
    camera_vector = pp_wcs-camera_origin
    ax1.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
               camera_vector[0], camera_vector[1], camera_vector[2],
               color='r')

    im = ax2.imshow(depth, origin='lower')
    plt.colorbar(im, cax=ax2)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.clear()
    ax3.imshow(rgb)
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_ylim(ax3.get_ylim()[::-1])

    if detailed:
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
        ax7.set_position([box.x0, box.y0, box.width * 0.5, box.height * 1.0])
        ax7.set_xlim(ax7.get_xlim()[::-1])
        ax7.set_ylim(ax7.get_ylim()[::-1])


    return points_in_buffer, colors_in_buffer


# setting camera parameters
Teecamera = np.eye(4)
# bc the DH transformation matrix only accounts for the translation of the flange from j7 and not its rotation of 45 degrees
flange_angle = 45 * np.pi/180
Teecamera[:2,:2] = [[np.cos(flange_angle), -np.sin(flange_angle)],
                    [np.sin(flange_angle), np.cos(flange_angle)]]
Teecamera[:3,3] = [ 0.0145966,  -0.05737133, -0.03899948]
#Teecamera[:3,2] = [0.07378408, 0.02544135, 1.00632009] # minor rotation
camera_resolution = (240,320)
principal_point = (240//2, 320//2)
focal_length = 0.00193


# initializing subplots
if args.detail: subplot_row, subplot_column = 4,2
else: subplot_row, subplot_column = 2,2
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')
ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column,4)
box = None
if args.detail:
    ax4 = fig.add_subplot(subplot_row, subplot_column,5)
    ax5 = fig.add_subplot(subplot_row, subplot_column,6)
    ax6 = fig.add_subplot(subplot_row, subplot_column,7)
    ax7 = fig.add_subplot(subplot_row, subplot_column,8)
    box = ax7.get_position()

# configuring plotting details
i=0
n_points = 475
bufsize = 4
points_in_buffer = np.empty((bufsize, n_points, 4))
colors_in_buffer = np.empty((bufsize, n_points, 3))


# connect to data stream
#broker = ctrl.initialize()
broker = pab.broker("tcp://localhost:51468")
broker.request_signal("franka_state", pab.MsgType.franka_state)
broker.request_signal("realsense_images", pab.MsgType.realsense_image)


while (True): # continuous stream

    # get data stream
    rgb, depth = grab_image(broker)
    state_msg = broker.recv_msg("franka_state", -1)
    joint = state_msg.get_j_pos()

    # plot/redraw data values
    redraw_camera(Teecamera, joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer, colors_in_buffer, box, args.detail)
    fig.canvas.draw()
    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()

    # inp = input('write stop to stop ')
    # if inp == 'stop':
    #     sys.exit()

# visualize trace of lidar joint velocity
# conditin traj to avoid seen objects
# img segm to pt cld segm
# jt config wrt pixel depth values