# make a composed reconstruction of image data to wcs

# code works to construct world model
# however, as camera moves around robot, the robot moves as well
# which, unless we incorporate joint data, this occlusion/movement impedes with the concept of a world model...

from utils import *



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
subplot_row, subplot_column = 2,2
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.5, 0.5)
ax1.set_zlim(-0.1, 1.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column,4)


# connect to data stream
#broker = ctrl.initialize()
broker = pab.broker("tcp://localhost:51468")
broker.request_signal("franka_state", pab.MsgType.franka_state)
broker.request_signal("realsense_images", pab.MsgType.realsense_image)

wcs_plotted = np.empty((0,7))# to avoid too many overlaps in plot; holds the rounded coordinates rather than calc'd coords
i = 0
while (True):

    # get data stream
    rgb, depth = grab_image(broker)
    state_msg = broker.recv_msg("franka_state", -1)
    joint = state_msg.get_j_pos()

    print('grabbed data')

    T0ee, _, _, _ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    ccs_points, point_colors,_ = img_to_ccs(depth, principal_point, camera_resolution, skip=5, rgb_image=rgb)
    wcs_points = np.array([np.dot(T0cam, ccs_point) for ccs_point in ccs_points]) # still homogeneous
    wcs_points = np.round(wcs_points, 2)

    mask_to_plot = np.in1d(wcs_points[:,0], wcs_plotted[:,0]) * np.in1d(wcs_points[:,1], wcs_plotted[:,1]) * np.in1d(wcs_points[:,2], wcs_plotted[:,2])
    mask_to_plot = ~mask_to_plot
    wcs_to_plot = wcs_points[np.where(mask_to_plot)[0],:]
    colors_to_plot = point_colors[np.where(mask_to_plot)[0],:]
    wcs_plotted = np.append(wcs_plotted, np.concatenate((wcs_to_plot, colors_to_plot), axis=1), axis=0)
    # assumes the mapped point represents the same consistent data in reality
    # not cross-checking with color value..

    ax1.scatter(wcs_to_plot[:, 0], wcs_to_plot[:, 1], wcs_to_plot[:, 2], s=10, c=colors_to_plot/255, alpha=0.5)
    ax1.set_title('frame # ' + str(i) + ' with ' + str(sum(mask_to_plot)) + ' new points; ' + str(wcs_plotted.shape[0]) + ' total points')

    im = ax2.imshow(depth, origin='lower')
    plt.colorbar(im, cax=ax2)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.imshow(rgb)
    #ax3.set_xlim(ax3.get_xlim()[::-1])
    #ax3.set_ylim(ax3.get_ylim()[::-1])

    fig.canvas.draw()
    plt.pause(0.1)
    time.sleep(0.1)
    plt.show()

    i+=1

pd.DataFrame(wcs_plotted).to_pickle("measurements/wcs_points_torquecontrol0.pkl")