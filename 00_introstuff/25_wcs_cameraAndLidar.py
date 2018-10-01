# script to model lidar and camera data on same world coordinate map, in either real time or from a fixed dataset

from utils import *

def redraw_camera(Teecamera, joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer, colors_in_buffer, i, box=None, detailed=False):
    '''assumes subplot axes to already exist'''

    T0ee, _,_,_ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    # K = np.array([[focal_length,0,principal_point[1]/ camera_resolution[1]],
    #              [0, focal_length, principal_point[0]/ camera_resolution[0]],
    #              [0,0,1]])
    # T0caminternal = np.dot(T0cam, np.linalg.inv(K))
    ccs_points, point_colors = img_to_ccs(depth, principal_point, camera_resolution, skip=13, rgb_image=rgb)
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
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.5,0.5)
    ax1.set_zlim(-0.1, 1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    camera_origin = np.dot(T0cam, np.array([0,0,0,1]))
    camera_target = np.dot(T0cam, np.array([0,0,1,1])*depth[principal_point[0], principal_point[1]]/1000)
    ax1.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
               camera_target[0], camera_target[1], camera_target[2],
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



# set up program mode and data import
argparser = argparse.ArgumentParser()
argparser.add_argument("-mr", "--mode_realtime",
                       help="Generate WCS model in realtime",
                       action="store_true")
argparser.add_argument("-md", "--mode_dataset",
                       help="Generate WCS model from a dataset",
                       action="store_true")
args = argparser.parse_args()

assert(args.mode_realtime + args.mode_dataset == 1)

if args.mode_realtime:
    # connect to data stream
    broker = pab.broker("tcp://localhost:51468")
    broker.request_signal("franka_state", pab.MsgType.franka_state)
    broker.request_signal("realsense_images", pab.MsgType.realsense_image)
    broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)
    n = 1000 # limit on how long real time can run

if args.mode_dataset:
    filename = sys.argv[3]
    data = pd.DataFrame(pd.read_pickle(filename))
    data.reset_index(inplace=True)
    n = data.shape[0]


# initializing subplots
subplot_row, subplot_column = 2,2
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')
ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column,4)


# configuring plotting details, including a buffer
i=0
n_points = 475
bufsize = 4
points_in_buffer = np.empty((bufsize, n_points, 4))
colors_in_buffer = np.empty((bufsize, n_points, 3))
camera_resolution = (240,320)
principal_point = (240//2, 320//2)
lidar_info = {}
for l in transformation_values.keys():
    if l != 'camera':
        lidar_info[l] = {
            'T_b_l': None,
            'wcp_origin': None,
            'sensor_reading': None,
            'wcp_target': None,
            'buffer': np.empty((bufsize,4))
        }


while True:

    # gather data based on mode
    if args.mode_realtime:
        state_msg = broker.recv_msg("franka_state", -1)
        joint = state_msg.get_j_pos()
        lidar_readings = broker.recv_msg("franka_lidar", -1).get_data()[:]/1000
        rgb, depth = grab_image(broker)

    if args.mode_dataset:
        joint = data.iloc[i]['state_j_pos']
        lidar_readings = data.iloc[i]['lidar_data']/1000
        rgb = data.iloc[i]['realsense_rgbdata']
        data = data.iloc[i]['realsense_depthdata']

    for l,v in transformation_values.items():
        if l != 'camera':
            lidar_idx = int(l[-1:])
            _, _, T_b_j, _ = get_jointToCoordinates(joint, untilJoint=v['joint_number'])
            lidar_info[l]['T_b_l'] = np.dot(T_b_j, v['transformation_matrix'])
            lidar_info[l]['wcp_origin'] = np.dot(lidar_info[l]['T_b_l'], np.array([0, 0, 0, 1]))
            lidar_info[l]['sensor_reading'] = lidar_readings[lidar_idx]
            lidar_info[l]['wcp_target'] = np.dot(lidar_info[l]['T_b_l'], np.array([0,0, lidar_info[l]['sensor_reading'], 1]))
            lidar_info[l]['buffer'][i % bufsize, :] = lidar_info[l]['wcp_target']

    # plot data
    # not having a build/generate mode bc the camera will move around too much to overlap the plot data
    # TODO would be cool to incorporate the robot/joint skeleton
    points_in_buffer, colors_in_buffer = redraw_camera(transformation_values['camera']['transformation_matrix'], joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer,colors_in_buffer,i)
    for l,v in lidar_info.items():
        ax1.scatter(v['buffer'][:,0], v['buffer'][:,1], v['buffer'][:,2], c=transformation_values[l]['color'])
        ax1.quiver(v['wcp_origin'][0], v['wcp_origin'][1], v['wcp_origin'][2],
                   v['wcp_target'][0], v['wcp_target'][1], v['wcp_target'][2], color=transformation_values[l]['color'])

    fig.canvas.draw()
    plt.pause(3)
    time.sleep(3)
    plt.show()

    # collect data mapped into wcs


    i+=1
    if i >= n: break