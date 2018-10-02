# script to model lidar and camera data on same world coordinate map, in either real time or from a fixed dataset

from utils import *

def redraw_camera(Teecamera, joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer, colors_in_buffer, i, box=None, detailed=False, bound=1):
    '''assumes subplot axes to already exist'''

    T0ee, _,_,_ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    # K = np.array([[focal_length,0,principal_point[1]/ camera_resolution[1]],
    #              [0, focal_length, principal_point[0]/ camera_resolution[0]],
    #              [0,0,1]])
    # T0caminternal = np.dot(T0cam, np.linalg.inv(K))
    ccs_points, point_colors, ccs_point_pp = img_to_ccs(depth, principal_point, camera_resolution, skip=13, rgb_image=rgb)
    wcs_points = np.array([np.dot(T0cam, ccs_point) for ccs_point in ccs_points])
    if bound and (not boundingBox_min): wcs_points = np.array([p*bound/max(abs(p)) if max(abs(p)) > bound else p for p in wcs_points])
    if boundingBox_min: wcs_points = np.array([p for p in wcs_points if ((p[:3]<boundingBox_max).all() & (p[:3]>boundingBox_min).all())])

    if boundingBox_min: # then cannot handle collecting points for buffer, since arrays of varying size
        #points_in_buffer = np.array([]).reshape(0,4)
        mask_boundingBox_min = wcs_points > boundingBox_min
        points_in_buffer = wcs_points
        colors_in_buffer =
        points_in_buffer = points_in_buffer.unsqueeze()
    else:
        points_in_buffer[i % bufsize, :,:] = wcs_points
        colors_in_buffer[i % bufsize, :,:] = np.array(point_colors).reshape(-1,3)/255

    colors_in_buffer = colors_in_buffer.reshape(-1,3)
    if colors_in_buffer.shape[0] == 1: colors_in_buffer = np.squeeze(colors_in_buffer, axis=0)


    #ax1.scatter(list(zip(*wcs_points))[0], list(zip(*wcs_points))[1], list(zip(*wcs_points))[2], s=50, c='b', alpha=0.1)
    ax1.clear()
    ax1.scatter(points_in_buffer[:,:,0], points_in_buffer[:,:,1], points_in_buffer[:,:,2], s=50, c=colors_in_buffer, alpha=0.5)
    colors_in_buffer = colors_in_buffer.reshape((bufsize, -1, 3))
    #ax1.axis('equal')
    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1)
    ax1.set_zlim(-1,1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # ray of view to principal point
    camera_origin = np.dot(T0cam, np.array([0,0,0,1]))
    camera_pp_target = np.dot(T0cam, ccs_point_pp)
    camera_pp_vector = camera_pp_target - camera_origin
    if bound and max(abs(camera_pp_target)) > bound: camera_pp_vector = camera_pp_target*bound/max(abs(camera_pp_target)) - camera_origin
    ax1.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
               camera_pp_vector[0], camera_pp_vector[1], camera_pp_vector[2],
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


    return points_in_buffer, colors_in_buffer, wcs_points, camera_origin



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
    n = 5 # limit on how long real time can run
    batch_number = '_rt0'

if args.mode_dataset:
    filename = "/home/dlrc1/measurements/20181002T0821450000.pkl"
    data = pd.DataFrame(pd.read_pickle(filename))
    data.reset_index(inplace=True)
    n = data.shape[0]
    batch_number = filename.split('/')[-1].split('.')[0]


# initializing subplots
subplot_row, subplot_column = 2,2
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')
ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column,4)


# configuring plotting details, including a buffer
bound = None # in meter
boundingBox_min = [-0.7, -0.7, -0.1] # bounding box for the pts seen by the camera
boundingBox_max = [0.7, 0.7, 0.9]
bufsize = 1 # if boundingBox_min
n_points = 475
n_images = 5 # limit on data image in wcs collected
i=0
points_in_buffer = np.array()
colors_in_buffer = np.array([np.array([])]).reshape(bufsize, 0, 3)
# points_in_buffer = np.empty((bufsize, n_points, 4))
# colors_in_buffer = np.empty((bufsize, n_points, 3))
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


data_collect_real = []
data_collect_wcs = np.array([]).reshape(0,15)
while True:

    # gather data based on mode
    if args.mode_realtime:
        state_msg = broker.recv_msg("franka_state", -1)
        joint = state_msg.get_j_pos()
        lidar_readings = broker.recv_msg("franka_lidar", -1).get_data()[:]/1000
        rgb, depth = grab_image(broker)

        readings_partial = {
            'state_j_pos': joint,
            'lidar_data': lidar_readings,
            'realsense_rgbdata': rgb,
            'realsense_depth': depth
        }
        data_collect_real.append(readings_partial)

    if args.mode_dataset:
        joint = data.iloc[i]['state_j_pos']
        lidar_readings = data.iloc[i]['lidar_data']/1000
        rgb = data.iloc[i]['realsense_rgbdata']
        depth = data.iloc[i]['realsense_depthdata']

    for l,v in transformation_values.items():
        if l != 'camera':
            lidar_idx = int(l[-1:])
            _, _, T_b_j, _ = get_jointToCoordinates(joint, untilJoint=v['joint_number'])
            lidar_info[l]['T_b_l'] = np.dot(T_b_j, v['transformation_matrix'])
            lidar_info[l]['wcp_origin'] = np.dot(lidar_info[l]['T_b_l'], np.array([0, 0, 0, 1]))
            lidar_info[l]['wcp_target'] = np.dot(lidar_info[l]['T_b_l'], np.array([0,0, lidar_readings[lidar_idx], 1]))
            t = lidar_info[l]['wcp_target']
            if bound and max(abs(t))>bound: lidar_info[l]['wcp_target'] = t*bound/max(abs(t))
            lidar_info[l]['buffer'][i % bufsize, :] = lidar_info[l]['wcp_target']

    # plot data
    # not having a build/generate mode bc the camera will move around too much to overlap the plot data
    # TODO would be cool to incorporate the robot/joint skeleton
    points_in_buffer, colors_in_buffer, wcs_points_camera, camera_origin = redraw_camera(transformation_values['camera']['transformation_matrix'], joint, depth, rgb, principal_point, camera_resolution, bufsize, points_in_buffer,colors_in_buffer,i,bound)
    for l,v in lidar_info.items():
        ax1.scatter(v['buffer'][:,0], v['buffer'][:,1], v['buffer'][:,2], c=transformation_values[l]['color'])
        ax1.quiver(v['wcp_origin'][0], v['wcp_origin'][1], v['wcp_origin'][2],
                   v['wcp_target'][0]-v['wcp_origin'][0], v['wcp_target'][1]-v['wcp_origin'][1], v['wcp_target'][2]-v['wcp_origin'][2],
                   color=transformation_values[l]['color'])
    ax1.set_title('frame ' + str(i))

    # collect both data original (above, for better msmt acc) and data mapped into wcs (that already includes pre-processing)
    #data_collect_wcs.append(joint + camera_origin + wcs_points_camera) # for purpose of classification NN
    data_collect_wcs = np.vstack([data_collect_wcs, np.concatenate((np.tile(joint.reshape(-1,7), (n_points,1)), np.tile(camera_origin.reshape(-1,4), (n_points,1)), wcs_points_camera), axis=1)])
    # joint and camera_origin broadcasted so that each wcs pixel gets full set of data
    # the data type/format that you choose to train NN with == data type to test NN with

    fig.canvas.draw()
    plt.pause(3)
    time.sleep(3)
    plt.show()

    i+=1
    if i >= n: break

# pickle with "batch"
joint_names = ['j'+str(j) for j in range(7)]
data_collect_wcs = pd.DataFrame(data_collect_wcs.reshape(-1,15), columns = joint_names + ['co_x', 'co_y', 'co_z', 'co_w', 'ct_x', 'ct_y', 'ct_z', 'ct_w'])
data_collect_real = pd.DataFrame(data_collect_real)
picklename = 'measurements/datawcs_robot_batch' + str(batch_number) + '.pkl'
data_collect_wcs.to_pickle(picklename)
data_collect_real.to_pickle(picklename.replace('wcs', 'orig'))
print('data pickled as', picklename)
