import sys
sys.path.insert(0, '/home/dlrc1/Desktop/00_introstuff')
from utils import *


#from home.dlrc1.Desktop.00_introstuff.utils import *


directory_data = 'data_robot/'
database = pd.read_pickle(directory_data + 'datawcs_robot_batches0020.pkl')

# set up program mode and data import
argparser = argparse.ArgumentParser()
argparser.add_argument("-mr", "--mode_realtime",
                       help="Generate WCS model in realtime",
                       action="store_true")
argparser.add_argument("-md", "--mode_dataset",
                       help="Generate WCS model from a dataset",
                       action="store_true")
argparser.add_argument("-l", "--lidar",
                       help="Include lidar readings in data collection and plotting",
                       action="store_true")
argparser.add_argument("-d", "--detail",
                       help="Plot camera reconstruction with additional detail of histograms",
                       action="store_true")
args = argparser.parse_args()
assert(args.mode_realtime + args.mode_dataset == 1)



# functions for hacky classification
# TODO maybe introduce some padding, to account for msmt noise
from scipy.optimize import linprog
def hacky_clf_linear(points_positive, points_test):
    n_cloud = points_positive.shape[0]
    n_dim = points_positive.shape[1]
    c = np.zeros(n_cloud)
    A = np.r_[points_positive.T, np.ones((1,n_cloud))]
    points_pred = [linprog(c, A_eq=A, b_eq=np.r_[p, np.ones(1)]).success for p in points_test]
    return points_pred

from scipy.spatial import Delaunay
def hacky_clf_hull(points_positive, points_test):
    # points_positive[:,:2] *= 0.9 # padding / expansion of hull
    # points_positive[:,2] -= 0.05
    if not isinstance(points_positive, Delaunay): points_positive = Delaunay(points_positive)
    points_pred = points_positive.find_simplex(points_test)>=0
    return points_pred


def get_points_at_joint(database, query_joint):
    # be flexible with the joint values (round, bin)
    # be conscious about until which joint matters

    db_matches = (np.abs(database['j1'].values - query_joint[0]) < 0.5) * \
                 (np.abs(database['j2'].values - query_joint[1]) < 0.5) * \
                 (np.abs(database['j3'].values - query_joint[2]) < 0.5) * \
                 (np.abs(database['j4'].values - query_joint[3]) < 0.5)
                 # (np.abs(database['j5'].values - query_joint[4]) < 0.5) * \
                 # (np.abs(database['j6'].values - query_joint[5]) < 0.5) * \
                 # (np.abs(database['j7'].values - query_joint[6]) < 0.5)
    idx_matches = np.argwhere(db_matches).squeeze(axis=1)
    print('n_wcs_matches', idx_matches.shape)
    #print('n_joint_matches')  # TODO

    cols_wcp = ['ct_x', 'ct_y', 'ct_z']
    wcp_joint = database.iloc[idx_matches.tolist()][cols_wcp].values
    return wcp_joint


def clf_and_plot(rgb, depth, joint, Teecamera, database, min_positive, frame_number):

    # map (depth,joint) -> wcp
    # conversion can be of diff resolution than the database's point cloud resolution
    ccs_points, rgb_colors, pp_ccs_point = img_to_ccs(depth_image=depth, principal_point=(240//2, 320//2), camera_resolution=(240,320), skip=7, rgb_image=rgb)
    T0ee,_,_,_,_ = get_jointToCoordinates(thetas=joint)
    T0cam = np.dot(T0ee, Teecamera)
    wcs_points = np.array([np.dot(T0cam, ccs_point) for ccs_point in ccs_points])[:,:3]
    #wcs_points2 = np.tensordot()

    # gather positive wcp at that joint config
    wcp_joint = get_points_at_joint(database,joint)

    ax1.clear()
    ax1.set_title('frame#' + str(frame_number))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_zlim(-0.1, 1)
    if wcp_joint.shape[0] > min_positive:
        # generate predictions
        points_pred = hacky_clf_hull(points_positive=wcp_joint, points_test=wcs_points)
        idx_pos = np.argwhere(points_pred).squeeze(axis=1)
        print('positive points', len(idx_pos), 'negative points', len(points_pred)-len(idx_pos))

        # TODO incorporate rgb colors
        ax1.scatter(wcs_points[idx_pos, 0], wcs_points[idx_pos, 1], wcs_points[idx_pos, 2], s=15, c=rgb_colors[idx_pos,:], alpha=1, edgecolors='green', linewidths=0.3)
        ax1.scatter(wcs_points[~idx_pos, 0], wcs_points[~idx_pos, 1], wcs_points[~idx_pos, 2], s=15, c=rgb_colors[~idx_pos,:], alpha=1, edgecolors='red', linewidths=0.3)

    else:
        ax1.scatter([0],[0],[0], c='blue') # TODO some creative warning that current joint position not familiar/insufficient prior data

    ax2.clear()
    im = ax2.imshow(depth, origin='lower')
    plt.colorbar(im, cax=ax2)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.clear()
    ax3.imshow(rgb)
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_ylim(ax3.get_ylim()[::-1])



if args.mode_realtime:
    # connect to data stream
    broker = pab.broker("tcp://localhost:51468")
    broker.request_signal("franka_state", pab.MsgType.franka_state)
    broker.request_signal("realsense_images", pab.MsgType.realsense_image)
    n = 500 # limit on #frames for real time

if args.mode_dataset:
    filename = "/home/dlrc1/Desktop/00_introstuff/measurements/dataorig_robot_batch_rt20.pkl"
    data = pd.DataFrame(pd.read_pickle(filename))
    data = data.rename(columns={"realsense_depth": "realsense_depthdata"})
    data.reset_index(inplace=True)
    n = data.shape[0]

plt.ion()
fig = plt.figure()
subplot_row, subplot_column = 2,2
ax1 = fig.add_subplot(subplot_row, subplot_column, 1, projection='3d')
ax2 = fig.add_subplot(subplot_row, subplot_column, 3)
ax3 = fig.add_subplot(subplot_row, subplot_column, 4)


i=0
while (i<n):

    # gather data based on mode
    if args.mode_realtime:
        state_msg = broker.recv_msg("franka_state", -1)
        joint = state_msg.get_j_pos()
        if args.lidar: lidar_readings = broker.recv_msg("franka_lidar", -1).get_data()[:]/1000
        rgb, depth = grab_image(broker)

    if args.mode_dataset:
        joint = data.iloc[i]['state_j_pos']
        if args.lidar: lidar_readings = data.iloc[i]['lidar_data']/1000
        rgb = data.iloc[i]['realsense_rgbdata']
        depth = data.iloc[i]['realsense_depthdata']

    # processing one frame at a time
    clf_and_plot(rgb, depth, joint, Teecamera, database, min_positive=50, frame_number=i)
    time.sleep(0.05)
    fig.canvas.draw()
    plt.pause(0.05)
    time.sleep(0.05)
    plt.show()
    i+=1
