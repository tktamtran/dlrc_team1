# make a composed reconstruction of image data to wcs

# get full scope data, ie at all joint config combos, of both camera and lidar data
# make sure nothing foreign is in environment of one meter on all sides

from utils import *

data = pd.DataFrame(pd.read_pickle("/home/dlrc1/measurements/20180928T1329150000.pkl"))[40:]
#data = pd.DataFrame(pd.read_pickle("/home/dlrc1/measurements/17_cameracalibration.pkl"))[20:]
data.reset_index(inplace=True)
depth_images = data['realsense_depthdata']
rgb_images = data['realsense_rgbdata']
joint_configs = data['state_j_pos']

subplot_row, subplot_column = 2,2
n = data.shape[0]
print('n',n)



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



wcs_plotted = np.empty((0,4))# to avoid too many overlaps in plot; holds the rounded coordinates rather than calc'd coords
for i in range(n):

    T0ee, _, _, _ = get_jointToCoordinates(thetas=joint_configs.iloc[i])
    T0cam = np.dot(T0ee, Teecamera)
    ccs_points, point_colors = img_to_ccs(depth_images[i], principal_point, camera_resolution, skip=5, rgb_image=rgb_images[i])
    wcs_points = np.array([np.dot(T0cam, ccs_point) for ccs_point in ccs_points]) # still homogeneous
    wcs_points = np.round(wcs_points, 2)

    mask_to_plot = np.in1d(wcs_points[:,0], wcs_plotted[:,0]) * np.in1d(wcs_points[:,1], wcs_plotted[:,1]) * np.in1d(wcs_points[:,2], wcs_plotted[:,2])
    mask_to_plot = ~mask_to_plot
    wcs_to_plot = wcs_points[np.where(mask_to_plot)[0],:]
    colors_to_plot = point_colors[np.where(mask_to_plot)[0],:]
    wcs_plotted = np.append(wcs_plotted, wcs_to_plot, axis=0)

    ax1.scatter(wcs_to_plot[:, 0], wcs_to_plot[:, 1], wcs_to_plot[:, 2], s=10, c=colors_to_plot/255, alpha=0.5)
    ax1.set_title('image frame # ' + str(i) + ' with ' + str(sum(mask_to_plot)) + ' new points plotted')

    im = ax2.imshow(depth_images[i], origin='lower')
    plt.colorbar(im, cax=ax2)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.imshow(rgb_images[i])
    #ax3.set_xlim(ax3.get_xlim()[::-1])
    #ax3.set_ylim(ax3.get_ylim()[::-1])

    fig.canvas.draw()
    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()