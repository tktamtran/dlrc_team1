import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load and construct the necessary data
# lidar 0
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1203550000.pkl"))

# lidar 1 on joint 5
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1223410000.pkl"))
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1247570000.pkl")) # at z=-0.058

# lidar 2 on joint 5
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1249030000.pkl"))

# lidar 3 on joint 5
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1408000000.pkl")) # at z=-0.058

# lidar 4 on joint 5
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1410000000.pkl"))

#lidar 5 on joint 4
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1413110000.pkl"))

#lidar 6 on joint 3
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1414480000.pkl"))

# lidar 7 on joint 3
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1419370000.pkl")) # at approximately x=-0.6

#lidar 8 on joint 3
#data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180925T1426080000.pkl"))


sensor_type = 'camera'
joint_idx = 8
known_dim = 'z'
known_value = 0 #-0.058


if sensor_type == 'camera':
    T_index = 0 #0 T0ee, 2 T0joint
    principal_point = (240//2, 320//2)
    data = pd.DataFrame(pd.read_pickle(r"/home/dlrc1/measurements/20180926T0812290000.pkl"))
    depth_images = data.realsense_depthdata
    lidar = [depth_images[i][principal_point[0], principal_point[1]]/1000 for i in range(len(depth_images))]
    plt.plot(lidar)
    plt.show()

elif sensor_type == 'lidar':
    lidar_idx = 7
    T_index = 2 # T0joint
    rawlidar = [d[lidar_idx]/1000 for d in data.lidar_data]
    medlidar = scipy.signal.medfilt(rawlidar, 5)
    plt.plot(medlidar)
    plt.show()
    lidar = medlidar


thetas = [jp for jp in data.state_j_pos]
# the right joint is joint 6
T = [get_jointToCoordinates(theta, untilJoint=joint_idx)[T_index] for theta in thetas]
Rt = [t[0:3, 0:3].T for t in T]
P = [t[0:3,3] for t in T]

# to solve the linear problem A * x = b, we need to combine the data in the
# right fashion: A = (I, reading*I, -R, 0, 0, ....)

if sensor_type == 'lidar':
    A = np.tile(np.eye(3), (len(data),1)) # first three columns: I
    sI = np.tile(np.repeat(lidar, 3), (3,1)).T* A
    A = np.hstack((A, sI)) # next three: s*I
    A = np.hstack((A, -scipy.linalg.block_diag(*Rt)))
    b = -scipy.linalg.block_diag(*Rt) @ np.hstack(P)

elif sensor_type == 'camera':
    A = np.tile(np.eye(3), (len(data), 1))  # first three columns: I
    A = np.hstack((A, -scipy.linalg.block_diag(*Rt)))
    b = -scipy.linalg.block_diag(*Rt) @ np.hstack(P) - np.concatenate(tuple([[0,0,lid] for lid in lidar]), axis=0)

# we know the z component of every measurement to be zero, so we can eliminate
# the corresponding cols of A and rows of x while b stays the same
# their indices are deduced by: 0..5 are fixed identity, then follow the
# rotation matrices, so their indices are (5+3i)

# modification: enabling separation of known and unknown parts of matrix to be beyond z=0
dim_idx = {'x':0, 'y':1, 'z':2}
known_dim = dim_idx[known_dim]
known_idx = np.arange(6+known_dim, A.shape[0], 3)
A_unknown = np.delete(A, known_idx, 1)
b_unknown = b - np.dot(A[:,known_idx], np.ones(len(known_idx))*known_value)
x_unknown, residuals, rank, singval = np.linalg.lstsq(A_unknown,b_unknown)

x = x_unknown
T_lidar = np.array(np.hstack((np.zeros((3,2)), 
                              np.reshape(x[3:6], (3,1)), 
                              np.reshape(x[ :3], (3,1)))))
T_lidar = np.array(np.vstack((T_lidar,np.array([0,0,0,1]))))
# first three elements are offset in meters, second three are direction
# direction should have norm 1
norm_rotation = np.linalg.norm(x[3:6])
print(x[:6])
print(f'Linear least squares suggests an offset of {x[:3]} (in meters, xyz)')
print(f'and rotation of {x[3:6]} (norm: {norm_rotation})')
print(f'Residual norm: {residuals}')
print('sum of error', sum(abs(np.dot(A_unknown, x_unknown) - b_unknown)))

# per-datapoint residuals for the translation and rotation of the lidar:
xy = x[6:]
xidx = np.arange(0,len(xy), 2)
yidx = np.arange(1, len(xy),2)
#plt.scatter(xy[xidx], xy[yidx], c=np.arange(len(xy[xidx])))

# write data to file and animate
table_xy = np.hstack((xy[xidx], xy[yidx]))
joint_origin = P
joint_direction = [R.T for R in Rt]
base_lidar_T = [Ti @ T_lidar for Ti in T]
base_origin = [bl_Ti[:3,3] for bl_Ti in base_lidar_T]
base_z = [bl_Ti[:3,:3] @ np.reshape(np.array([0,0,1]), (3,1))
          for bl_Ti in base_lidar_T]

#vizdata = dict()
#vizdata['table_xy'] = table_xy
#vizdata['joint_origin'] = joint_origin
#vizdata['joint_direction'] = joint_direction
#vizdata['base_lidar_T'] = base_lidar_T
#vizdata['base_origin'] = base_origin
#vizdata['base_z'] = base_z
#
#with open("LidarViz.pkl", "wb") as f:
#    pickle.dump(vizdata, f)



# recalculate for x_unknown, ignoring data outliers
xr = x_unknown
Ar = A_unknown
br = b_unknown
if abs(1.00 - norm_rotation) > 0.005:

    error = np.dot(Ar, xr) - br
    #assert(np.isclose(np.zeros(6), error[:6], atol=1e-3).all())
    print('x[:6]', x[:6])
    std_error, mean_error = np.std(error), np.mean(error)
    print('std_error', std_error)
    outlier_idx = np.array(np.where(abs(error-mean_error) > std_error*1))
    print('')
    print('percent that were outlier', outlier_idx.shape[1], len(error), float(outlier_idx.shape[1] / len(error)))

    Ar = np.delete(Ar, outlier_idx, axis=0)
    br = np.delete(br, outlier_idx, axis=0)
    xr, residuals, rank, singval = np.linalg.lstsq(Ar, br)
    norm_rotation = np.linalg.norm(xr[3:6])

    print(xr[:6])
    print(f'Linear least squares suggests an offset of {xr[:3]} (in meters, xyz)')
    print(f'and rotation of {xr[3:6]} (norm: {norm_rotation})')
    print(f'Residual norm: {residuals}')
    print('sum of error', sum(abs(np.dot(Ar,xr) - br)))

