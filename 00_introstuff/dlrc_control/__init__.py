# This library helps with controlling the Franka Emika Panda. Used in
# in conjunction with Data:Lab's python controls
# TODO: also implement signal requests

import time
import py_at_broker as pab
import numpy as np
from pyquaternion import Quaternion
import utils


# For the collision avoidance task, define collision avoidance exceptions
# this is nice as it enforces checks if a robot moved without collision
class CollisionException(Exception):
    pass


# TODO: create broker class instance that does not need to be passed

def initialize(addr_broker_ip="tcp://localhost:51468", realsense = False, lidar = False):
    broker = pab.broker(addr_broker_ip)
    #broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
    #broker.register_signal("franka_des_tau", pab.MsgType.des_tau)
    broker.request_signal("franka_state", pab.MsgType.franka_state, True)
    if realsense:
        broker.request_signal("realsense_images", pab.MsgType.realsense_image)
    if lidar:
        broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)
    time.sleep(1)
    return broker


def set_new_pos(broker, new_pos, ctrl_mode, time_to_go):
    # substitute for static counter (for the fnumber)
    if not hasattr(set_new_pos, "fnumber"):
        broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
        set_new_pos.fnumber = 0  # it doesn't exist yet, so initialize it
    msg = pab.target_pos_msg()
    msg.set_timestamp(time.monotonic())
    msg.set_ctrl_t(ctrl_mode)  # 0:cartesian space, 1:joint space
    msg.set_fnumber(set_new_pos.fnumber)
    #msg.set_fnumber(fnumber)
    set_new_pos.fnumber += 1
    if type(new_pos) is not np.array:
        new_pos = np.asarray(new_pos)
    msg.set_pos(new_pos)
    msg.set_time_to_go(time_to_go)
    broker.send_msg("franka_target_pos", msg)

def random_joint_config():
    """
    this creates a new random joint configuration for the robot so that we can explore the joint space effectively
    :param broker:      the broker to use
    :param time_to_go:  the time allowed to go to the new position
    """
    # The joint limits according to https://frankaemika.github.io/docs/control_parameters.html are
    q_limits = np.array([[-2.8973, 2.8973],
                         [-1.7628, 1.7628],
                         [-2.8973, 2.8973],
                         [-3.0718, -0.0698],
                         [-2.8973, 2.8973],
                         [-0.0175, 3.7525],
                         [-2.8973, 2.8973]])
    i = 0
    while True:  # only break from the while loop if a valid config was found
        rand_config = [np.random.uniform(q_limits[idx, 0], q_limits[idx, 1]) for idx in range(7)]
        # check if all joints (+ 10 centimeters for now) are in the positive world z coordinates
        _,_,_,_,T_list = utils.get_jointToCoordinates(rand_config)
        T_offsets = np.array([T[:3, 3] for T in T_list])
        #print(T_offsets)
        T_z = np.array([T[2] for T in T_offsets])
        i+=1
        if np.all(T_z > 0.005):
            # this means that every joint center is at least 10 centimeters above the table
            # so that is valid configuration for us and we can stop the loop
            print(rand_config, 'after', i, 'attempts')
            i = 0
            break

    # create random configs within the joint range and make sure that all joints stay above table
    # take into account that the joint centers are some centimeters within the robot shell
    return rand_config

def random_joint_config_constrained(xlim = np.array([-0.8, 0.8]), ylim = np.array([-0.8, 0.8]), zlim = np.array([0.1, 1])):
    """
    this creates a new random joint configuration for the robot so that we can explore the joint space effectively
    :param broker:      the broker to use
    :param time_to_go:  the time allowed to go to the new position
    """
    # The joint limits according to https://frankaemika.github.io/docs/control_parameters.html are
    xlim = sorted(xlim)
    ylim = sorted(ylim)
    zlim = sorted(zlim)
    q_limits = np.array([[-2.8973, 2.8973],
                         [-1.7628, 1.7628],
                         [-2.8973, 2.8973],
                         [-3.0718, -0.0698],
                         [-2.8973, 2.8973],
                         [-0.0175, 3.7525],
                         [-2.8973, 2.8973]])
    q_limits *= 0.97 # to avoid the absolute joint limits

    i = 0
    while True:  # only break from the while loop if a valid config was found
        rand_config = [np.random.uniform(q_limits[idx, 0], q_limits[idx, 1]) for idx in range(7)]
        # check if all joints (+ 10 centimeters for now) are within the limits
        _, _, _, _, T_list = utils.get_jointToCoordinates(rand_config)
        T_offsets = np.array([T[:3, 3] for T in T_list])
        # print(T_offsets

        # check if all joints are at least 10 cm apart from each other to avoid
        # the self-collision reflex
        # but do not take the directly nearest joint into account, as these
        # have fixed distances anyway
        T_distances = np.ones((T_offsets.shape[0], T_offsets.shape[0])) * np.inf
        for i in range(T_offsets.shape[0]):
            for j in range(T_offsets.shape[0]):
                if i == j:
                    # don't measure the distance to the joint itself
                    continue
                T_distances[i,j] = np.linalg.norm(T_offsets[i,:] - T_offsets[j,:])
        print(T_distances)
        # the relevant distances are stored in the lower triangular part that
        # excludes the main diagonal and first off diagonal
        T_dist = np.ones((T_offsets.shape[0], T_offsets.shape[0])) * np.inf
        # lower triangular indices
        tril_indices = np.tril_indices(T_distances.shape[0], -3)
        T_dist[tril_indices] = T_distances[tril_indices]
        print(T_dist)
        print((T_dist > 0.1).all())


        T_x = np.array([T[0] for T in T_offsets])
        T_y = np.array([T[1] for T in T_offsets])
        T_z = np.array([T[2] for T in T_offsets])
        i += 1
        # check if all joints are within the specified limits in that config
        if (xlim[0] < T_x).all() and (T_x < xlim[1]).all():
            # print('X ok')
            if np.all(ylim[0] < T_y).all() and (T_y < ylim[1]).all():
                # print('Y ok')
                if np.all(zlim[0] < T_z).all() and (T_z < zlim[1]).all():
                    # print('Z ok')
                    # print(rand_config, 'after', i, 'attempts')
                    i = 0
                    break

    # create random configs within the joint range and make sure that all joints stay above table
    # take into account that the joint centers are some centimeters within the robot shell
    return rand_config


def gen_joint_configs(initial, n_configs):
    '''generate valid joint configs from an initial valid joint config'''

    q_limits = np.array([[-2.8973, 2.8973],
                         [-1.7628, 1.7628],
                         [-2.8973, 2.8973],
                         [-3.0718, -0.0698],
                         [-2.8973, 2.8973],
                         [-0.0175, 3.7525],
                         [-2.8973, 2.8973]])

    configs = np.empty((n_configs,7))
    configs[0,:] = initial

    i = 1
    while i < n_configs:
        configs[i, :] = configs[i-1,:]
        configs[i, i%7] = float("%.2f" % utils.get_truncated_normal(mean=configs[i,i%7], sd=1, low=q_limits[i%7, 0], upp=q_limits[i%7, 1]).rvs())

        _,_,_,_,Tlist = utils.get_jointToCoordinates(configs[i,:])
        T_offset = np.array(Tlist)[:,:3,3]
        T_z = np.array(Tlist)[:,2,3]
        if (T_z > 0.005).all(): i+=1

    return configs


def move_basic_ca(broker, new_pos, ctrl_mode=0, time_to_go=0.5):
    set_new_pos(broker, new_pos, ctrl_mode, time_to_go)
    wait_til_ready_ca(broker)


def set_zero_torques(broker):
    # substitute for static counter (for the fnumber)
    if not hasattr(set_zero_torques, "fnumber"):
        set_zero_torques.fnumber = 0  # it doesn't exist yet, so initialize it
        broker.register_signal("franka_des_tau", pab.MsgType.des_tau)
    msg = pab.des_tau_msg()
    msg.set_timestamp(time.monotonic())
    msg.set_fnumber(set_zero_torques.fnumber)
    set_zero_torques.fnumber += 1
    msg.set_j_torque_des(np.zeros(7))
    broker.send_msg("franka_des_tau", msg)


def wait_til_ready(broker):
    while True:
        msg = broker.recv_msg("franka_state", -1)
        if msg.get_flag_ready():
            break

def wait_til_ready_ca(broker):
    '''
    wait_til_ready_ca waits for the robot to perform the current movement but commands the robot to stop if any of the
    sensors sense something too close.
    This command is used after issuing a new position command. So wait_til_ready_ca will stop and raise a
    CollisionException on an impending collision. If that does not happen, it will block until the robot signals that it
    is ready to receive a new command
    :param broker: the broker to use
    :return:
    '''
    nothing_is_close = True
    while nothing_is_close: # in this basic case equivalent to 'while True' as an exception will be called if false
        state_msg = broker.recv_msg("franka_state", -1)
        lidar_msg = broker.recv_msg("franka_lidar", -1)
        #realsense_msg = broker.recv_msg("realsense_images", -1)
        last_j_pos = state_msg.get_j_pos()
        lidar_distances = lidar_msg.get_data() / 1000
        # TODO: wrapper around lidar readings that filters invalid results in
        # the same way for every call to lidar readings, so that we don't access
        # the raw values every time
        lidar_distances = np.array([d if 0 < d < 2 else np.nan
                           for d in lidar_distances])
        # TODO: remove the next line after debugging
        lidar_distances[2] = np.nan #JUST FOR DEBUGGING
        #realsense_distances = realsense_msg.get_depth()
        #realsense_distances = realsense_distances.reshape(realsense_msg.get_shape_depth())
        # strip first row of distances as first line only contains zeros
        #realsense_distances = realsense_distances[1:, :] / 1000
        min_distance = 0.06  # readings are given in meters!
        #realsense_too_close = (realsense_distances < min_distance).any()
        lidar_too_close = (lidar_distances < min_distance).any()
        if  lidar_too_close:# or realsense_too_close:
            # something is too close!
            nothing_is_close = False
            lidar_idx = np.flatnonzero(lidar_distances < min_distance)
            #realsense_idx = np.where(realsense_distances < min_distance)
            # set the robot to keep last known joint position
            # ctrl_mode = 1 signifies joint space control
            set_new_pos(broker, last_j_pos, ctrl_mode=1, time_to_go=1)
            raise CollisionException(f"Collision avoidance detected something too close\n"
                                     f"Lidars {lidar_idx}\n"
                                     f"Lidar readings {lidar_distances}\n")
                                     #f"Image pixels {realsense_idx}")
        msg = broker.recv_msg("franka_state", -1)
        if msg.get_flag_ready():
            break


def check_if_path_free(broker):
    '''
    check_if_path_free blocks until every sensor allows movement (no sensor reads
    something close
    :param broker: the broker to use
    :return:
    '''
    something_is_close = True
    while something_is_close:
        lidar_msg = broker.recv_msg("franka_lidar", -1)
        #realsense_msg = broker.recv_msg("realsense_images", -1)
        lidar_distances = lidar_msg.get_data() / 1000
        # TODO: wrapper around lidar readings that filters invalid results in
        # the same way for every call to lidar readings, so that we don't access
        # the raw values every time
        lidar_distances = np.array([d if 0 < d < 2 else np.nan
                           for d in lidar_distances])
        # TODO: remove the next line after debugging
        lidar_distances[2] = np.nan #JUST FOR DEBUGGING
        #realsense_distances = realsense_msg.get_depth()
        #realsense_distances = realsense_distances.reshape(realsense_msg.get_shape_depth())
        # strip first row of distances as first line only contains zeros
        #realsense_distances = realsense_distances[1:, :] / 1000
        min_distance = 0.06  # readings are given in meters!
        #realsense_too_close = (realsense_distances < min_distance).any()
        lidar_too_close = (lidar_distances < min_distance).any()
        if  lidar_too_close:# or realsense_too_close:
            # something is too close!
            something_is_close = True
            lidar_idx = np.flatnonzero(lidar_distances < min_distance)
            #realsense_idx = np.where(realsense_distances < min_distance)
            print('Cannot move! Something is too close')
        else:
            # nothing is close
            # we can move again
            print("Can move again")
            something_is_close = False


def move_straight(broker, start_pos, target_pos, **kwargs):
    '''
    move_straight moves the robot in a straight line
    interpolating the positions in between

    Kwargs:
        num_steps:      the number of steps to interpolate between
                        (default 10)
        max_stepwidth:  the maximum euclidean distance between two steps
                        (default 0.05)
        by default, the one requiring more steps will be used
        time_to_go:     the available time to go to target
                        (default 0.5)
    '''
    # check the dimensionality of the positions and verify it's either 3 or 7
    if (np.asarray(start_pos).size == 7 or np.asarray(start_pos).size == 3) \
            and \
            (np.asarray(target_pos).size == 7 or np.asarray(target_pos).size == 3) \
            and \
            (np.asarray(start_pos).size == np.asarray(target_pos).size):
        # all parameters seem valid
        pass
    else:
        raise ValueError("The input positions need to be the same size and have"
                         "either 3 or 7 components")
    # check how many steps we need to take
    if 'num_steps' in kwargs:
        num_steps = kwargs['num_steps']
    else:
        num_steps = 10
    if 'max_stepwidth' in kwargs:
        max_stepwidth = kwargs['max_stepwidth']
    else:
        max_stepwidth = 0.05
    if 'time_to_go' in kwargs:
        time_to_go = kwargs['time_to_go']
    else:
        time_to_go = 0.5

    # calculate the offset/displacement between start and target
    displacement = target_pos - start_pos
    distance = np.linalg.norm(displacement[0:3])  # consider only position
    # calculate the number of steps by choosing whichever value needs more
    num_steps = np.amax(np.asarray([
        np.ceil(distance / max_stepwidth),
        num_steps]))
    num_steps = np.int64(num_steps)

    for step in range(num_steps):
        fraction = step / num_steps
        new_xyz = start_pos[0:3] + fraction * displacement[0:3]
        if start_pos.size == 7 and target_pos.size == 7:
            new_quat = Quaternion.slerp(
                Quaternion(start_pos[3:]),
                Quaternion(target_pos[3:]),
                fraction)
            new_pos = np.concatenate([new_xyz, np.asarray(new_quat.q)])
        else:
            new_pos = new_xyz
        set_new_pos(broker, new_pos, ctrl_mode=0, time_to_go=time_to_go)
        wait_til_ready(broker)


def look_at(xyz_stand, xyz_target, up=np.asarray([0,0,1])):
    '''
    look_at receives a cart. position of the end effector and a cart. position
    of a target to look at
    :param xyz_stand:       the xyz position for the end effector to stand at
    :param xyz_target:      the xyz position of the target to look at
    :param parallel_plane:  a reference plane for the EE orientation as normal
                            vector
    :return:                robot pose oriented at target
    '''
    # the displacement vector looks from the stand point to the target point
    # TODO: increase robustness by checking for collinearity and zero
    # code taken from
    # https://stackoverflow.com/questions/18558910/direction-vector-to-rotation-matrix
    # end effector position
    cam_pos = xyz_stand  # end effector position
    look_at = xyz_target  # look at position
    vector_to = cam_pos - look_at
    zero_vec = np.array([0, 0, -1])  # end effector zero position

    # math to find quaternion between two vectors (https://www.gamedev.net/forums/topic/429507-finding-the-quaternion-betwee-two-vectors/)
    c = np.cross(zero_vec, vector_to)

    w = np.sqrt((zero_vec[0] ** 2 + zero_vec[1] ** 2 + zero_vec[2] ** 2) * (
                vector_to[0] ** 2 + vector_to[1] ** 2 + vector_to[2] ** 2)) + np.dot(zero_vec, vector_to)

    q = np.append(w, c)  # quaternion

    # pos_and_ori = np.append(cam_pos, q)  # to send in the message

    return Quaternion(q)


def project_to_plane(vector, plane_normal):
    # equation for parallel component in plane from
    # http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/index.htm
    projected = np.cross(plane_normal, np.cross(vector, plane_normal))
    projected = projected/(np.linalg.norm(plane_normal)**2)
    return projected

def smooth_filter_hampel(vals_orig, k=7, t0=3):
    '''
    this so-called hampel filter filters the data in a nonlinear way: if the
    value under consideration is outside the 3-sigma range in the local
    neighbourhood k, then it is replaced with the median of those 7 values, but
    unchanged if it is within the 3-sigma range
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    return(vals)

