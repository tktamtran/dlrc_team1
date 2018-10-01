# This library helps with controlling the Franka Emika Panda. Used in
# in conjunction with Data:Lab's python controls
# TODO: also implement signal requests

import time
import py_at_broker as pab
import numpy as np
from pyquaternion import Quaternion


# TODO: create broker class instance that does not need to be passed

def initialize(addr_broker_ip="tcp://localhost:51468"):
    broker = pab.broker(addr_broker_ip)
    #broker.register_signal("franka_state", pab.MsgType.franka_state)
    broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
    broker.register_signal("franka_des_tau", pab.MsgType.des_tau)
    broker.request_signal("franka_state", pab.MsgType.franka_state, True)
    time.sleep(1)
    return broker


def set_new_pos(broker, new_pos, ctrl_mode=0, time_to_go=0.5):
    # substitute for static counter (for the fnumber)
    if not hasattr(set_new_pos, "fnumber"):
        broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
        set_new_pos.fnumber = 0  # it doesn't exist yet, so initialize it
    msg = pab.target_pos_msg()
    msg.set_timestamp(time.monotonic())
    msg.set_ctrl_t(ctrl_mode)  # 0:cartesian space, 1: joint space
    msg.set_fnumber(set_new_pos.fnumber)
    set_new_pos.fnumber += 1
    if type(new_pos) is not np.array:
        new_pos = np.asarray(new_pos)
    msg.set_pos(new_pos)
    msg.set_time_to_go(time_to_go)
    broker.send_msg("franka_target_pos", msg)


def set_zero_torques(broker):
    # substitute for static counter (for the fnumber)
    if not hasattr(set_zero_torques, "fnumber"):
        set_zero_torques.fnumber = 0  # it doesn't exist yet, so initialize it
        # broker.register_signal("franka_des_tau", pab.MsgType.des_tau)
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