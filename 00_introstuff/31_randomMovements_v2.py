import numpy as np
import dlrc_control as ctrl
import py_at_broker as pab
import time
import math

def get_state(msg):
    data = dict()
    data['timestamp'] = msg.get_timestamp()
    data['fnumber'] = msg.get_fnumber()
    data['flag_real_robot'] = msg.get_flag_real_robot()
    data['ndofs'] = msg.get_n_dofs()
    data['j_pos'] = msg.get_j_pos()
    data['j_vel'] = msg.get_j_vel()
    data['j_load'] = msg.get_j_load()
    data['last_cmd'] = msg.get_last_cmd()
    data['c_pos'] = msg.get_c_pos()
    data['c_vel'] = msg.get_c_vel()
    data['c_ori_quat'] = msg.get_c_ori_quat()
    data['dc_ori_quat'] = msg.get_dc_ori_quat()
    data['flag_gripper'] = msg.get_flag_gripper()
    data['gripper_state'] = msg.get_gripper_state()
    data['mass'] = msg.get_mass()
    data['coriolis'] = msg.get_coriolis()
    data['gravity'] = msg.get_gravity()
    data['flag_ready'] = msg.get_flag_ready()
    return data

def wait_for_new_pose(broker):
    ready = False
    time.sleep(0.2)
    while not ready:
        ctrl.wait_til_ready(broker)
        msg = broker.recv_msg("franka_state", -1)
        j_vel = msg.get_j_vel()
        print(f'Velocity norm: {math.degrees(np.linalg.norm(j_vel))}')
        # define low velocity if joints move by less than a degree per second
        low_velocity = math.degrees(np.linalg.norm(j_vel)) < 1
        if low_velocity:
            ready = True
            break  # beat a dead horse

broker = ctrl.initialize()
broker.request_signal("franka_state", pab.MsgType.franka_state)

np.set_printoptions(precision=2)

iteration = 0

while True:
    # generate new joint configuration, set it as the new target and wait until
    # the robot is ready again
    x_lims = np.array([-0.7,0.7])
    y_lims = np.array([-0.7,0.7])
    z_lims = np.array([0.15,0.9])
    random_q = ctrl.random_joint_config_constrained(x_lims, y_lims, z_lims)
    random_q = np.array(random_q[:7])
    print(f'Iteration {iteration}\tNew joint config: {random_q}')
    ctrl.set_new_pos(broker, random_q, ctrl_mode=1, time_to_go=5)
    wait_for_new_pose(broker)
    iteration+=1
s