import numpy as np
import py_at_broker as pab
import SLRobot
import time
# import signal
# import sys
import time
import os

# import matplotlib.pyplot as plt
# matplotlib inline
import pickle

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def quaternion_mult(q, r):
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def point_rotation_by_quaternion(point, q):
    r = [0] + point
    q_conj = [q[0], -1 * q[1], -1 * q[2], -1 * q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


def prioritization(dq, ddq, ddx, J, dJ, sigma_ddq, sigma_ddx):
    # prioritization(dq, ddq, ddx, J, dJ)
    # q: joints
    # x: end-effector
    # d: velocity
    # dd: acceleration
    # J: Jacobian
    # alpha_joints = 0.01
    # alpha_EF = 0.1
    # num_joints = 7
    # num_ef = 3
    # sigma_ddq =  alpha_joints*np.identity(num_joints)
    # sigma_ddq=np.diag([50, 50, 25, 10, 10, 5, 1])
    # sigma_ddx =  alpha_EF*np.identity(num_ef)

    JT = np.transpose(J)
    J_left = np.dot(sigma_ddq, JT)
    J_right = np.linalg.solve(sigma_ddx + np.dot(np.dot(J, sigma_ddq), JT) + J_reg * np.eye(3), np.eye(3))
    J_star = np.dot(J_left, J_right)
    # it is more accurate to include dJ:
    mu_ddq_t = np.dot(J_star, ddx - np.dot(dJ, dq)) + np.dot(np.identity(7) - np.dot(J_star, J), ddq)
    # mu_ddq_t = np.dot(J_star, ddx) + np.dot(np.identity(num_joints)-np.dot(J_star, J), ddq)
    sigma_ddq_t = (np.dot((np.identity(7) - np.dot(J_star, J)), sigma_ddq)
                   + np.dot(np.dot(J_star, sigma_ddx), J_star.transpose()))
    return mu_ddq_t, sigma_ddq_t


def obstacle_force(p, p0, sigma_):
    f = np.array([0.0, 0.0, 0.0])
    mask_ = (p < p0) & (p > -p0)
    if (np.any(p[mask_] == 0)):
        p[mask_] = p[mask_] + 0.00001

    f[mask_] = -sigma_ * (1.0 / p[mask_] - 1.0 / p0[mask_])
    return f


## Init Variables
# Control gains
pgain_null = 0.00125 * np.array([600.0, 600.0, 600.0, 600.0, 250.0, 80.0, 50.0], dtype=np.float64)
# dgain_null = 10*0.5 / 8.0 * pgain_null
dgain_null = 10 * 0.5 / 8.0 * pgain_null * 5
# End-Eff gain
ku = 50 * np.array([0.4, 1.5, 1.4], dtype=np.float64)
pgain_ee = 0.8 * ku
dgain_ee = 0.8 * ku / 4.0
# Torque limits
rate_limit = 0.6  # Rate torque limit original: 0.6
# torque_limit = np.array([80, 80, 80, 80, 10, 10, 10], dtype=np.float64) # Absolute torque limit
torque_limit = np.array([3, 3, 3, 3, 2, 2, 2], dtype=np.float64)  # Absolute torque limit
alpha = 0.01  # Exp. interpolation constant
J_reg = 1e-3  # Jacobian regularization constant
# Null-space theta configuration
target_th_null = np.zeros(7, dtype=np.float64)
target_th_null[3] = -1.55
target_th_null[5] = 1.9
# Max deviation between current and desired Cart pos (to avoid integration to inf)
maxCartDev = 0.2  # in m
max_sync_jitter = 0.2
use_inv_dyn = False
robot_name = "franka"  # for use in signals

try:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(35))
except OSError as err:
    print("Error: Failed to set proc. to real-time scheduler\n{0}".format(err))

## Register Signals
b = pab.broker()
b.register_signal(robot_name + "_des_tau", pab.MsgType.des_tau)
# b.request_signal("task_policy", pab.MsgType.task_policy, True)
b.request_signal(robot_name + "_state", pab.MsgType.franka_state, True)
b_lidar = pab.broker()
print("Before Lidar")
b_lidar.request_signal("franka_lidar", pab.MsgType.franka_lidar, -1)
print("Setup completed. Starting...")
# set maximum distances
p0 = np.array([0.3, 0.3, 0.3])
sigma_ddq_null = 0.5 * np.identity(7)
sigma_ddx_ee = 0.3 * np.identity(3)
# space mouse first
counter = 0  # Frame number
initCounter = 0  # For interpolating to target

# Moving
while True:

    if counter == 0 or initCounter == 1:
        msg_panda = b.recv_msg(robot_name + "_state", -1)
        msg_lidar = b_lidar.recv_msg("franka_lidar", -1)

    # msg_task = b.recv_msg("task_policy", -1)
    msg_panda = b.recv_msg(robot_name + "_state", 0)
    msg_lidar = b_lidar.recv_msg("franka_lidar", 0)

    recv_stop = time.clock_gettime(time.CLOCK_MONOTONIC)  # TODO

    current_c_pos = msg_panda.get_c_pos()
    current_c_vel = msg_panda.get_c_vel()
    current_j_vel = msg_panda.get_j_vel()
    current_j_pos = msg_panda.get_j_pos()

    lidar_data = msg_lidar.get_data()
    print(lidar_data)
    initial_pos = np.array([0.628, -0.002, 0.497])
    # target_c_vel = np.array([0.001,0.,0.]) #msg_task.get_c_acc_mean() * 0.1  # TODO magic scaling of input pos
    target_c_pos = initial_pos + np.array([0, -0.2, 0])
    # P control towards the target
    target_c_vel = 0.01 * (target_c_pos - current_c_pos)

    # Init after reset
    if initCounter == 0:
        last_trig = recv_stop  # TODO check the last_trig
        target_c_pos = current_c_pos
        last_c_vel = current_c_vel

    # Integration of des. vel. for generating des. position
    dt = 0.01  # recv_stop - last_trig
    target_c_pos = target_c_pos + target_c_vel * dt  # TODO copy vWall and clip
    # target_c_pos = np.clip(target_c_pos, current_c_pos - maxCartDev, current_c_pos + maxCartDev)

    # Interpolate to the null space to avoid torque jumps
    if initCounter < 50:
        target_out = current_c_pos  # combination
        target_th_null_out = current_j_pos
    elif initCounter < 550:
        target_out = (1 - alpha) * target_out + alpha * target_c_pos  # combination
        target_th_null_out = (1 - alpha) * target_th_null_out + alpha * target_th_null
    else:
        target_out = target_c_pos  # combination
        target_th_null_out = target_th_null

    J = SLRobot.Jacobian(np.array([current_j_pos]), 6)  # index of the end-eff link
    J = J[:3]

    qd_null = pgain_null * (target_th_null_out - current_j_pos) - dgain_null * current_j_vel
    x_dd = pgain_ee * (target_out - current_c_pos) - dgain_ee * current_c_vel
    target_j_acc, target_j_sigma = prioritization(dq=np.array([0] * 7),
                                                  ddq=qd_null,
                                                  ddx=x_dd,
                                                  J=J,
                                                  dJ=np.zeros(J.shape),
                                                  sigma_ddx=sigma_ddx_ee,
                                                  sigma_ddq=sigma_ddq_null)

    # TODO joint 4 --> lidar
    # first obstacle
    # n_obs_link = 3  # the index of the obstacle link, link 4: joint 6; link 3: joint 4.
    # J = SLRobot.Jacobian(np.array([current_j_pos]), n_obs_link)
    # J = J[:3]
    #
    # FK_ = SLRobot.FK(np.array([current_j_pos]))
    # quat_ = FK_[0, (n_obs_link - 1) * 10 + 6:n_obs_link * 10]
    #
    # # acc from lidar
    # obs_j4 = np.ones([3])*10000.0  # defaul: very far away
    # obs_j4[0] = -1.0*lidar_data[7]*0.001
    # #obs_j4[2] = 1.0*lidar_data[6]*0.001
    # #print('lidar:', lidar_data[0])
    #
    # x_dd_t = obstacle_force(obs_j4, p0, sigma_=1.0) * 0.1 *0.3
    # x_dd = np.array(point_rotation_by_quaternion(x_dd_t.tolist(), quat_.tolist()))
    #
    # alpha_x = 0.1 *0.15 # 5.0 more SP, 0.1 more lidar # TODO this should be removed
    # sigma_ddx = alpha_x*np.identity(3)
    #
    # # print("OA x_dd : {}".format(x_dd))
    #
    # target_j_acc, target_j_sigma = prioritization(dq=np.array([0] * 7),
    #                                               ddq=target_j_acc,
    #                                               ddx=x_dd,
    #                                               J=J,
    #                                               dJ=np.zeros(J.shape),
    #                                               sigma_ddx=sigma_ddx,
    #                                               sigma_ddq=target_j_sigma)

    # second obstacle
    # TODO joint 6 --> lidar 0,1,3,4
    n_obs_link = 4  # the index of the obstacle link, link 4: joint 6; link 3: joint 4.
    J = SLRobot.Jacobian(np.array([current_j_pos]), n_obs_link)
    J = J[:3]

    FK_ = SLRobot.FK(np.array([current_j_pos]))
    quat_ = FK_[0, (n_obs_link - 1) * 10 + 6:n_obs_link * 10]

    # acc from lidar
    obs_j6 = np.ones([3]) * 10000.0  # defaul: very far away
    # lidar 0 and 4 look in opposite directions along y-axis, only consider the closer lidar reading
    if (lidar_data[0] < lidar_data[4]):
        obs_j6[1] = 1.0 * lidar_data[0] * 0.001
    else:
        obs_j6[1] = -1.0 * lidar_data[4] * 0.001
    # lidar 1 and 3 look in opposite directions along z-axis, only consider the closer lidar reading
    if (lidar_data[1] < lidar_data[3]):
        obs_j6[2] = 1.0 * lidar_data[1] * 0.001
    else:
        obs_j6[2] = -1.0 * lidar_data[3] * 0.001
    # obs_j6[0] = -1.0*lidar_data[2]*0.001  # lidar 2 too close to self
    # print(lidar_data[10])

    x_dd_t = obstacle_force(obs_j6, p0, sigma_=1.0) * 0.6# 0.1
    x_dd = np.array(point_rotation_by_quaternion(x_dd_t.tolist(), quat_.tolist()))

    alpha_x = 0.7  # 5.0 more SP, 0.1 more lidar # TODO this should be removed
    sigma_ddx = alpha_x * np.identity(3)

    # print("OA x_dd : {}".format(x_dd))

    target_j_acc, target_j_sigma = prioritization(dq=np.array([0] * 7),
                                                  ddq=target_j_acc,
                                                  ddx=x_dd,
                                                  J=J,
                                                  dJ=np.zeros(J.shape),
                                                  sigma_ddx=sigma_ddx,
                                                  sigma_ddq=target_j_sigma)

    if use_inv_dyn:
        mass = msg_panda.get_mass().reshape(msg_panda.get_mass_dim())
        uff = mass.dot(target_j_acc) + msg_panda.get_coriolis()
    else:
        uff = target_j_acc

    # Catch start of control  TODO this is buggy use the last uff from robot
    if initCounter == 0:
        uff_last = uff

    # Clip and rate limit torque
    uff_diff = uff - uff_last
    uff_diff = np.clip(uff_diff, -rate_limit, rate_limit)
    uff = uff_last + uff_diff
    uff = np.clip(uff, -torque_limit, torque_limit)
    # print(uff)

    uff_last = uff
    last_c_vel = current_c_vel

    # Send out messages
    msg_out = pab.des_tau_msg()
    msg_out.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg_out.set_fnumber(counter)
    msg_out.set_j_torque_des(uff)
    b.send_msg(robot_name + "_des_tau", msg_out)

    initCounter += 1
    counter += 1
