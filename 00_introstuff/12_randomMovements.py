# this file generates random movements for the robot
import dlrc_control as ctrl
import numpy as np
from pyquaternion import Quaternion
import time
import py_at_broker as pab

#broker = ctrl.initialize()

broker = pab.broker("tcp://localhost:51468")
broker.register_signal("franka_target_pos", pab.MsgType.target_pos)

def gen_msg(broker, fnumber, new_pos, ctrl_mode):
    msg = pab.target_pos_msg()
    msg.set_timestamp(time.time())
    msg.set_ctrl_t(ctrl_mode)
    msg.set_fnumber(fnumber)
    msg.set_pos(new_pos)
    msg.set_time_to_go(0.2)
    broker.send_msg("franka_target_pos", msg)

ctrl_mode = 2


if ctrl_mode == 0:

    # moving to specific points
    f = 0
    while (True):

        pos = np.array([0.5, 0.5, 0.3])
        gen_msg(broker, f, pos, 0)
        time.sleep(2)

        for i in range(10):  # go up and down 10 times
            for f in np.arange(30, 70, 1):
                # gen_msg(broker, f, np.array([0.3, 0.4, 0.00]))
                gen_msg(broker, f, np.array([0.50, 0.50, f / 100]), 0)
                print(f / 100)
                time.sleep(0.2)

            for f in np.arange(70, 30, -1):
                gen_msg(broker, f, np.array([0.50, 0.50, f / 100]), 0)
                print(f / 100)
                time.sleep(0.2)
        f +=1

    # for i in range(2000):
    #     target = position
    #     target[2] = 0
    #     q = ctrl.look_at(position, target)
    #     pose = np.concatenate((position, q.q))
    #     ctrl.set_new_pos(broker, position, ctrl_mode=0, time_to_go=5, fnumber=i)

if ctrl_mode == 1:

    working_joint_configs = []
    f = 1000
    while(True):
        position = ctrl.random_joint_config()
        position = np.array(position)
        working_joint_configs.append(position)
        #ctrl.set_new_pos(broker, position, ctrl_mode=1, time_to_go=5, fnumber=f)
        gen_msg(broker, f, position[:7], 1)
        f+=1

if ctrl_mode == 2:

    joint_configs = [[-0.012, -0.005, 0.009, -1.558, -0.015, 1.879, -0.075], #go0
                     [-1.816, -0.265, 2.773, -2.089, -0.452, 1.481, -1.873],
                     ]

    joint_configs = ctrl.gen_joint_configs(initial = joint_configs[0], n_configs=10)

    f = 0
    while (True):
        #ctrl.set_new_pos(broker, j, ctrl_mode=1, time_to_go=5, fnumber=i)
        gen_msg(broker, f, joint_configs[f%joint_configs.shape[0]], 1)
        time.sleep(5)
        f+=1