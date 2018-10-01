# this file generates random movements for the robot
import dlrc_control as ctrl
import numpy as np
from pyquaternion import Quaternion
import time

broker = ctrl.initialize()


position = np.array([0.45315024, 0.46458719, 0.26050831])
# position = np.array([0.4, 0.2, 0.3])
# position = np.array([0.4, 0.1, 0.3])
# position = np.array([0.4, -0.1, 0.3])

for i in range(200):
    target = position
    target[2] = 0
    q = ctrl.look_at(position, target)
    pose = np.concatenate((position, q.q))
    ctrl.set_new_pos(broker, position, ctrl_mode=0, time_to_go=5)
