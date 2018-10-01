# this file generates random movements for the robot for camera dataset
# generation
import dlrc_control as ctrl
import numpy as np
from pyquaternion import Quaternion
import time

broker = ctrl.initialize()

target = np.array([0, 0, 0])

for x in np.arange(0,1,0.1):
    for y in np.arange(-0.8,0.8,0.1):
        for z in list(np.arange(0.15,0.8,0.05)) + list(np.arange(0.15,0.8,0.05)):
            position = [x,y,z]
            q = ctrl.look_at(position, target)
            pose = np.concatenate((position, q.q))
            ctrl.set_new_pos(broker, pose, ctrl_mode=0, time_to_go=5)
            ctrl.wait_til_ready(broker)
            print(pose)


# numvalid = 0
# while numvalid < 1000:

