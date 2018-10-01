# this file generates random movements for the robot for camera dataset
# generation
import dlrc_control as ctrl
import numpy as np
import time
from pyquaternion import Quaternion
import time

broker = ctrl.initialize()

target = np.array([0, 0, 0])


numvalid = 0
while numvalid < 1000:
    # generate random position
    x = np.random.uniform(0, 1)
    y = np.random.uniform(-1,1)
    z = np.random.uniform(0.15,1)
    position = [x, y, z]
    # look at random height
    target[2] = np.random.uniform(-0.5, 0.5)
    q = ctrl.look_at(position, target)
    pose = np.concatenate((position, q.q))
    cmdtime = time.time()
    ctrl.set_new_pos(broker, pose, ctrl_mode=0, time_to_go=5)
    ctrl.wait_til_ready(broker)
    # if the command passed in less than 100ms, it was probably not valid
    if time.time - cmdtime < 0.1:
        pass
    else:
        numvalid += 1
    print(pose)