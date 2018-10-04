# this file is used to check if the basic collision avoidance works as intended
import dlrc_control as ctrl
import numpy as np
import time
from pyquaternion import Quaternion

broker = ctrl.initialize(addr_broker_ip="tcp://localhost:51468", realsense=True, lidar=True)

# pos = np.array([0.272, 0.01, 0.52])
gripper_q = Quaternion(np.array([0.054, 0.921, 0.364, 0.131])).unit
init_pos = np.array([0.55, 0.0, 0.50])
init_pos = np.concatenate([init_pos, gripper_q.q])
start_pos = np.array([0.55, 0.0, 0.42])
start_pos = np.concatenate([start_pos, gripper_q.q])
target_pos = np.array([0.55, 0.01, 0.60])
target_pos = np.concatenate([target_pos, gripper_q.q])
i = 0

while True:
    try:
        print(f'Moving to starting position. Cycle {i}')
        ctrl.check_if_path_free(broker)
        ctrl.set_new_pos(broker, start_pos, ctrl_mode=0, time_to_go=3)
        ctrl.wait_til_ready_ca(broker)
        time.sleep(3)
        print(f'Moving to target position. Cycle {i}')
        ctrl.check_if_path_free(broker)
        ctrl.set_new_pos(broker, target_pos, ctrl_mode=0, time_to_go=3)
        ctrl.wait_til_ready_ca(broker)
        time.sleep(3)
        i += 1
    except ctrl.CollisionException as e:
        print(e)
        time.sleep(3)
        print("Collision Avoidance stopped the robot\nRestarting...")
        ctrl.check_if_path_free(broker)
        ctrl.set_new_pos(broker, init_pos, ctrl_mode=0, time_to_go=1)
        time.sleep(2)

# observation: it seems like this will be the basic pattern of 'safe movement'
#              check if nothing is in the way, set the new position and stop if
#              something steps in its way