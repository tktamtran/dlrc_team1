import py_at_broker as pab
import numpy as np
import time
import pandas as pd



broker = pab.broker("tcp://localhost:51468")
broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)
broker.request_signal("franka_state", pab.MsgType.franka_state)


def gen_msg(broker, fnumber, new_pos, time_to_go=0.5):
    msg = pab.target_pos_msg()
    msg.set_timestamp(time.time())
    msg.set_ctrl_t(0)
    msg.set_fnumber(fnumber)
    msg.set_pos(new_pos)
    msg.set_time_to_go(time_to_go)
    broker.send_msg("franka_target_pos", msg)


# moving to specific points
while(True):
    pos = np.array([0.5,0.5,0.05])
    gen_msg(broker, 0,pos, time_to_go=5)
    time.sleep(3)

    readings = []
    frame = 1
    for i in range(2): # go up and down 10 times
        for f in np.arange(5,30,1):
            pos = [0.50,0.20,f/100]
            gen_msg(broker, frame, np.array(pos))
            msmt = [None]*9
            msmt = broker.recv_msg("franka_lidar").get_data().tolist()
            assert(len(msmt)==9)
            joint = broker.recv_msg("franka_state").get_j_pos().tolist()
            assert(len(joint)==7)
            readings.append(pos + msmt + joint)
            print(f/100)
            time.sleep(0.5)
            frame+=1

        for f in np.arange(30, 5, -1):
            pos = [0.50,0.20,f/100]
            gen_msg(broker, frame, np.array(pos))
            msmt = [None]*9
            msmt = broker.recv_msg("franka_lidar").get_data().tolist()
            assert(len(msmt)==9)
            joint = broker.recv_msg("franka_state").get_j_pos().tolist()
            assert(len(joint)==7)
            readings.append(pos + msmt + joint)
            print(f/100)
            time.sleep(0.5)
            frame+=1

    lidar_names = ['lidar'+str(l) for l in range(9)]
    joint_names = ['joint'+str(j) for j in range(7)]

    readings = pd.DataFrame(readings, columns=['posX', 'posY', 'posZ']+lidar_names+joint_names)
    pd.to_pickle(readings, 'readings_z_again.pkl')
    break