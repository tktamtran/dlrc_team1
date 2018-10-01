# This script was born out of the realization that is is much easier and
# cleaner to have a single script for each sensor signal which avoids many
# concurrency problems

import py_at_broker as pab
import numpy as np
import datetime
import sys
import signal
import pickle
import os

store = list()

def signal_handler(signal, frame):
    print("\nCapturing stopped")
    # check the file name
    # create ISO 8601 UTC time stamped file name
    fname = datetime.datetime.utcnow() \
        .replace(tzinfo=datetime.timezone.utc) \
        .replace(microsecond=0).isoformat()
    fname = "".join([c for c in fname if c.isalpha()
                     or c.isdigit() or c == ' ']).rstrip()
    fname = fname + "_STATE.pkl"
    fname = os.path.join(os.path.expanduser("~"), "measurements", fname)
    print(f"File will be saved as {fname}")
    with open(fname, "wb") as f:
        pickle.dump(store, f,pickle.HIGHEST_PROTOCOL)
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    broker = pab.broker("tcp://10.250.144.21:51468")
    broker.register_signal("franka_target_pos", pab.MsgType.target_pos)
    # input('st 6 in xpanda...')
    broker.request_signal("franka_state", pab.MsgType.franka_state)

    while(True):
        msg = broker.recv_msg("franka_state", -1)
        data = get_state(msg)
        store.append(data)


if __name__ == "__main__":
    main()

def get_state(msg):
    data = dict()
    data['systemtime'] = datetime.datetime.now()
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