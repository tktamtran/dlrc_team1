# This script continuously reads the data from the franka robot
# and its mounted sensors and records them into a pickle
# this is not as fast as recording each sensor on its own, but much simpler

import argparse
import py_at_broker as pab
import datetime
import os
import signal
import pickle
import sys

store = list()

fname = ""

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

def get_lidar(msg):
    data = dict()
    data['data'] = msg.get_data()
    data['timestamp'] = msg.get_timestamp()
    data['fnumber'] = msg.get_fnumber()
    return data

def get_realsense(msg):
    data = dict()
    data['timestamp'] = msg.get_timestamp()
    data['fnumber'] = msg.get_fnumber()
    data['fnumber'] = msg.get_fnumber()
    rgbdata = msg.get_rgb()
    data['rgbdata'] = rgbdata.reshape(msg.get_shape_rgb())
    depthdata = msg.get_depth()
    data['depthdata'] = depthdata.reshape(msg.get_shape_depth())
    return data


def signal_handler(signal, frame):
    print("\nCapturing stopped")
    # check the file name
    # create ISO 8601 UTC time stamped file name
    fname = datetime.datetime.utcnow() \
        .replace(tzinfo=datetime.timezone.utc) \
        .replace(microsecond=0).isoformat()
    fname = "".join([c for c in fname if c.isalpha()
                     or c.isdigit() or c == ' ']).rstrip()
    imname = fname
    fname = fname + "_ALL.pkl"

    fname = os.path.join(os.path.expanduser("~"), "measurements", fname)
    print(f"File will be saved as {fname}")
    with open(fname, "wb") as f:
        pickle.dump(store, f, pickle.HIGHEST_PROTOCOL)
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--name",
                        help="Specifies a pickle file name to use")
    argparser.add_argument("-s", "--simulator",
                        help="Indicates working with the simulator",
                        action="store_true")
    argparser.add_argument("-r","--realsense",
                        help="Indicates if realsense data should be captured",
                        action="store_true")

    args = argparser.parse_args()
    if not args.simulator:
        print("Working with the REAL ROBOT.\n"
              "Provide the argument --simulator or -s"
              " to record simulator data")
    else:
        print("Working with the SIMULATOR")

    # check the file name
    if not args.name:
        print("No pickle file added, creating one based on time stamp")
        # create ISO 8601 UTC time stamped file name
        fname = datetime.datetime.utcnow()\
            .replace(tzinfo=datetime.timezone.utc)\
            .replace(microsecond=0).isoformat()
        fname = "".join([c for c in fname if c.isalpha()
                         or c.isdigit() or c==' ']).rstrip()
        if args.simulator:
            fname = fname+"_SIMULATOR"
        else:
            fname = fname+"_REALBOT"
        fname = fname+".pkl"
    elif os.access(os.path.dirname(args.name), os.W_OK):
        fname = args.name
    else:
        raise ValueError("No valid pickle file name given")


    # Set up the broker
    broker = pab.broker("tcp://10.250.144.21:51468") #
    # in any case, record the robot data
    broker.request_signal("franka_state", pab.MsgType.franka_state)
    # do not record the lidar data in simulator
    if not args.simulator:
        broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)
    # only record the camera data of requested
    if args.realsense:
        print('Recording REALSENSE data')
        broker.request_signal("realsense_images", pab.MsgType.realsense_image)

    # simply record the data
    while(True):
        reading = dict()
        reading['systemtime'] = datetime.datetime.now()
        print(f'time: {reading["systemtime"]}\tstate ', end='')
        msg = broker.recv_msg("franka_state", -1)

        states = get_state(msg)
        for key in states.keys():
            reading['state_'+key] = states[key]

        if not args.simulator:
            print('lidar', end='')
            msg = broker.recv_msg("franka_lidar", -1)
            lidars = get_lidar(msg)
            for key in lidars.keys():
                reading['lidar_' + key] = lidars[key]

        if args.realsense:
            print('realsense')
            msg = broker.recv_msg("realsense_images", -1)
            realsenses = get_realsense(msg)
            for key in realsenses.keys():
                reading['realsense_' + key] = realsenses[key]

        store.append(reading)


if __name__ == "__main__":
    main()