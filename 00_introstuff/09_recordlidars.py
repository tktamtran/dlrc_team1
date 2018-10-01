# This script was born out of the realization that is is much easier and
# cleaner to have a single script for each sensor signal which avoids many
# concurrency problems

import py_at_broker as pab
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
    fname = fname + "_LIDAR.pkl"
    fname = os.path.join(os.path.expanduser("~"), "measurements", fname)
    print(f"File will be saved as {fname}")
    with open(fname, "wb") as f:
        pickle.dump(store, f,pickle.HIGHEST_PROTOCOL)
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    broker = pab.broker("tcp://10.250.144.21:51468")
    broker.request_signal("franka_lidar", pab.MsgType.franka_lidar)

    while(True):
        data = dict()
        msg = broker.recv_msg("franka_lidar", -1)
        data['systemtime'] = datetime.datetime.now()
        data['data'] = msg.get_data()
        # print(data['data'])
        data['timestamp'] = msg.get_timestamp()
        data['fnumber'] = msg.get_fnumber()
        store.append(data)


if __name__ == "__main__":
    main()
