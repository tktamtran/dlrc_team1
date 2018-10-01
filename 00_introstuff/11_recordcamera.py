# This script was born out of the realization that is is much easier and
# cleaner to have a single script for each sensor signal which avoids many
# concurrency problems

import py_at_broker as pab
import datetime
import sys
import signal
import pickle
import os
import skimage as sk
from skimage import io

# TODO: store data if it becomes too large for memory (does that even happen?)

store = list()


# specify whether to store images in png (huge, lossless)
# and/or jpg (smaller, lossy) format
png = True
jpg = False

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
    fname = fname + "_realsense.pkl"

    # save the images as image files
    for d in store:
        curname = "_"+imname
        rgbname = curname +"_RGB"
        depthname = curname + "_Depth"
        rgbdata = d['rgbdata']
        depthdata = d['depthdata']
        timestamp = d['timestamp']
        timestamp = "T"+str(timestamp).replace(".","_")
        rgbname = timestamp+rgbname
        depthname = timestamp+depthname
        if(png):
            pngrgb = os.path.join(os.path.expanduser("~"),
                                  "measurements", "camera",
                                  rgbname+".png")
            pngdepth = os.path.join(os.path.expanduser("~"),
                                    "measurements", "camera",
                                    depthname+".png")
            sk.io.imsave(pngrgb, rgbdata)
            sk.io.imsave(pngdepth, depthdata)
        if(jpg):
            jpgrgb = os.path.join(os.path.expanduser("~"),
                                  "measurements", "camera",
                                  rgbname + ".jpg")
            jpgdepth = os.path.join(os.path.expanduser("~"),
                                    "measurements", "camera",
                                    depthname + ".jpg")
            sk.io.imsave(jpgrgb, rgbdata)
            sk.io.imsave(jpgdepth, depthdata)
    fname = os.path.join(os.path.expanduser("~"), "measurements", fname)
    print(f"File will be saved as {fname}")
    with open(fname, "wb") as f:
        pickle.dump(store, f, pickle.HIGHEST_PROTOCOL)
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    broker = pab.broker("tcp://10.250.144.21:51468")
    broker.request_signal("realsense_images", pab.MsgType.realsense_image)

    while (True):
        msg = broker.recv_msg("realsense_images", -1)
        data = dict()
        data['systemtime'] = datetime.datetime.now()
        data['timestamp'] = msg.get_timestamp()
        data['fnumber'] = msg.get_fnumber()
        rgbdata = msg.get_rgb()
        data['rgbdata'] = rgbdata.reshape(msg.get_shape_rgb())
        depthdata = msg.get_depth()
        data['depthdata'] = depthdata.reshape(msg.get_shape_depth())
        store.append(data)


if __name__ == "__main__":
    main()