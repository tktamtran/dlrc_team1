import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# the readings are huge, maybe just load a subportion of it
readings = pd.DataFrame(pd.read_pickle('/home/dlrc1/measurements/20180928T1329150000.pkl'))
print("data loaded")
plt.ion()

framerate = 50
depthimgs = readings.realsense_depthdata
print(f"{len(depthimgs)} images. At {framerate} fps this will take at "
      f"least {len(depthimgs)/framerate} seconds")

img = plt.imshow(np.log10(depthimgs[0]+1))
plt.colorbar()

for im in depthimgs:
    if img is None:
        img = plt.imshow(np.log10(im+1))
    else:
        img.set_data(np.log10(im+1))
    plt.pause(1/framerate)

    plt.draw()

