import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# the readings are huge, maybe just load a subportion of it
readings = pd.DataFrame(pd.read_pickle('measurements/dataorig_robot_batch_rt0.pkl'))
print("data loaded")
plt.ion()

framerate = 20
depthimgs = readings.realsense_depth
rgbimgs = readings.realsense_rgbdata
print(f"{len(depthimgs)} images. At {framerate} fps this will take at "
      f"least {len(depthimgs)/framerate} seconds")

#img = plt.imshow(np.log10(depthimgs[0]+1))
#plt.colorbar()
subplot_row, subplot_column = 1,2
fig = plt.figure()
ax1 = fig.add_subplot(subplot_row, subplot_column, 1)
ax2 = fig.add_subplot(subplot_row, subplot_column, 2)

for r,d in zip(rgbimgs, depthimgs):
    # if d is None:
    #     d = plt.imshow(np.log10(im+1))
    # else:
    #     img.set_data(np.log10(im+1))

    ax1.imshow(r)
    ax2.imshow(d)
    plt.pause(1/framerate)
    plt.draw()

