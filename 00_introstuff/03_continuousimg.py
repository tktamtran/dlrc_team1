import py_at_broker as pab
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import numpy as np
import sys


import cv2

# in a separate terminal, $ add_broker
# in a separate terminal, $ ./start_camera


def grab_image():
    broker = pab.broker("tcp://localhost:51468")
    broker.request_signal("realsense_images", pab.MsgType.realsense_image)

    img1 = broker.recv_msg("realsense_images", -1)
    img1_rgb = img1.get_rgb()
    img1_rgbshape = img1.get_shape_rgb()
    img1_rgb = img1_rgb.reshape(img1_rgbshape)

    img1_depth = img1.get_depth()
    img1_depthshape = img1.get_shape_depth()
    img1_depth = img1_depth.reshape(img1_depthshape)   
 
    return img1_rgb, img1_depth

def update(i):
    rgb, depth = grab_image()
    im1.set_data(rgb)
    im2.set_data(depth)
    # compute statistics
    # print(f'mean: {np.mean(depth):.2f}\t5%: {np.percentile(depth, 0.05):.2f}\t'
    #       f'median: {np.median(depth):.2f}\t95%:{np.median(0.95):.2f}')



# while(True):
#    img = grab_image()
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#    cv2.imshow('ourStream', img)
#    cv2.waitKey(0)
#
# cv2.destroyAllWindows()

while(True):
    rgb, depth = grab_image()
    ax1 = plt.subplot(121)
    im1 = ax1.imshow(rgb)

    ax2 = plt.subplot(122)
    im2 = ax2.imshow(depth)
    plt.colorbar(im2,ax=ax2)

    ani = FuncAnimation(plt.gcf(), update, interval=30)
    plt.show()

    inp = input('write stop to stop ')
    if inp == 'stop':
        sys.exit()



