import py_at_broker as pab
import matplotlib.pyplot as plt

# in a separate terminal, $ add_broker
# $ ./start_camera

# create a broker to connect with the address broker
ourbroker = pab.broker("tcp://localhost:51468")
ourbroker.request_signal("realsense_images", pab.MsgType.realsense_image)
img1 = ourbroker.recv_msg("realsense_images", -1)
img1_rgb = img1.get_rgb()
img1_rgbshape = img1.get_shape_rgb()
img1_rgb = img1_rgb.reshape(img1_rgbshape)

img1_depth = img1.get_depth()
img1_depthshape = img1.get_shape_depth()
img1_depth = img1_depth.reshape(img1_depthshape)



plt.imshow(img1_depth)
plt.show()

plt.imshow(img1_rgb)
plt.show()