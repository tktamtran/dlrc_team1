import dlrc_control as ctrl
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import time, sys, argparse
import py_at_broker as pab
import pandas as pd



# deriving the transformation matrices using the DH params
# the Transformation matrix from (i-1) to (i) is
# there are actually two distinct conventions for denavit hartenberg
# which drastically change the result
# one is the "wkipedia" DH
# the other one is given on p. 83/75 of http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040


def get_jointToCoordinates(thetas, trueCoordinates=None, untilJoint=None):
    '''
    gets coordinates of tip of end-effector depending on the 7 joint orientations
    params:
    thetas: list of joint orientations, length 7
    trueCoordinates: include this if want to calc difference from true coordinates, derived from cwhere
    returns:
    Tproduct: transformation matrix of the 7th joint/end-effector to the robot base wcs'''

    Tlist = []
    Tproduct = np.eye(4, 4)

    # for 7 joints
    aa = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    dd = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
    alphas = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
    if type(thetas) != list: thetas = thetas.tolist()
    thetas += [0]
    for a, d, alpha, theta in zip(aa, dd, alphas, thetas):
        T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                      [np.cos(alpha) * np.sin(theta), np.cos(alpha) * np.cos(theta), -np.sin(alpha),
                       -np.sin(alpha) * d],
                      [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                      [0, 0, 0, 1]])

        # make sure this is a proper transformation matrix composed of a rotation and translational part:
        if not np.isclose(T[0:3, 0:3].T, np.linalg.inv(T[0:3, 0:3]), 1e-4, 1e-4).all(): raise ValueError(
            'transformation matrix invalid')

        Tproduct = np.dot(Tproduct, T)
        Tlist.append(T)

    # for end-effector
    Tee = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.11], [0, 0, 0, 1]])
    #     Tee = np.array([[1,0,0,0.11],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    Tlist.append(Tee)
    Tproduct = np.dot(Tproduct, Tee)  # transformation matrix from robot base to ~end-effector

    EE_coord = Tproduct.dot(np.array([0, 0, 0, 1]))
    assert (EE_coord[-1] == 1.00)

    if trueCoordinates:
        print('difference of ', np.sqrt(sum((trueCoordinates - EE_coord[:3]) ** 2)))

    if untilJoint:
        Tjoint = np.eye(4, 4)
        for T in Tlist[:untilJoint]:
            Tjoint = np.dot(Tjoint, T)
    else:
        Tjoint = None

    return Tproduct, Tlist, Tjoint, EE_coord





def img_to_ccs(depth_image, principal_point, camera_resolution, skip, rgb_image):

    depth_image = np.array(depth_image) / 1000

    ccs_points = []
    rgb_colors = []
    for x in np.arange(1,camera_resolution[1], skip):
        for y in np.arange(0,camera_resolution[0], skip):
            a = np.sin((x-principal_point[1])/camera_resolution[1] * 91.2 * np.pi/180) /1.45
            b = np.sin((y-principal_point[0])/camera_resolution[0] * 65.5 * np.pi/180) /1.45
            #a = ((x-principal_point[1])/0.00193) * 2.32
            #b = ((y - principal_point[0]) / 0.00193) * 2.32
            #a = (x - principal_point[1]) / camera_resolution[1] * 0.30
            #b = (y - principal_point[0]) / camera_resolution[0] * 0.30
            #a = (x - principal_point[1]) / camera_resolution[1]
            #b = (y - principal_point[0]) / camera_resolution[0]
            ccs_point = depth_image[y,x] * np.array([a,b,1/1.015,0]) #1.035
            ccs_point[-1] = 1
            ccs_points.append(ccs_point)

            rgb_colors.append([rgb_image[y*2,x*2]]) # rgb image is double the resolution of depth

    ccs_points = np.array(ccs_points)
    rgb_colors = np.array(rgb_colors).squeeze(axis=1)

    return ccs_points, rgb_colors


def grab_image(broker=None):

    if not broker: # either do these initialization steps inside or outside function, but either way happens just once
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