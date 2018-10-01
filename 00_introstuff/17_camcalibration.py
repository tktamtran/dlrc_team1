
import numpy as np
import scipy
import scipy.linalg as scla
import utils
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing


def normalize_points(list_points):

    dim = len(list_points[0])
    meann = np.mean(np.array(list_points), axis=0) #[np.mean(list(zip(*list_points))[d]) for d in range(dim)]
    meann = np.append(meann, 1) # for homo dim
    stds = np.std(np.array(list_points), axis=0) #[np.std(list(zip(*list_points))[d]) for d in range(dim)]
    stds = np.append(stds, 1)

    T = np.eye(dim+1)
    T[:dim,-1] -= meann[:dim]
    T[0,:] /= stds[0]
    T[1,:] /= stds[1]
    T[2,:] /= stds[2]

    norm_points = [np.dot(T, list_points[i]+[1]).tolist() for i in range(len(list_points))]
    return norm_points, T


def gen_Amatrix(points_2D, points_3D):
    '''generate A matrix, of which its null space is P the camera projection matrix
    params:
    points_2D: list of 6 image points for the point correspondences
    points_3D: list of 6 points in 3D space
    returns:
    A: 11x12 matrix
    '''
    assert(len(points_2D) ==  len(points_3D))

    # add homogeneous dimension
    if len(points_2D[0]) == 2: points_2D = [list(p)+[1] for p in points_2D]
    if len(points_3D[0]) == 3: points_3D = [list(p)+[1] for p in points_3D]
    assert(len(points_2D[0]) == 3)
    assert(len(points_3D[0]) == 4)

    As = []
    for i in range(len(points_2D)):
        X = np.array(points_3D[i]).reshape(1,4)
        x,y,w = points_2D[i]
        A0 = np.concatenate((np.zeros((1,4)), -1*w*X, y*X), axis=1) #shape should be 1x12
        A1 = np.concatenate((np.array(w*X), np.zeros((1,4)), np.array(-1*x*X)), axis=1)
        As.append(A0)
        As.append(A1)

    As = tuple(As[:11])
    A = np.concatenate(As, axis=0) #shape should be 11x12
    assert(A.shape == (11,12))
    print(A)
    return A, points_2D, points_3D


def get_P_from_points(points_2D, points_3D):
    A, points_2D, points_3D = gen_Amatrix(points_2D, points_3D)
    P = scla.null_space(A)
    assert(1 - np.linalg.norm(P) < 1e-2)# st norm(P)=1
    print('A.shape', A.shape, 'P.shape', P.shape)
    assert(np.isclose(np.dot(A,P), np.zeros(A.shape), atol=1e-3).all())
    P = np.reshape(P, (3, 4))
    print('P', P)
    # evaluate P by measuring geometric error
    error = [sum((two - np.dot(P, three))**2) for two,three in zip(points_2D, points_3D)]
    print('geometric error of P', error, 'sum of error', sum(error))

    return P


def decomp_of_P(P):
    '''return camera center C, principal point (px,py) '''
    M = P[:, :3]  # 3x3
    R,K = scipy.linalg.qr(M)
    print('K \n', K)
    print('R \n', R)
    principal_point = (K[0, 2], K[1, 2])
    C = scla.null_space(P)
    C /= C[3]
    print('C', C.shape, C)
    print('principal point', principal_point)
    return C, principal_point


def get_T0camera(joint_configuration, points_2D, points_3D):
    '''the point correspondences should be taken from a single image of the camera in a single position,
    for which the joint configuration matches it'''

    T0ee, _, _, _ = utils.get_jointToCoordinates(thetas = joint_configuration)
    print('T0ee \n', T0ee)

    # normalize points
    points_2Dn, T2 = normalize_points(points_2D)
    points_3Dn, T3 = normalize_points(points_3D)


    Pn = get_P_from_points(points_2Dn, points_3Dn)
    # denormalize P
    P = np.dot(np.dot(np.linalg.inv(T2), Pn), T3)

    # evaluate P by measuring geometric error
    error = [sum((two+[1] - np.dot(P, three+[1]))**2) for two,three in zip(points_2D, points_3D)]
    print('geometric error of P', error, 'sum of error', sum(error))

    C,_= decomp_of_P(P)
    b = np.dot(T0ee,C)
    Teecamera = np.concatenate((np.concatenate((np.eye(3), np.zeros((1,3))), axis=0), b), axis=1)
    print('Teecamera', Teecamera.shape, 'should be 4x4')
    print(Teecamera)
    T0camera = np.dot(T0ee, Teecamera)
    print('T0camera', T0camera.shape, 'should be 4x4')
    print(T0camera)
    return T0camera


def convert_depth_to_wcs(T0camera, img_depth, depth_threshold_self, principal_point):
    '''returns the camera depth value as a world coordinate point, with the robot base as the origin of the wcs
       must hold the camera fixed then, for this depth image capture
       AKA each depth image has a unique joint config of the robot'''

    img_self = img_depth * (img_depth < depth_threshold_self) # label self vs non-self

    depth_at_pp = img_self[principal_point[0], principal_point[1]]
    if depth_at_pp: # if img/depth of robot is captured in principal point
        Tpp = np.eye(4)
        Tpp[2,3] = depth_at_pp
        pp_in_wcs = np.dot(np.dot(T0camera, Tpp), np.array([0,0,0,1]))
        return pp_in_wcs[:3]
    else:
        print('principal point of image does not capture robot. No wcs point returned')
        return None


def process_camera_to_wcs(joint_configuration, points_2D, points_3D, img_depth=None, depth_threshold_self=None):
    '''move the camera to a certain joint configuration, capture the depth image,
    and assuming that the depth image is valid and the threshold is sufficient to discriminate bw self and non-self
    then process will return a wcs point of self at the principal point of the depth image'''

    T0camera = get_T0camera(joint_configuration, points_2D, points_3D)
    return T0camera
    #pp_in_wcs = convert_depth_to_wcs(T0camera, img_depth, depth_threshold_self, principal_point)
    #return pp_in_wcs

def main():
    # points identified in rgb image 230
    # don't confuse the two: image programs give values as x,y (width, height)
    # but matrix indexing is row, col (height, width)



    #points_2D = [[370, 425], [334,377], [264,489], [231,334], [160,421], [269,181]]
    #points_2D = [[477,371], [378,333], [489,262], [333,228], [422,159], [180,269]]
    points_2D = [[192,65], [312,163], [561,320], [332,299], [209,324], [135,226]]
    points_2D = [[p[0], 480-p[1]] for p in points_2D]
    points_3D = [[0.35,0.4,-0.058], [0.35,0.341,-0.03], [0.2,0.38,0.0], [0.3,0.3,0.0], [0.228,0.341,0.0],  [0.3,0.4,0.0]]
    #joint_configuration = [-2.28319192, -1.30045152,  2.34534454, -0.11885948,  0.73130912, 0.82119083, -1.87425208]
    joint_configuration = [ 0.51965302,  1.46444499,  0.80328143, -0.11341175, -0.60040778,
        0.65744787,  0.95883638]

    process_camera_to_wcs(joint_configuration, points_2D, points_3D)


if __name__ == "__main__":
    main()