import numpy as np
import transforms3d as t3d
import pyquaternion as pq


def pose_difference(curPose, desPose):
    curQuat = curPose[-4:]
    desQuat = desPose[-4:]

    cur_c_pos = curPose[:3]
    des_c_pos = desPose[:3]

    x_err = np.zeros(6)
    x_err[:3] = des_c_pos - cur_c_pos
    temp_cur = curQuat[0]
    curQuat[0] = curQuat[3]
    curQuat[3] = temp_cur
    temp_des = desQuat[0]
    desQuat[0] = desQuat[3]
    desQuat[3] = temp_des

    x_err[3] = (curQuat[0] * desQuat[1]
                - desQuat[0] * curQuat[1]
                - curQuat[3] * desQuat[2]
                + curQuat[2] * desQuat[3])

    x_err[4] = (curQuat[0] * desQuat[2]
                - desQuat[0] * curQuat[2]
                + curQuat[3] * desQuat[1]
                - curQuat[1] * desQuat[3])

    x_err[5] = (curQuat[0] * desQuat[3]
                - desQuat[0] * curQuat[3]
                - curQuat[2] * desQuat[1]
                + curQuat[1] * desQuat[2])
    return x_err


q = pq.Quaternion(axis=(0, 0, 1), degrees=90)
print(t3d.euler.quat2euler(q.q))

x_start = np.array([0, 0, 0, 1, 0, 0, 0])
x_target = np.array([0, 0, 1, 0.707, 0.707, 0, 0])

print(pose_difference(x_start, x_target))
