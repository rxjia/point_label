import os

import geometry_msgs.msg
import numpy as np

from geometry_msgs.msg import Quaternion, Vector3, PoseStamped
from tf import transformations as tfs
from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation as sR
import modern_robotics as mr

# 按照内旋方式，Z-Y-X旋转顺序（指先绕自身轴Z，再绕自身轴Y，最后绕自身轴X），可得旋转矩阵（内旋是右乘） R.from_euler('ZYX', [1,2,3]).as_matrix()
# 按照外旋方式，X-Y-Z旋转顺序（指先绕固定轴X，再绕固定轴Y，最后绕固定轴Z），可得旋转矩阵（外旋是左乘） R.from_euler('xyz', [1,2,3]).as_matrix()

PATH_ROOT = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))


def cv_rt2mat44(rvec, tvec):
    M = np.identity(4)
    M[:3, :3] = sR.from_rotvec(rvec.squeeze()).as_matrix()
    M[:3, 3] = tvec.squeeze()
    return M


def tfs_cv_pts(T, cv_pts):
    """
    T:       4x4
    cv_pts:  nx1x3
    """
    return (T[None, :3, :3] @ cv_pts.transpose(0, -1, -2) + T[None, :3, [3]]).transpose(0, -1, -2)


def Rot(mat44):
    M = np.identity(4)
    M[:3, :3] = mat44[:3, :3].copy()
    return M


def RT_to_mat44(R, T):
    M = np.identity(4)
    M[:3, :3] = R.copy()
    M[:3, 3] = T.copy()
    return M


def pose2xyz_wxyz(pose):
    p = pose
    pos_x = p.position.x
    pos_y = p.position.y
    pos_z = p.position.z
    quat_w = p.orientation.w
    quat_x = p.orientation.x
    quat_y = p.orientation.y
    quat_z = p.orientation.z
    pos_goal = np.array([pos_x, pos_y, pos_z])
    quat_goal = np.array([quat_w, quat_x, quat_y, quat_z])
    return pos_goal, quat_goal


def pose2xyz_xyzw(pose):
    p = pose
    pos_x = p.position.x
    pos_y = p.position.y
    pos_z = p.position.z
    quat_w = p.orientation.w
    quat_x = p.orientation.x
    quat_y = p.orientation.y
    quat_z = p.orientation.z
    pos_goal = np.array([pos_x, pos_y, pos_z])
    quat_goal = np.array([quat_x, quat_y, quat_z, quat_w])
    return pos_goal, quat_goal


def wxyz2xyzw(wxyz):
    return [*wxyz[1:], wxyz[0]]


def save_file(str, filepath):
    with open(filepath, "w") as file:
        file.write(str)


def xyz_to_mat44(pos):
    return tfs.translation_matrix((pos.x, pos.y, pos.z))


def xyzw_to_mat44(ori):
    return tfs.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))


def xyzquat_to_mat44(xyzquat):
    return np.dot(tfs.translation_matrix(xyzquat[:3]), tfs.quaternion_matrix(xyzquat[3:]))


def xyzrpy_to_pose(x, y, z, roll, pitch, yaw):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    quat = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def xyzrpy_to_mat44(x, y, z, roll, pitch, yaw):
    return np.dot(tfs.translation_matrix([x, y, z]), tfs.euler_matrix(roll, pitch, yaw))


def xyzrpydeg_to_mat44(x, y, z, roll, pitch, yaw):
    return np.dot(tfs.translation_matrix([x, y, z]), tfs.euler_matrix(*np.deg2rad([roll, pitch, yaw])))


def gen_pose(x, y, z, rx, ry, rz, rw):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = rx
    pose.orientation.y = ry
    pose.orientation.z = rz
    pose.orientation.w = rw
    return pose


def mat44_to_xyzrpy(matrix):
    xyz = tfs.translation_from_matrix(matrix)
    rpy = tfs.euler_from_matrix(matrix)
    return [*xyz, *rpy]


def mat44_to_xyzrpy_deg(matrix):
    xyz = tfs.translation_from_matrix(matrix)
    rpy = tfs.euler_from_matrix(matrix)
    return [*xyz, rpy[0] * (180. / np.pi), rpy[1] * (180. / np.pi), rpy[2] * (180. / np.pi)]


def mat44_to_xyzquat(matrix):
    xyz = tfs.translation_from_matrix(matrix)
    quat = tfs.quaternion_from_matrix(matrix)
    return [*xyz, *quat]


def mat44_to_pose(matrix):
    return gen_pose(*mat44_to_xyzquat(matrix))


def quat2msg(quat):
    """
    :param quat: rotation quaternion expressed as a tuple (x,y,z,w)
    :return: geometry_msgs.msg.Quaternion
    """
    return Quaternion(*quat)


def pose_to_mat44(pose):
    """
    :param pose: geometry_msgs.msg.Pose
    :return:
    """
    return np.dot(xyz_to_mat44(pose.position), xyzw_to_mat44(pose.orientation))


def mat44_to_transform(mat44):
    transform = geometry_msgs.msg.Transform()
    xyz = tfs.translation_from_matrix(mat44)
    transform.translation = Vector3(*xyz)
    transform.rotation = quat2msg(tfs.quaternion_from_matrix(mat44))
    return transform


def transrot_to_mat44(translation, rotation):
    """
    :param translation: translation expressed as a tuple (x,y,z)
    :param rotation: rotation quaternion expressed as a tuple (x,y,z,w)
    :return: a :class:`numpy.matrix` 4x4 representation of the transform
    :raises: any of the exceptions that :meth:`~tf.Transformer.lookupTransform` can raise

    Converts a transformation from :class:`tf.Transformer` into a representation as a 4x4 matrix.
    """

    return np.dot(tfs.translation_matrix(translation), tfs.quaternion_matrix(rotation))


def transform_to_mat44(transform):
    t = transform.translation
    r = transform.rotation
    return transrot_to_mat44([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])


def rtp2xyz(r, theta, phi):
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    ball_x = r * s_theta * np.cos(phi)
    ball_y = r * s_theta * np.sin(phi)
    ball_z = r * c_theta
    return ball_x, ball_y, ball_z


def xyz2rtp(xyz):
    r = np.linalg.norm(xyz)
    theta = np.arccos(xyz[2] / r)
    phi = np.arctan2(xyz[1], xyz[0])
    return np.array([r, theta, phi])


def rtp_rad2show(rtp):
    ret = np.array(rtp)
    ret[1:] = np.rad2deg(ret[1:])
    return ret


def rtp2deg(rtp):
    ret = [rtp[0], np.rad2deg(rtp[1]), np.rad2deg(rtp[2])]
    return ret


def tw2twistmsg(tw):
    msg = geometry_msgs.msg.Twist()
    msg.linear.x = tw[0]
    msg.linear.y = tw[1]
    msg.linear.z = tw[2]

    msg.angular.x = tw[3]
    msg.angular.y = tw[4]
    msg.angular.z = tw[5]
    return msg
# using mr.TransTotw
# def mat442xyzomg(mat44):
#     xyz = tfs.translation_from_matrix(mat44)
#     omg = R.from_matrix(mat44[:3,:3]).as_rotvec()
#     return np.array([*xyz, *omg])
#
#
# def xyzomg2mat44(xyzomg):
#     M = np.identity(4)
#     M[:3,3] = xyzomg[:3]
#     M[:3,:3] = R.from_rotvec(xyzomg[3:]).as_matrix()
#     return M
