from copy import deepcopy
import threading
import rospy
import tf as ros_tf
import tf2_ros
import cv2
from cv_bridge import CvBridge
import helper as helper
from geometry_msgs.msg import TransformStamped

class RosListener:
    def __init__(self, base_frame=None, imshow=True) -> None:
        self.bridge = CvBridge()
        self.tf_listener = ros_tf.TransformListener()
        self.tf_broadcaster = ros_tf.TransformBroadcaster()
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.base_frame = base_frame
        
        self._imshow_lock = threading.Lock()
        self._imshow_img = None
        self._imshow_shutdown = False
        self._thread_imgshow = threading.Thread(target=self.run_imshow, name='thread_imgshow')
        if imshow:
            self._thread_imgshow.start()
        print("RosListener init done")

    def run_imshow(self):
        cv2.namedWindow("imshow", cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            with self._imshow_lock:
                if self._imshow_shutdown:
                    break
                if self._imshow_img is not None:
                    cv2.imshow("imshow", self._imshow_img)
            cv2.waitKey(33)

    def showimg(self, img):
        with self._imshow_lock:
            self._imshow_img = img.copy()
    
    def get_showimg(self):
        with self._imshow_lock:
            return deepcopy(self._imshow_img)

    def get_tf(self, target_frame, source_frame, time=rospy.Time(0), timeout=rospy.Duration(0)):
        """

        :param target_frame:
        :param source_frame:
        :param time: The time at which to get the transform. (0 will get the latest)
        :param timeout: (Optional) Time to wait for the target frame to become available.
        :return:
        """
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, time, timeout)
            tr_target2source = self.tf_listener.lookupTransform(target_frame, source_frame, time)
            tf_target2source = self.tf_listener.fromTranslationRotation(*tr_target2source)
        except Exception as e:
            print(e)
            tf_target2source = None
        return tf_target2source

    def pubtf(self, mat44, stamp_now, child, parent=None):
        if parent is None:
            assert self.base_frame is not None
            parent = self.base_frame

        xyzquat = helper.mat44_to_xyzquat(mat44)
        self.tf_broadcaster.sendTransform(xyzquat[:3],
                                          xyzquat[3:],
                                          stamp_now,
                                          child,
                                          parent)

    def pub_static_tf(self, mat44, time, child, parent=None):
        if parent is None:
            assert self.base_frame is not None
            parent = self.base_frame

        xyzquat = helper.mat44_to_xyzquat(mat44)
        translation = xyzquat[:3]
        rotation = xyzquat[3:]

        t = TransformStamped()
        t.header.frame_id = parent
        t.header.stamp = time
        t.child_frame_id = child
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        self.tf_static_broadcaster.sendTransform(t)
