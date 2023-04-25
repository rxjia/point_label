from copy import deepcopy
import threading

import numpy as np
from util_funs import load_board_param
from util_ros import RosListener
import rospy

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters
import cv2 as cv
from cv2 import aruco
from image_geometry import PinholeCameraModel
import helper


class Listener(RosListener):
    def __init__(self) -> None:
        """ subscribe rbg/depth image simultaneously"""
        super().__init__(imshow=False)
        self.base_frame = 'board_0'
        self.T_base2boards = load_board_param()
        self._msg_lock = threading.Lock()
        self.msg_latest = None
        self.click_points = []

        # ARUCO
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
        pix_sz = int(600/25.4 + 0.5)  # dpi/25.4
        size_unit = (210-5*2)/(0.04*5+0.01*4)/1000  # 33.33333 mm

        boards = {}
        for bid in range(3):
            name = f"board_{bid}"
            if name in self.T_base2boards.keys():
                board = cv.aruco.GridBoard((5, 7), 0.04*size_unit, 0.01*size_unit, dictionary, ids=np.arange(35)+bid*35)
                boards[name] = board

        self.dictionary = dictionary
        self.boards = boards
        # ----------------------------------------

        self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1, buff_size=2**24*2)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1, buff_size=2**24*2)

        self.cam_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo, queue_size=1)

        # sync rgb/depth
        self.rgb_depth_sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.cam_info_sub], 1, 0.03)
        self.rgb_depth_sync.registerCallback(self.callback)

        self.det_pub = rospy.Publisher("/camera/color/detect_img", Image, queue_size=1)

    def callback(self, rgb: Image, depth: Image, cam_info: CameraInfo):
        """ callback function """
        img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        stamp_now = rgb.header.stamp
        out_img = img.copy()
        # Load the dictionary that was used to generate the markers.
        dictionary = self.dictionary

        # Initialize the detector parameters using default values
        parameters = cv.aruco.DetectorParameters()

        detector = cv.aruco.ArucoDetector(dictionary, parameters)
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)

        if markerIds is not None:
            cv.aruco.drawDetectedMarkers(out_img, markerCorners, markerIds)

            cam_model = PinholeCameraModel()
            cam_model.fromCameraInfo(cam_info)
            camK = cam_model.K
            camD = cam_model.D

            def get_matched_pts(markerCorners, markerIds):
                l_objPoints = []
                l_imgPoints = []
                for board_name, board in self.boards.items():
                    board = self.boards[board_name]
                    objPoints, imgPoints = board.matchImagePoints(markerCorners, markerIds)
                    if objPoints is not None:
                        objPoints = helper.tfs_cv_pts(self.T_base2boards[board_name], objPoints)
                        l_objPoints.append(objPoints.copy())
                        l_imgPoints.append(imgPoints.copy())
                if len(l_objPoints) > 0:
                    l_objPoints = np.concatenate(l_objPoints)
                    l_imgPoints = np.concatenate(l_imgPoints)
                    return l_objPoints, l_imgPoints
                else:
                    return None, None

            T_cam2base = None
            objPoints, imgPoints = get_matched_pts(markerCorners, markerIds)
            if objPoints is not None:
                markersOfBoardDetected = len(objPoints) // 4
            else:
                markersOfBoardDetected = 0

            with self._msg_lock:
                if markersOfBoardDetected > 4:
                    ret, rvec, tvec, inliner = cv.solvePnPRansac(objPoints, imgPoints, camK, camD)
                    cv.drawFrameAxes(out_img, camK, camD, rvec, tvec, 0.1)
                    T_cam2base = helper.cv_rt2mat44(rvec, tvec)

                    self.T_cam2base = deepcopy(T_cam2base)
                    self.msg_latest = rgb, depth, cam_model, T_cam2base

                    self.pubtf(T_cam2base, stamp_now, self.base_frame, rgb.header.frame_id)

                    for pt in self.click_points:
                        cam2pt = helper.tfs_cv_pts(T_cam2base, pt).squeeze()
                        uv = cam_model.project3dToPixel(cam2pt)
                        uv = np.round(np.array(uv)).astype(int)

                        cv.drawMarker(out_img, position=uv, color=(0, 0, 255), markerSize=50, markerType=cv.MARKER_CROSS, thickness=3)

                self.showimg(out_img)
                out_msg = self.bridge.cv2_to_imgmsg(out_img, "bgr8", header=rgb.header)
                self.det_pub.publish(out_msg)

    def mouse_cb(self, event, u, v, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            rospy.loginfo("L %d, %d" % (u, v))
            with self._msg_lock:
                if self.msg_latest is not None:
                    rgb, depth, cam_model, T_cam2base = self.msg_latest
                    img_depth = self.bridge.imgmsg_to_cv2(depth)
                    z = img_depth[v, u]

                    xy1 = np.array([(u - cam_model.cx()) / cam_model.fx(),
                                    (v - cam_model.cy()) / cam_model.fy(),
                                    1])
                    pos = xy1*z/1000.

                    if len(self.click_points) >= 50:
                        self.click_points = []

                    p_in_base = helper.tfs_cv_pts(helper.mr.TransInv(T_cam2base), pos[None, None, :])
                    self.click_points.append(p_in_base)
                    rospy.loginfo(f"L {u}, {v}, {p_in_base.squeeze()}")

        elif event == cv.EVENT_MBUTTONDBLCLK:
            rospy.loginfo("clear click_points")
            with self._msg_lock:
                self.click_points = []

    def run(self):
        cv.namedWindow("ui click", cv.WINDOW_NORMAL)
        cv.setMouseCallback("ui click", self.mouse_cb)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            img_show = self.get_showimg()
            if img_show is None:
                rate.sleep()
                continue

            cv.imshow("ui click", img_show)
            key = cv.waitKey(10)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("listener", anonymous=True)
    listener = Listener()
    listener.run()
    rospy.spin()
