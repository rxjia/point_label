from copy import deepcopy
import pickle

import numpy as np
from util_funs import save_board_param
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
        # self.base_frame = "world"
        self.base_frame = 'board_0'
        self.T_cam2boards = {}
        
        self.rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1, buff_size=2**24*2)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=1, buff_size=2**24*2)

        self.cam_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo,queue_size=1)

        # sync rgb/depth
        self.rgb_depth_sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.cam_info_sub], 1, 0.03)
        self.rgb_depth_sync.registerCallback(self.callback)
        
        self.det_pub = rospy.Publisher("/camera/color/detect_img", Image, queue_size=1)
    
    def callback(self, rgb:Image, depth:Image, cam_info:CameraInfo):
        """ callback function """
        img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        stamp_now = rgb.header.stamp
        out_img = img.copy()
        #Load the dictionary that was used to generate the markers.
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)

        # Initialize the detector parameters using default values
        parameters =  cv.aruco.DetectorParameters()

        detector = cv.aruco.ArucoDetector(dictionary,parameters)
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        
        # if markerIds is not None:
        cv.aruco.drawDetectedMarkers(out_img, markerCorners, markerIds)
        # corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
        #         image = gray,
        #         board = board,
        #         detectedCorners = corners,
        #         detectedIds = ids,
        #         rejectedCorners = rejectedImgPoints,
        #         cameraMatrix = cameraMatrix,
        #         distCoeffs = distCoeffs)  
        # print(markerIds)
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(cam_info)
        camK = cam_model.K
        distCoeffs = cam_model.D
        
        idx_start = 35*2
        pix_sz = int(600/25.4 + 0.5) # dpi/25.4
        size_unit = (210-5*2)/(0.04*5+0.01*4)/1000 # 33.33333
        boards = [cv.aruco.GridBoard((5, 7), 0.04*size_unit, 0.01*size_unit, dictionary, ids=np.arange(35)+idx_start) for idx_start in [0,35*1,35*2]]
        
        
        def get_board_pose(board):
            objPoints, imgPoints = board.matchImagePoints(markerCorners, markerIds)
            if objPoints is not None:
                markersOfBoardDetected = len(objPoints) // 4
            else:
                markersOfBoardDetected = 0

            if markersOfBoardDetected > 4:
                ret, rvec, tvec, inliner = cv.solvePnPRansac(objPoints,imgPoints, camK, distCoeffs)
                cv.drawFrameAxes(out_img, camK, distCoeffs, rvec, tvec, 0.1)
                
                T_cam2board = helper.cv_rt2mat44(rvec, tvec)
                return T_cam2board
            else:
                return None
        
        det_results = {}
        for idx_board, board in enumerate(boards):
            T_cam2board = get_board_pose(board)
            if T_cam2board is not None:
                board_name = f"board_{idx_board}"
                det_results[board_name] = T_cam2board
                self.pubtf(T_cam2board, stamp_now, f"board_{idx_board}", rgb.header.frame_id)
        
        
        T_base2boards={}
        if len(det_results)>=1:
            if self.base_frame in det_results.keys():
                T_cam2base = det_results[self.base_frame]
                T_base2boards[self.base_frame] = np.eye(4)
                for k, v in det_results.items():
                    if k != self.base_frame:
                        T_cam2board = v
                        T_base2boards[k] = helper.mr.TransInv(T_cam2base).dot(T_cam2board)

        if len(T_base2boards)>=1:
            rospy.loginfo(f"Get boards: {T_base2boards}")
            save_board_param(T_base2boards)
            cv.namedWindow("board_detect", cv.WINDOW_NORMAL)
            cv.imshow("board_detect", out_img)
            cv.waitKey()
            rospy.signal_shutdown("finish")
        

if __name__ == "__main__":
    rospy.init_node("get_board_param")
    listener = Listener()
    rospy.spin()