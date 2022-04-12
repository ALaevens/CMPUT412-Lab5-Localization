#!/usr/bin/env python3

from cv2 import undistort
import rospy
import os
import numpy as np
import yaml
import cv2
import os
from tag import Tag
import math
from dt_apriltags import Detector
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Range, CompressedImage, Image
import cv2
import cv_bridge
import math
from std_msgs.msg import String

bridge = cv_bridge.CvBridge()





class MyNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
     
    
        self.img_sub = rospy.Subscriber('/csc22905/camera_node/image/compressed', CompressedImage, self.image_callback)
        self.range_sub = rospy.Subscriber("/csc22905/front_center_tof_driver_node/range", Range, self.range_callback)
        self.undistorted_pub = rospy.Publisher("/csc22905/lab5/undistorted/compressed", CompressedImage, queue_size=1)
        self.location_pub = rospy.Publisher("/csc22905/lab5/location_string", String, queue_size=1)

        TAG_SIZE = .08
        FAMILIES = "tagStandard41h12"
        self.tags = Tag(TAG_SIZE, FAMILIES)

        # Add information about tag locations
        # Function Arguments are id, x, y, z, theta_x, theta_y, theta_z (euler) 
        # for example, self.tags.add_tag( ... 
        self.tags.add_tag(0, 0, 0, 1, 0, -(3*math.pi)/2, 0)
        self.tags.add_tag(1, 1, 0, 2, 0, 0, 0)
        self.tags.add_tag(2, 2, 0, 1, 0, -math.pi/2, 0)
        self.tags.add_tag(3, 1, 0, 0, 0, -math.pi, 0)

        # Load camera parameters
        with open("/data/config/calibrations/camera_intrinsic/csc22905.yaml") as file:
                camera_list = yaml.load(file,Loader = yaml.FullLoader)

        self.camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3,3)
        self.distortion_coeff = np.array(camera_list['distortion_coefficients']['data']).reshape(5,1)

        self.camera = None
        self.undistorted = None

    def image_callback(self, data: CompressedImage):
        camera = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        self.camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
                # TODO: add your subsribers or publishers here
    
    def range_callback(self, data: Range):
        pass


    def undistort(self, img):
        '''
        Takes a fisheye-distorted image and undistorts it

        Adapted from: https://github.com/asvath/SLAMDuck
        '''
        height = img.shape[0]
        width = img.shape[1]

        newmatrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic_matrix,
            self.distortion_coeff, 
            (width, height),
            1, 
            (width, height))

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_intrinsic_matrix, 
            self.distortion_coeff,  
            np.eye(3), 
            newmatrix, 
            (width, height), 
            cv2.CV_16SC2)

        undistorted_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
       
        return undistorted_image   
             

    def detect(self, img):
        '''
        Takes an images and detects AprilTags
        '''
        PARAMS = [
            self.camera_intrinsic_matrix[0,0],
            self.camera_intrinsic_matrix[1,1],
            self.camera_intrinsic_matrix[0,2],
            self.camera_intrinsic_matrix[1,2]] 


        TAG_SIZE = 0.08 
        detector = Detector(families="tagStandard41h12", nthreads=1)
        detected_tags = detector.detect(
            img, 
            estimate_tag_pose=True, 
            camera_params=PARAMS, 
            tag_size=TAG_SIZE)

        return detected_tags

    def run(self):
        rate = rospy.Rate(10)

        stop = False
        while not stop and not rospy.is_shutdown():
            if self.camera is not None:
                undistorted = self.undistort(self.camera)
                msg = bridge.cv2_to_compressed_imgmsg(undistorted)
                self.undistorted_pub.publish(msg)
                tags_found = self.detect(undistorted)
                
                avg_pose = np.array([[0.0], [0.0], [0.0]])
                avg_rot = np.array([0.0, 0.0, 0.0])
                for tag in tags_found:
                    pose_estimate = self.tags.estimate_pose(tag.tag_id, tag.pose_R, tag.pose_t)
                    avg_pose += pose_estimate
                    rot_estimate = self.tags.estimate_euler_angles(tag.tag_id, tag.pose_R, tag.pose_t)
                    #print(rot_estimate)
                    avg_rot += rot_estimate

                if len(tags_found) > 0:
                    avg_pose /= len(tags_found)
                    avg_rot /= -1*len(tags_found)
                    self.location_pub.publish(f"{avg_pose[0,0]} {avg_pose[2, 0]}")
                    print(f"AVERAGE POSE: (X: {avg_pose[0, 0]}, Y:{avg_pose[1, 0]}, Z:{avg_pose[2, 0]})")
                    print(f"AVERAGE Theta Y: {avg_rot[1]}")
                             

            rate.sleep()

if __name__ == "__main__":
    node = MyNode(node_name="lab5_node")
    node.run()

