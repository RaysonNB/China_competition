#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def say(a): 
    publisher_speaker.publish(a) 

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
    
def get_real_xyz(x, y):
    global _depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth[y][x]
    h, w = _depth.shape[:2]
    x = x - w // 2
    y = y - h // 2
    real_y = y * 2 * d * np.tan(a / 2) / h
    real_x = x * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
    
    
def get_target(poses):
    target = -1
    target_d = 9999999
    for i, pose in enumerate(poses):
        for num in [7,8,9,10]:
            cx, cy = get_pose_target(pose,num)
            _, _, d = get_real_xyz(cx, cy)
        if target == -1 or (d != 0 and d < target_d):
            target = i
            target_d = d
    if target == -1: return None
    return poses[target]
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.sleep(1)
    ddn_rcnn = FasterRCNN()
    net_pose = HumanPoseEstimation()
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None: continue
        if _depth is None: continue
        frame = _frame.copy()
        poses = net_pose.forward(frame)
        pose = get_target(poses)
        if pose is not None:
            
            # 1 pose
            
            # 2 points
            
            # objects
            
            # calc
            for num in [7,8,9,10]:
                cx, cy = get_pose_target(pose,num)
                _, _, d = get_real_xyz(cx, cy)
                cv2.circle(frame, (cx,cy), 5, (0, 255, 0), -1)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    
    rospy.loginfo("demo node end!")
