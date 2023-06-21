#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math

#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global depth1
    depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    frame1 = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image1)

    depth1= None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth1)
    
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        
        if frame1 is None or frame2 is None: continue
        if depth1 is None or depth2 is None: continue
        
        detections = dnn_yolo.forward(frame)[0]["det"]
        for i, detection in enumerate(detections):
            fall=0
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            if class_id != 0:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px,py,pz=get_real_xyz(cx, cy)
            if pz<=2000:
              print("w: ",x2-x1,"h: ", y1-y2)
              if cy<=160:
                  fall+=1
              if h<w:
                  fall+=1
            if fall==2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2)
              #cv2.putText(frame, str(int(pz)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 3)
        
