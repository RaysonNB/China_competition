#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
from mr_voice.msg import Voice
from pcms.pytorch_models import *
from mr_voice.msg import Voice
from std_msgs.msg import String

def callback_voice(msg):
    global s
    s = msg.text
    
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def say(g):
    publisher_speaker.publish(g)
    
if __name__ == "__main__":
    rospy.init_node("one")
    rospy.loginfo("demo node start!")
    rospy.Subscriber("/voice/text", Voice, callback_voice)

    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    # ROS Topics
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    rospy.wait_for_message("/camera/rgb/image_raw", Image)
    say("started")
    rospy.loginfo("idiot")
    # PyTorch
    ddn_rcnn = FasterRCNN()
    dnn_yolo = Yolov5()

    # MAIN LOOP
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        frame = _frame.copy()

        # Torch
        boxes = ddn_rcnn.forward(frame)
        for id, index, conf, x1, y1, x2, y2 in boxes:
            name=ddn_rcnn.labels[index]
            if name=="suitcase" or name=="backpack":
                cv2.putText(frame, name, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        '''
        boxes = dnn_yolo.forward(frame)
        for id, index, conf, x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, dnn_yolo.labels[index], (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        '''
        # show image
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

