#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
import math
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
    

def find_all():
    global ddn_rcnn
    global frame
    global boxes
    boxes = ddn_rcnn.forward(frame)
    if len(boxes) == 0:
        return "nothing"
    for id, index, conf, x1, y1, x2, y2 in boxes:
        name=ddn_rcnn.labels[index]
        #if namee=="bottle": #name=="suitcase" or name=="backpack":
        cv2.putText(frame, name, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = (x2 - x1) // 2 + x1
        cy = (y2 - y1) // 2 + y1
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        return (x1, y1), (x2, y2), (cx, cy), name


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
def get_x():
    global ax,ay,az,bx,by,bz,px,py,px
    num1=(bx-ax)*px+(by-ay)*py+(bz-az)*pz-(bx-ax)*ax-(by-ay)*ay-(bz-az)*az
    num2=(bx-ax)**2+(by-ay)**2+(bz-az)**2
    t=num1/num2
    qx=(bx-ax)*t + ax
    qy=(by-ay)*t + ay
    qz=(bz-az)*t + az
    distance = math.sqrt((px-qx)**2 +(py-qy)**2+(bz-az)**2)
    return distance
    
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
        boxes = ddn_rcnn.forward(frame)
        #[0, tensor(1), tensor(0.9491, grad_fn=<UnbindBackward>), 2, 102, 100, 264]
        #[0, tensor(1), tensor(0.8638, grad_fn=<UnbindBackward>), 118, 121, 194, 202]
        for id, index, conf, x1, y1, x2, y2 in boxes:
            name=ddn_rcnn.labels[index]
            if pose is not None:
                #for num in [7,9]: #[7,9] left, [8,10] right
                cx, cy = get_pose_target(pose,7)
                cv2.circle(frame, (cx,cy), 5, (0, 255, 0), -1)
                ax, ay, az = get_real_xyz(cx, cy)
                cx, cy = get_pose_target(pose,9)
                cv2.circle(frame, (cx,cy), 5, (0, 255, 0), -1)
                bx, by, bz = get_real_xyz(cx, cy)
            if name=="bottle": #name=="suitcase" or name=="backpack":
                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1
                cv2.circle(frame, (cx1, cy1), 5, (0, 255, 0), -1)
                px,py,pz=get_real_xyz(cx1, cy1)
                cv2.putText(frame, str(get_x()), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    
    rospy.loginfo("demo node end!")