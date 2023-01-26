#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist


def callback_image(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
    
def get_real_xyz(x, y):
    global depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = depth[y][x]
    h, w = depth.shape[:2]
    x = x - w // 2
    y = y - h // 2
    real_y = y * 2 * d * np.tan(a / 2) / h
    real_x = x * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
    
def get_pose_target(pose):
    p = []
    for i in [5, 6, 11, 12]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    min_x = max_x = p[0][0]
    min_y = max_y = p[0][1]
    for i in range(len(p)):
        min_x = min(min_x, p[i][0])
        max_x = max(max_x, p[i][0])
        min_y = min(min_y, p[i][1])
        max_y = max(max_y, p[i][1])
    
    cx = int(min_x + max_x) // 2
    cy = int(min_y + max_y) // 2
    return cx, cy
    
    
def get_target(poses):
    target = -1
    target_d = 9999999
    for i, pose in enumerate(poses):
        cx, cy = get_pose_target(pose)
        _, _, d = get_real_xyz(cx, cy)
        if target == -1 or (d != 0 and d < target_d):
            target = i
            target_d = d
    if target == -1: return None
    return poses[target]
    
    
def calc_angular_z(cx, tx):
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z
    
    
def calc_linear_x(cd, td):
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x
    

if __name__ == "__main__":
    rospy.init_node("demo2")
    rospy.loginfo("demo2 started!")
    
    # RGB Image Subscriber
    image = None
    topic_image = "/camera/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    
    # Depth Image Subscriber
    depth = None
    topic_depth = "/camera/depth/image_raw"
    rospy.Subscriber(topic_depth, Image, callback_depth)
    rospy.wait_for_message(topic_depth, Image)
    
    # cmd_vel Publisher
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist)
    
    # Models
    net_pose = HumanPoseEstimation()
    
    # Main loop
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        frame = image.copy()
        
        poses = net_pose.forward(frame)
        pose = get_target(poses)
        if pose is not None:
            cx, cy = get_pose_target(pose)
            _, _, d = get_real_xyz(cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            z = calc_angular_z(cx, 320)
            x = calc_linear_x(d, 1000)
            msg_cmd.linear.x = 0.0
            print(x)
            # msg_cmd.angular.z = z
            
        pub_cmd.publish(msg_cmd)
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    
    rospy.loginfo("demo2 end!")
    
