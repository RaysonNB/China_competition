#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack

from m1_function import *

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def say(a):
    publisher_speaker.publish(a)


def callback_depth(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def get_real_xyz(x, y):
    global _depth1
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth1[y][x]
    h, w = _depth1.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1
    return int(p[0][0]), int(p[0][1])


def pose_draw():
    cx7, cy7, cx9, cy9, cx5, cy5 = 0, 0, 0, 0, 0, 0
    global ax, ay, az, bx, by, bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1, n2, n3 = 6, 8, 10
    cx7, cy7 = get_pose_target(pose, n2)

    cx9, cy9 = get_pose_target(pose, n3)

    cx5, cy5 = get_pose_target(pose, n1)
    if cx7 == -1 and cx9 != -1:
        cv2.circle(rgb_image, (cx5, cy5), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(rgb_image, (cx9, cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 != -1 and cx9 == -1:

        cv2.circle(rgb_image, (cx5, cy5), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(rgb_image, (cx7, cy7), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx7, cy7)
    elif cx7 == -1 and cx9 == -1:
        print("where is your hand")
        #continue
    else:
        cv2.circle(rgb_image, (cx7, cy7), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)

        cv2.circle(rgb_image, (cx9, cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)


def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")

if __name__ == "__main__":    
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    # RGB Image Subscriber
    _image1 = None
    _topic_image1 = "/cam1/rgb/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    rospy.wait_for_message(_topic_image1, Image)
    
    # Depth Image Subscriber
    _depth1 = None
    _topic_depth1 = "/cam1/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)
    rospy.wait_for_message(_topic_depth1, Image)

    _frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)

    _depth1 = None
    _depth=_depth1
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    #rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    s=""
    #color
    
    mask = 0
    net_pose = HumanPoseEstimation()
    key = 0
    is_turning = False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z = 0.0, 0.0
    t=3.0
    '''
    joint1, joint2, joint3, joint4 = 0.000, 0.0, 0.0, 0.0
    set_joints(joint1, joint2, joint3, joint4, t)
    time.sleep(t)
    joint1, joint2, joint3, joint4 = 0.000, 0.4, -1.400, 0.900
    set_joints(joint1, joint2, joint3, joint4, t)
    time.sleep(t)
    open_gripper(t)
    '''
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("bagv3")
    dnn_yolo.classes = ['obj']
    '''
    chassis = RobotChassis()
    chassis.set_initial_pose_in_rviz()
    
    goal = [[-1.49, 8.48, 0.00247]]
    '''
    step="start"
    while not rospy.is_shutdown():
        t = 3.0
        rospy.Rate(10).sleep()
        if _frame is None:
            continue
        if _depth1 is None:
            continue
        print(ax, ay, az, bx, by, bz)
        flag = None
        depth = _depth1.copy

        min1 = 99999999
        
        rgb_image = _frame.copy()
        rgb_image=cv2.flip(rgb_image, 0)
        frame = rgb_image.copy()

        if step == "start":
            detections = dnn_yolo.forward(rgb_image)[0]["det"]
            for i, detection in enumerate(detections):
                print(detection)
                x1,y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                
                if score > 0.4 and class_id == 0:

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cx = (x2 - x1) // 2 + x1
                    cy = (y2 - y1) // 2 + y1
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    continue


            cnt = 1
            frame = _frame.copy()
            depth = _depth1.copy()

            pose = None
            poses = net_pose.forward(frame)

            for i, pose in enumerate(poses):
                point = []
                for j, (x, y, preds) in enumerate(pose):  # x: ipex 坐標 y: ipex 坐標 preds: 准度
                    if preds <= 0:
                        continue
                    x, y = map(int, [x, y])
                    for num in [8, 10]:
                        point.append(j)
                if len(point) == 2:
                    pose = poses[i]
                    #break
            '''
            if pose is not None:
                msg = Twist()
                pose_draw()
                print("idiot")
                if flag == "bag2":
                    mask = detector2.get_mask(rgb_image)
                    cnts = detector2.find_contours(mask)
                    if len(cnts) > 0:
                        cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 0, 255), 5)
                        cx, cy = detector2.find_center(cnts[0])
                        cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
                        #rospy.loginfo("yiooooo")
                        move1(cx, cy, msg)
                        if _depth1[cy][cx]<=10:
                            rospy.loginfo("E")
                            for i in range(10):
                                msg.linear.x = pre_x + 0.05
                                _cmd_vel.publish(msg)
                            close_gripper(t)
                            step="follow"
                            break
                        rospy.loginfo("ggggg")
                else:
                    mask = detector1.get_mask(rgb_image)
                    cnts = detector1.find_contours(mask)
                    if len(cnts) > 0:
                        cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 0, 255), 5)
                        cx, cy = detector1.find_center(cnts[0])
                        cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
                        #rospy.loginfo("yiooooo")
                        move1(cx, cy, msg)
                        if _depth1[cy][cx]<=10:
                            rospy.loginfo("E")
                            for i in range(10):
                                msg.linear.x = pre_x + 0.05
                                _cmd_vel.publish(msg)
                            close_gripper(t)
                            step="follow"
                            break
                        rospy.loginfo("ggggg")
                    '''
        #elif step=="back":
            #chassis.move_to(goal[i][0], goal[i][1], goal[i][2])
            #break
        #elif step == "follow":
            

        
        cv2.imshow("image", rgb_image)
        #cv2.imshow("image", _image1)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
