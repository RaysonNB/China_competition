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


class RobotControl:
    def __init__(self):
        self.service_names = {
            'set_joint_position': '/goal_joint_space_path',
            'set_tool_control': '/goal_tool_control'
        }
        self.services = {}

        for name in self.service_names:
            self.services[name] = rospy.ServiceProxy(self.service_names[name], SetJointPosition)

    def set_joints(self, joint1, joint2, joint3, joint4, t):
        try:
            request = SetJointPositionRequest()
            request.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
            request.joint_position.position = [joint1, joint2, joint3, joint4]
            request.path_time = t
            response = self.services['set_joint_position'](request)
            return response
        except Exception as e:
            rospy.loginfo("%s" % e)
            return False

    def set_gripper(self, angle, t):
        try:
            request = SetJointPositionRequest()
            request.joint_position.joint_name = ["gripper"]
            request.joint_position.position = [angle]
            request.path_time = t
            response = self.services['set_tool_control'](request)
            return response
        except Exception as e:
            rospy.loginfo("%s" % e)
            return False

    def open_gripper(self, t):
        return self.set_gripper(0.01, t)

    def close_gripper(self, t):
        return self.set_gripper(-0.01, t)


class Get:
    def __init__(self, depth1):
        self.depth1 = depth1
    
    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])

        if len(p) == 0:
            return -1, -1
        return int(p[0][0]), int(p[0][1])

    def get_real_xyz(self, x, y):
        a = 49.5 * np.pi / 180
        b = 60.0 * np.pi / 180
        d = self.depth1[y][x]
        h, w = self.depth1.shape[:2]
        x = int(x) - int(w // 2)
        y = int(y) - int(h // 2)
        real_y = round(y * 2 * d * np.tan(a / 2) / h)
        real_x = round(x * 2 * d * np.tan(b / 2) / w)
        return real_x, real_y, d

    def get_distance(self, px, py, pz, ax, ay, az, bx, by, bz):
        A, B, C, p1, p2, p3, qx, qy, qz, distance = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if ax <= 0 or bx <= 0 or az == 0 or bz == 0 or pz == 0:
            return 0
        A = int(bx)-int(ax)
        B = int(by)-int(ay)
        C = int(bz)-int(az)
        p1 = int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
        p2 = int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
        p3 = int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
        #print(p1,p2,p3)
        if (p1-p2) != 0 and p3 != 0:
            t = (int(p1)-int(p2))/int(p3)
            qx = int(A)*int(t) + int(ax)
            qy = int(B)*int(t) + int(ay)
            qz = int(C)*int(t) + int(az)
            distance = int(
                pow(((int(px)-int(qx))**2 + (int(py)-int(qy))**2+(int(pz)-int(qz))**2), 0.5))
            return int(distance)
        return 0


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


def move1(cx, cy, msg):
    global frame, rgb_image, pre_z, pre_x
    h, w, c = rgb_image.shape
    e = w // 2 - cx
    d = _depth1
    if abs(e) > 20:
        #rospy.loginfo("if")s
        v = 0.0025 * e
        cur_z = v
        dz = cur_z - pre_z
        if dz > 0:
            dz = min(dz, 0.1)
        if dz < 0:
            dz = max(dz, -0.1)
        msg.angular.z = pre_z + dz
    else:
        #rospy.loginfo("else")
        d = _depth1[cy][cx]
        rospy.loginfo("d: %d" % d)

        if d > 0 and d < 3000:
            v1 = 0.0001 * d
            msg.linear.x = v1
            cur_x = v1

            dx = cur_x - pre_x
            if dx > 0:
                dx = min(dx, 0.05)
            if dx < 0:
                dx = max(dx, -0.05)
            msg.linear.x = pre_x + dx
        else:
            msg.linear.x = 0.0
    pre_x, pre_z = msg.linear.x, msg.angular.z
    _cmd_vel.publish(msg)

