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
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
    
class ColorDetector(object):
    
    def __init__(self, lower, upper, min_size=1000):
        self.lower = lower
        self.upper = upper
        self.min_size = min_size
        
    def get_mask(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def find_contours(self, mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_size:
                results.append(cnt)
        results.sort(key=cv2.contourArea, reverse=True)
        return results
    
    def find_center(self, cnt):
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            x = int(np.round(m["m10"] / m["m00"]))
            y = int(np.round(m["m01"] / m["m00"]))
            return x, y
        return 0, 0

    def physical_distance(self, depth_image, x, y, angle=0, max_range=25):
        radian = float(angle) * math.pi / 180
        
        real_x = 0
        real_y = 0
        real_z = 0
        
        
        h, w = depth_image.shape
        flag = False
        e = 0
        while not flag and e < max_range:
            depth = depth_image[max(cy - e, 0):min(cy + e, h), max(cx - e, 0):min(cx + e, w)].copy()
            indices = np.nonzero(depth)
            if len(indices[0]) > 0:
                real_z = np.min(depth[indices])
                flag = True
            else:
                e = e + 1
        
        FOV_H = 60.0
        d = real_z
        lw = d * math.tan(FOV_H / 2 * math.pi / 180)
        lx = float(x) / w * lw * 2 - w / 2
        real_x = lx
        
        FOV_V = 49.5
        d = real_z
        lh = d * math.tan(FOV_V / 2 * math.pi / 180)
        ly = float(y) / h * lh * 2 - h / 2
        real_y = ly
        
        real_x = real_x
        real_y = real_y + real_z * math.sin(radian)
        real_z = real_z * math.cos(radian)
                
        return real_x, real_y, real_z
    
def get_real_xyz(x, y):
    global _depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth[y][x]
    h, w = _depth.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w): continue
                    d = _depth[y - k][j]
                    if d > 0: break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h): continue
                    d = _depth[i][x + k]
                    if d > 0: break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w): continue
                    d = _depth[y + k][j]
                    if d > 0: break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h): continue
                    d = _depth[i][x - k]
                    if d > 0: break
            if d > 0: break
            
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
    
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
    

def get_distance(px,py,pz,ax,ay,az,bx,by,bz):
    A,B,C,p1,p2,p3,qx,qy,qz,distance=0,0,0,0,0,0,0,0,0,0
    if ax<=0 or bx<=0 or az==0 or bz==0 or pz==0:
        return 0
    A=int(bx)-int(ax)
    B=int(by)-int(ay)
    C=int(bz)-int(az)
    p1=int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2=int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3=int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print(p1,p2,p3)
    if (p1-p2)!=0 and p3!=0:
        t=(int(p1)-int(p2))/int(p3)
        qx=int(A)*int(t) + int(ax)
        qy=int(B)*int(t) + int(ay)
        qz=int(C)*int(t) + int(az)
        distance = int(pow(((int(px)-int(qx))**2 +(int(py)-int(qy))**2+(int(pz)-int(qz))**2),0.5))
        return int(distance)
    return 0
    
def pose_draw():
    cx7,cy7,cx9,cy9,cx5,cy5=0,0,0,0,0,0
    global ax,ay,az,bx,by,bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        cv2.circle(rgb_image, (cx5,cy5), 5, (255, 0, 0), -1)
        ax,ay,az = get_real_xyz(cx5, cy5)
        
        cv2.circle(rgb_image, (cx9,cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 !=-1 and cx9 ==-1:

        cv2.circle(rgb_image, (cx5,cy5), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)
        
        cv2.circle(rgb_image, (cx7,cy7), 5, (255, 0, 0), -1)
        bx,by,bz = get_real_xyz(cx7, cy7)
    elif cx7 ==-1 and cx9 == -1:
        print("where is your hand")
        #continue
    else:
        cv2.circle(rgb_image, (cx7,cy7), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)
        
        cv2.circle(rgb_image, (cx9,cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
def move1(cx,cy,msg):
    global frame,rgb_image,pre_z,pre_x
    h, w, c = rgb_image.shape
    e = w // 2 - cx
    d=_depth
    if abs(e) > 20:
        #rospy.loginfo("if")s
        v = 0.0025 * e
        cur_z = v
        dz = cur_z - pre_z
        if dz > 0: dz = min(dz, 0.1)
        if dz < 0: dz = max(dz, -0.1)
        msg.angular.z = pre_z + dz
    else:
        #rospy.loginfo("else")
        d = _depth[cy][cx]
        rospy.loginfo("d: %d" % d)

        if d > 0 and d < 3000:
            v1 = 0.0001 * d
            msg.linear.x = v1
            cur_x = v1

            dx = cur_x - pre_x
            if dx > 0: dx = min(dx, 0.05)
            if dx < 0: dx = max(dx, -0.05)
            msg.linear.x = pre_x + dx
        else:
            msg.linear.x = 0.0
    pre_x, pre_z = msg.linear.x, msg.angular.z
    _cmd_vel.publish(msg)
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    #arm = Manipulator()
    #chassis = Kobuki()
    detector1 = ColorDetector((100, 16, 16), (110, 255, 255))
    detector2 = ColorDetector((50, 16, 16), (100, 255, 255))
    mask=0
    net_pose = HumanPoseEstimation()
    key = 0
    is_turning = False
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    pre_x, pre_z = 0.0, 0.0
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if _frame is None: continue
        if _depth is None: continue
        print(ax,ay,az,bx,by,bz)
        flag=None
        depth=_depth.copy
        
        min1=99999999
        
        rgb_image = _frame.copy()
        frame= rgb_image.copy
        mask = detector1.get_mask(rgb_image)
        cnts = detector1.find_contours(mask)
        if len(cnts) > 0:
            cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 255, 0), 2)
            cx, cy = detector1.find_center(cnts[0])
            cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
            px,py,pz=get_real_xyz(cx, cy)
            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
            cnt=int(cnt)
            if cnt< min1 and cnt<400 and cnt != 0:
                flag = "bag1"
            real_x, real_y, d = get_real_xyz(cx,cy)
            
            rx, ry, rz = detector1.physical_distance(_depth, cx, cy, 30)
            rx = rx + 40
            #rospy.loginfo("%.1f mm, %.1f mm, %.1f mm %d" % (rx, ry, rz,d))
        mask = detector2.get_mask(rgb_image)
        cnts = detector2.find_contours(mask)
        if len(cnts) > 0:
            cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 255, 0), 2)
            cx, cy = detector2.find_center(cnts[0])
            cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
            
            real_x, real_y, d = get_real_xyz(cx,cy)
            px,py,pz=get_real_xyz(cx, cy)
            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
            cnt=int(cnt)
            if cnt< min1 and cnt<400 and cnt != 0:
                flag = "bag2"
            rx, ry, rz = detector2.physical_distance(_depth, cx, cy, 30)
            rx = rx + 40
            #rospy.loginfo("%.1f mm, %.1f mm, %.1f mm %d" % (rx, ry, rz,d))   
                     
        cnt = 1
        frame = _frame.copy()
        depth=_depth.copy()
        
        
        pose=None
        poses = net_pose.forward(frame)
    
        for i, pose in enumerate(poses):
            point = []
            for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                if preds <= 0: continue
                x,y = map(int,[x,y])
                for num in [8,10]:
                    point.append(j)
            if len(point) == 2:
                pose = poses[i]
                break

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
                    move1(cx,cy,msg)
            else:
                mask = detector1.get_mask(rgb_image)
                cnts = detector1.find_contours(mask)
                if len(cnts) > 0:
                    cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 0, 255), 5)
                    cx, cy = detector1.find_center(cnts[0])
                    cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
                    move1(cx,cy,msg)
          

        cv2.imshow("image", rgb_image)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
