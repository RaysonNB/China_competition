#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
from mr_voice.msg import Voice
from RobotChassis import RobotChassis
from pcms.openvino_models import FaceDetection, HumanPoseEstimation

def callback_Image(msg):
    global image
    image = CvBridge().img_to_msg(msg,"bgr")

def say(text):
    global _pub_speaker
    if text is None: return
    if len(text) == 0: return
    rospy.loginfo("ROBOT: %s" % text)
    _pub_speaker.publish(text)
    rospy.sleep(1)

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def callback_voice(msg):
    global _voice
    _voice = msg

def get_real_xyz(x, y):
    global depth
    if depth is None:
        return -1,-1
    h, w = depth.shape[:2]
    d = depth[y][x]
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    real_y = (h / 2 - y) * 2 * d * np.tan(a / 2) / h
    real_x = (w / 2 - x) * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
def linear_PID(cd, td):
    e = cd - td
    p = 0.0002
    x = p * e
    if x > 0:
        x = min(x, 0.16)
        x = max(x, 0.1)
    if x < 0:
        x = max(x, -0.16)
        x = min(x, -0.1)
    return x
    
def get_target_d(frame):
    poses = dnn_human_pose.forward(frame)
    frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
    global _image
    image = _image.copy()
    x1, y1, x2, y2 = 1000, 1000, 0, 0
    nlist = []
    dlist = []
    targetd =100000
    if len(poses) != 0:
        for i in range(len(poses)):
            x, y, c = map(int, poses[i][0])
            nlist.append([x,y,i])
        for i in range(len(nlist)):
            _, _, d = get_real_xyz(nlist[i][0], nlist[i][1])
            dlist.append([d,nlist[i][2]])
        for i in range(len(dlist)):
            if dlist[i][0] < targetd:
                targetd = dlist[i][0]
                targeti = dlist[i][1]                
        pose = poses[targeti]
        for i, p in enumerate(pose):
            x, y, c = map(int, p)
            if x < x1 and x != 0: x1 = x
            if x > x2: x2 = x
            if y < y1 and y != 0: y1 = y + 5
            if y > y2: y2 = y
            # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        dnn_human_pose.forward(frame)
        appearance = dnn_appearance.forward(image)
        age = dnn_age.forward(image)
                    
        # rospy.loginfo(appearance)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # rospy.loginfo(cx,cy)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        # return cx, image
        _, _, d = get_real_xyz(cx, cy)
        a = 0
        if d != 0:
            a = max(int(50 - (abs(d - 1150) * 0.0065)),20)
        rospy.loginfo(a)
        print("d : "+str(d))
        cv2.rectangle(image, (x1, y1 - a), (x2, y2), (255, 0, 0), 2)
        if d == -1:
            return -1, -1
        return d, image
    return -1, -1
if __name__ == "__main__":
    rospy.init("demo.py")
    rospy.loginfo("demo stated")
    
    image = None
    rospy.Subscriber("camera/rgb/image_raw",Image,callback_Image)
    
    depth = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    
    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    path_openvino = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(path_openvino)
    dnn_human_pose = HumanPoseEstimation()
    
    status = 0
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        _image = image.copy()
        
        if status == 0:
            faces = dnn_face.forward(frame)
            if len(faces) > 0:
                cx = ((xlist[0] + xlist[1]) // 2)
                if max(cx, 315) == cx and min(cx, 325) == cx:
                    status += 1
                    print("finish")    
                else:
                    v = angular_PID(cx, 320)
                    msg_cmd.angular.z = v
                    print(v)
                pub_cmd.publish(msg_cmd)
        if status == 1:
            d, image = get_target_d(frame)
            if d != -1:
                if d != 0:
                    rospy.loginfo(d)
                    if d < 925 or d > 975:
                        v = linear_PID(d, 950)
                        msg_cmd.linear.x = v
                        print(v)
                    else:
                        status += 1
                        print("done")
                        say("how can I help you")
                frame = image
            pub_cmd.publish(msg_cmd)
        if status == 2:
            if _voice is None: continue
            if "dinner" in _voice.text and "having" in _voice.text:
                say("ok, there do you want to sit")
                status +=1
        if status == 3:
            #follow guest
        
        if status == 4:
        