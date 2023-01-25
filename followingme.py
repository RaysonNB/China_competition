#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from std_msgs.msg import String


def say(a): 
    publisher_speaker.publish(a) 
   
   
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def turn(v):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = 0.0
    msg.angular.z = v
    _cmd_vel.publish(msg)


def move(v):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = 0.0
    _cmd_vel.publish(msg)


def find_bottle():
    global ddn_rcnn
    global frame
    global boxes
    boxes = ddn_rcnn.forward(frame)
    if len(boxes) == 0:
        return "nothing"
    for id, index, conf, x1, y1, x2, y2 in boxes:
        name=ddn_rcnn.labels[index]
        if 1==1: #name=="suitcase" or name=="backpack":
            cv2.putText(frame, name, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx = (x2 - x1) // 2 + x1
            cy = (y2 - y1) // 2 + y1
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            return (x1, y1), (x2, y2), (cx, cy), name


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    ddn_rcnn = FasterRCNN()
    rospy.sleep(1)
    
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        
        if _frame is None: continue
        if _depth is None: continue
        #rospy.loginfo(_frame)
        #rospy.loginfo("%d, %d" % (_frame.shape[:2]))
        
        frame = _frame.copy()

        find_b = find_bottle()
        if find_b != "nothing":
            p1, p2, c, box = find_b
            cx, cy = c
            #rospy.loginfo("%d, %d" % (cx, cy))

            h, w, c = frame.shape
            e = w // 2 - cx
            if abs(e) > 20:
                rospy.loginfo("if")
                v = 0.000625 * e
                turn(v)
            else:
                rospy.loginfo("else")
                d = _depth[cy][cx]
                rospy.loginfo("d: %d" % d)
                
                if d > 0 and d < 3000:
                    if d < 800:
                        continue
                    move(0.1) 
                else:
                    move(0.0)
        else:
            turn(0.2)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    
    rospy.loginfo("demo node end!")
