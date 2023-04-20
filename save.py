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
            if pose is not None:
                msg = Twist()
                pose_draw()
                #print("idiot")
                if flag == "bag1":
                    cx,cy = cx1,cy1
                    k=0
                elif flag == "bag2":
                    cx,cy= cx2,cy2
                    k=1
                else:
                    continue
                if ys_no==1 and len(cnt_list)>=2:
                    sx1,sx2,sx3,sx4=cnt_list[k][0],cnt_list[k][1],cnt_list[k][2],cnt_list[k][3]
                    cv2.rectangle(rgb_image, (sx1,sx2), (sx3,sx4), (0, 0, 255), 2)
                    #cx,cy = (sx2 - sx1) // 2 + sx1, (sx4 - sx3) // 2 + sx3
                    cv2.circle(rgb_image, (cx,cy), 5, (0, 0, 255), -1)
                    #rospy.loginfo("yiooooo")
                    move1(cx, cy, msg)
                    if _depth1[cy][cx]<=10:
                        rospy.loginfo("E")
                        for i in range(10):
                            msg.linear.x = pre_x + 0.05
                            _cmd_vel.publish(msg)
                        joint1, joint2, joint3, joint4 = 0.000, 0.8, 0.0,0.0
                        set_joints(joint1, joint2, joint3, joint4, t)
                        time.sleep(t)
                        close_gripper(t)
                        step="no"
                        break
                    rospy.loginfo("ggggg")
                '''
                else:
                    if len(cnts) > 0:
