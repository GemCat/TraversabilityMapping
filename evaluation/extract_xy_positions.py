import rospy
import tf

if __name__ == '__main__':

    currX = 0
    currY = 0

    rospy.init_node('pose_extractor', anonymous=True)
    listener = tf.TransformListener()

    rospy.sleep(2)

    with open('/home/rsl/harveri_transforms/trail_xy_poses.txt', 'a') as f:
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            try:
                # Get the latest transform available
                (trans, rot) = listener.lookupTransform('/odom', '/BASE', rospy.Time(0))
                if currX != trans[0] or currY != trans[1]: # avoid repeating the same pose (eg. if robot stops)
                    print("getting translation")
                    f.write(f"{trans[0]} {trans[1]}\n")
                    currX = trans[0]
                    currY = trans[1]

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            rate.sleep()
