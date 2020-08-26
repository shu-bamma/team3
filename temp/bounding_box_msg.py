#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Header
from std_msgs.msg import String


def callback(data):
    for box in data.bounding_boxes:
        rospy.loginfo("Xmin: {}, Xmax: {} Ymin: {}, Ymax: {}".format(
                box.xmin, box.xmax, box.ymin, box.ymax
            )
        )

    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)


def main():
    while not rospy.is_shutdown():
        rospy.init_node('bbmsg', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes , callback)
        rospy.spin()


if __name__ == '__main__':
    try :
        main()
    except rospy.ROSInterruptException:
        pass
