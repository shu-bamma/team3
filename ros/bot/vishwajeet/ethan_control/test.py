#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
rospy.init_node("depth_data",anonymous=True)
data = rospy.wait_for_message("/camera/depth/points",PointCloud2)
gen = list(point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=False))
# print(gen[10)