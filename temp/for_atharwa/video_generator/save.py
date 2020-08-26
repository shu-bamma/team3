#!/usr/bin/env python
import cv2
import rospy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
rospy.init_node('realtime_test', anonymous=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
im = None
def image_callback(img_msg):
	global im
	try:
		im=bridge.imgmsg_to_cv2(img_msg,"bgr8")
	except CvBridgeError,e:
		rospy.logerr("CvBridgeError: {0}".format(e))

	

	
		# exit(0)

bridge=CvBridge()
sub_image = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
while True:
	if im is not None:
		cv2.imshow("test",im)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
out.release()
cv2.destroyAllWindows()