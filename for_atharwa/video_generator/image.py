#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import rospy
import cv2



def image_callback(img_msg):

	try:
		im=bridge.imgmsg_to_cv2(img_msg,"bgr8")
	except CvBridgeError,e:
		rospy.logerr("CvBridgeError: {0}".format(e))

	cv2.imwrite("/home/vishwajeet/Desktop/RealTime-DigitRecognition/test.png",im)
	cv2.imshow("test",im)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("yo")
        


rospy.init_node('realtime_test', anonymous=True)
		# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
		# Initialize the CvBridge class
bridge=CvBridge()
rospy.spin()
cv2.destroyAllWindows()
