#!/usr/bin/env python

import numpy as np
import time
import cv2
import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
import time

# from sklearn.externals import joblib

# import tensorflow.keras as keras

#function for asserting color configuratoin
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError


"""
def assert_color(img):
    ar=np.array(img,dtype='int')
    cnt=0
    for i in range(ar.shape[0]):
        for j in range(ar.shape[1]):
            if ar[i,j]==255:
                cnt=cnt+1
                if cnt>(ar.shape[0]*ar.shape[1]/100):
                    return 1
                else:
                    continue
    return 0
"""
r={0:[0,0,0,0],1:[0,0,0,0],2:[0,0,0,0],3:[0,0,0,0],4:[0,0,0,0],5:[0,0,0,0],6:[0,0,0,0],7:[0,0,0,0],8:[0,0,0,0],9:[0,0,0,0]}
yxz={0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0],5:[0,0,0],6:[0,0,0],7:[0,0,0],8:[0,0,0],9:[0,0,0]}

def get_centres(rect):
	a=[]
	a.append(rect[0]+rect[2]/2)
	a.append(rect[1]+rect[3]/2)
	return a


def callback_pointcloud(data):
    global r
    global xyz
    assert isinstance(data, PointCloud2)
    gen = list(point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=False))
    for i in range(10):
    	uv=get_centres(r[i])
    	po=(uv[0]*640+uv[1])-1
    	xyz[i]=[gen[po][0],gen[po][1],gen[po][2]]
    	#xyz coordinates being stored corresponding to the numbers
    time.sleep(1)


def image_callback(img_msg):
	

	
	#read the converted input image
	try:
		im=bridge.imgmsg_to_cv2(img_msg,"bgr8")
	except CvBridgeError,e:
		
		rospy.logerr("CvBridgeError: {0}".format(e))
	x_list=[]
	y_list=[]



	# Convert to grayscale and apply Gaussian filtering
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	ret, im_th = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY_INV)
	# Threshold the image
	th3 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		    cv2.THRESH_BINARY_INV,11,2)

	    
	_,ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	    
	rects=list()
	for ctr in ctrs:
		area = cv2.contourArea(ctr)
		if area>3000 and area<8000:
		    rect=cv2.boundingRect(ctr)
		    if rect[2]/rect[3]<2 and rect[3]/rect[2]<2:
		        rects.append(rect)
		        print(rect[0], ' ' ,rect[1])

			if len(x_list)<=12:

		        	x_list.append(rect[0])
		        	y_list.append(rect[1])
		# Draw the rectangles
	   

	x_list.sort()
	y_list.sort()
	def get_key(rect):
		if rect[0] in range(x_list[0]-15,x_list[0]+15):
		    if rect[1] in range(y_list[0]-15,y_list[0]+15):
		        return '1'
		    elif rect[1] in range(y_list[4]-15,y_list[4]+15):
		        return '5'
		    elif rect[1] in range(y_list[8]-15,y_list[8]+15):
		        return ''
		if rect[0] in range(x_list[3]-15,x_list[3]+15):
		    if rect[1] in range(y_list[0]-15,y_list[0]+15):
		        return '2'
		    elif rect[1] in range(y_list[4]-15,y_list[4]+15):
		        return '6'
		    elif rect[1] in range(y_list[8]-15,y_list[8]+15):
		        return '9'
		if rect[0] in range(x_list[6]-15,x_list[6]+15):
		    if rect[1] in range(y_list[0]-15,y_list[0]+15):
		        return '3'
		    elif rect[1] in range(y_list[4]-15,y_list[4]+15):
		        return '7'
		    elif rect[1] in range(y_list[8]-15,y_list[8]+15):
		        return '0'
		if rect[0] in range(x_list[9]-15,x_list[9]+15):
		    if rect[1] in range(y_list[0]-15,y_list[0]+15):
		        return '4'
		    elif rect[1] in range(y_list[4]-15,y_list[4]+15):
		        return '8'
		    elif rect[1] in range(y_list[8]-15,y_list[8]+15):
		        return ''


	for rect in rects:
		# Draw the rectangles
		if len(rects)==12:

			cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
		"""print(rect[0], ' ' ,rect[1])
		imCrop = im[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
	 
	   
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
		# Resize the image
		try:
		   
		    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
		    roi = cv2.dilate(roi, (3, 3))
		    
		except:
		    continue
		# Calculate the HOG features
		roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
		nbr = clf.predict(np.array([roi_hog_fd], 'float64'))"""
		if(len(rects)<13):
			#storing the rectangles with the corresponding numbers
			global r
		    r[int(get_key(rect))]=[rect[0],rect[1],rect[2],rect[3]]
			cv2.putText(im, get_key(rect), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	cv2.imshow('img',im)
    #show  the frame with detection
	cv2.waitKey(3)		
if __name__ == '__main__':
	try:
		# Load the classifier
		# clf=joblib.load("cls.pkl")
		#initialise the ros node
		rospy.init_node('realtime_test', anonymous=True)
		# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
		sub_image = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
		# Initialize the CvBridge class
		bridge=CvBridge()
		#for converting to xyz
		rospy.init_node('pcl_listener', anonymous=True)
		rospy.Subscriber('/camera/depth/points', PointCloud2, callback_pointcloud)
		rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("node terminated")
