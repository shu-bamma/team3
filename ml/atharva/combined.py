#!/usr/bin/env python
import cv2
from skimage.feature import hog
import numpy as np
import time
import cv2
import os
import darknet
# from sklearn.externals import joblib
import joblib
# import tensorflow.keras as keras
from tensorflow.keras.models import load_model
#function for asserting color configuratoin
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# model = tf.keras.models.load_model("test12.h5")
image_height = 28
image_width = 28
num_channels = 1
num_classes = 10

def build_model():
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.load_weights('/home/vishwajeet/catkin_ws/src/ethan_control/test.h5')

def predictor(model_name,img):
  global model
  nl=np.array(img,dtype='float')
  nk=np.zeros((1,28,28,1),dtype='float')
  nk[0,:,:,0]=nl
  res=model.predict_on_batch(nk)
  res=np.around(res)
  for i in range(res.shape[1]):
    if res[0,i]==1:

      print(i)
      return i
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
labelsPath = os.path.join("obj.names")
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.join("yolov3_custom_train_500.weights")
configPath = os.path.join("yolov3_custom_train.cfg")
# Loading the neural network framework Darknet (YOLO was created based on this framework)
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
def predict(image):
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.2
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = (255,0,0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)
    return image
    """
def image_callback(img_msg):

	boundaries={{'blue':([110,50,50], [130,255,255]),
            'orange':([15,50,50], [20,255,255]),
            'red':([0,150,50], [10,255,255]),
            'yellow':([25,150,50], [35,255,255]),
            'green':([45,150,50], [65,255,255])   
           }   
    	        }

	
	#read the converted input image
	try:
		im=bridge.imgmsg_to_cv2(img_msg,"bgr8")
	except CvBridgeError,e:
		rospy.logerr("CvBridgeError: {0}".format(e))

	# Convert to grayscale and apply Gaussian filtering
	im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray=cv2.GaussianBlur(im_gray, (5,5), 0)  


	# Threshold the image
	_,im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)
	th3 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		cv2.THRESH_BINARY_INV,11,2)
		# Find contours in the image
	_,ctrs,_ = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects=list()
		# Get rectangles contains each contour
    # For each rectangular region, calculate HOG features and predict the digits
	for ctr in ctrs:
		area = cv2.contourArea(ctr)
		if area>500 and area<10000:
			rect=cv2.boundingRect(ctr)
			if rect[2]/rect[3]<4 and rect[3]/rect[2]<4:
				if rect[2]<400:
					rects.append(rect)
	for rect in rects:
		# Draw the rectangles 
		color=''
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
		imCrop = im[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
		hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
		for key,value in boundaries.items():
			# create NumPy arrays from the boundaries
			lower = np.array(value[0], dtype = "uint8")
			upper = np.array(value[1], dtype = "uint8")
			mask = cv2.inRange(hsv, lower, upper)
			# find the colors within the specified boundaries and apply
            # the mask
			if assert_color(mask)== 1:     #alternative: if cv2.countNonZero(mask) != 0:
				color=key
				break
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        #resize the image
		try:
			roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
			roi = cv2.dilate(roi, (3, 3))
		except:
			continue
        # Calculate the HOG features
		cv2.putText(im, str(predictor('my_model.h5',roi))+' '+str(color), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 2)
        im=detect(net,meta,im)
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
		sub_image = rospy.Subscriber("/camera1/image_raw", Image, image_callback)
		# Initialize the CvBridge class
		bridge=CvBridge()
		rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("node terminated")
