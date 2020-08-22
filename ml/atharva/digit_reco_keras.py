#!/usr/bin/env python
import cv2
from skimage.feature import hog
import numpy as np
import sys
from PIL import Image, ImageOps
import PIL.Image
from skimage import transform
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

Dict={'red':None,'green':None,'yellow':None,'blue':None,'orange':None}

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
model.load_weights('test.h5')

def load(im):

   nk=np.zeros((1,28,28,1),dtype='float')
   np_image = np.array(im).astype('float32')/255
   nk[0,:,:,0]=np_image
#np_image = transform.resize(np_image, (28, 28, 1))
   #np_image = np.expand_dims(np_image, axis=0)
   return nk
def predictor(model_name,img):
  global model
  image = load(img)
  res=model.predict(image)
  i=np.argmax(res, axis=1)
  prob=res[0,i]
  return i,prob
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
      
def image_callback(img_msg):

	boundaries={'blue':([110,150,150], [130,255,255]),
            'orange':([15,150,150], [20,255,255]),
            'red':([0,150,150], [7,255,255]),
            'yellow':([25,150,150], [35,255,255]),
            'green':([55,150,150], [65,255,255])   
           }

    	        

	#read the converted input image
	global Dict
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
		cls , prob = predictor('my_model.h5',roi)
		if prob>0.7:
				
			cv2.putText(im, str(cls)+' '+str(color), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 2)

		if predictor('my_model.h5',roi)!=None:
			Dict[str(color)]=str(predictor('my_model.h5',roi))
	
	cv2.imshow('img',im)
    #show  the frame with detection
	cv2.waitKey(3)		
if __name__ == '__main__':
	try:
		global Dict
		# Load the classifier
		# clf=joblib.load("cls.pkl")
		#initialise the ros node
		rospy.init_node('realtime_test', anonymous=True)
		# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
		sub_image = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
		# Initialize the CvBridge class
		bridge=CvBridge()
		if any([v==None for v in Dict.values()])==True:
			rospy.spin()
		else:
			file1=open("code.txt","w")
			file1.write(Dict['red']+'\n'+Dict['green']+'\n'+Dict['yellow']+'\n'+Dict['blue']+'\n'+Dict['orange'])
			sys.exit()
	except rospy.ROSInterruptException:
		rospy.loginfo("node terminated")
