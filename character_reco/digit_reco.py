# Import the modules
import cv2
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
#function for asserting color configuratoin
def assert_color(img):
    ar=np.array(img,dtype='int')
    cnt=0
    for i in range(ar.shape[0]):
        for j in range(ar.shape[1]):
            if ar[i,j]==255:
                cnt=1
                break

    if cnt>0:
        return 1
    else:
        return 0
# Load the classifier
clf = joblib.load("digits_cls (1).pkl" )
capture = cv2.VideoCapture("./autonomous_tilted_camera.avi")


boundaries={'blue':([110,50,50], [130,255,255]),
            'red':([0,50,50], [10,255,255]),
            'yellow':([25,50,50], [35,255,255]),
            'orange':([11,50,50], [24,255,255]),
            'green':([35,50,50], [75,255,255])   
           }

while(True):
     
    ret, frame = capture.read()
    # Read the input image 
    im = frame
    
    
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)
    th3 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,31,2)
    # Find contours in the image
    _,ctrs,_ = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour

    # For each rectangular region, calculate HOG features and predict the digits
    rects=list()
    for ctr in ctrs:
        area = cv2.contourArea(ctr)
        if area>500 and area<10000:
            rect=cv2.boundingRect(ctr)
            if rect[2]/rect[3]<4 and rect[3]/rect[2]<4:
                if rect[2]<400:
                    rects.append(rect)
        approx = cv2.approxPolyDP(ctr, 0.01*cv2.arcLength(ctr, True), True)
        
      
        # Draw the rectangles
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
            # find the colors within the specified boundaries and apply
                # the mask
            mask = cv2.inRange(hsv, lower, upper)

            if cv2.countNonZero(mask) != 0:
                color=key
                break
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
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0]))+' '+str(color), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('img',im)
    if cv2.waitKey(100) == 27:
            break
 
capture.release()
cv2.destroyAllWindows()
