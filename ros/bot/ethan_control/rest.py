#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
import random
import numpy as np
from math import *
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
p=pi
#from ethan_control.msg import Angles

angle=[]
with open('Angles.txt',"r") as f:
     for coord in f:
         m=coord.split()
         for i in range(len(m)):
                angle.append(float(m[i]))

x=angle[0]
y=angle[1] 
z=angle[2]

a_1=0.10 #middle arm length
a_2 = 0.10 # end arm length

q2=  acos( ( (a_1**2 + a_2**2) - (x**2 + y**2 + z**2) )/2*a_1*a_2)
print(q2)
q1= atan( (y) / (np.sqrt(x**2 + z**2)) ) + atan( (a_2*(sin(q2)) )/(a_1 + a_2*(cos(q2)) ) )
print(q1)

a=0 # Platform Height
b=1.5 #Camera Holder Angle
if q1 < p/2:    #Middle Arm Angle
   c=  p/2 - q1
   d=  c + q2  # End arm angle 
else:
   c= q1 - p/2
   d= q2 - c

print(c)
print(d)
e=  atan(z/x)  # Disk Angle
print(e)


def callback(msg):
    # if len(msg.transforms)<2:
    #     print(msg.transforms[0].transform.translation)
    print(msg.pose[1])
    # print(dir(msg.transforms[0]))
   
    # print("yo")

if __name__ =="__main__":
    rospy.init_node("publisher")
    gripper_pub = rospy.Publisher('/platform_state_controller/command', Float64MultiArray, queue_size=1)
    bot_control_pub = rospy.Publisher('/diff_drive_controller/cmd_vel',Twist,queue_size = 2)
    mat = Float64MultiArray()
    mat.layout.dim.append(MultiArrayDimension())
    mat.layout.dim.append(MultiArrayDimension())
    mat.data = [a,b,c,d,e]
    i = 0
    while not rospy.is_shutdown():
        i +=1
        gripper_pub.publish(mat)  
        if i > 100000:
            break

print("Now you can work!")


