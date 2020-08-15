 Copy this three folders to your ros workspace

 **For launching the bot**

`roslaunch ethan_gazebo ethan.launch`

`rosrun ethan_control rest.py`
 (this command is neccessary for to make gripper in rest position)

**For navigation of the bot**

Edit the explore.launch file in explore-lite package (which is installed on your laptop)

Edit its first line to this : 
`<param name="robot_base_frame" value="link_chassis"/>`

And lauch the navigation by command : 
`roslaunch ethan_gazebo ethan.launch`

 **For controlling the arm**

end arm length : 90mm, middle arm length : 100m

 you can refer rest.py in ethan_control package

 mat.data = [a,b,c,d,e]

 a = platform position

 a = camera holder angle

 c = middle arm angle

 d = end arm angle

 e = disc angle