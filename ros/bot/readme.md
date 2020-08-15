 **For launching the bot**

`roslaunch ethan_gazebo ethan.launch`


**For navigation of the bot**

Edit the explore.launch file in explore-lite package (which is installed on your laptop)

And edit its first line to this : 
`<param name="robot_base_frame" value="link_chassis"/>`

