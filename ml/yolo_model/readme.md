
follow the steps to generate ssh key in your github account:-
\
https://support.atlassian.com/bitbucket-cloud/docs/set-up-an-ssh-key/
\
clone the repo yolo_darknet via SSH after setting up ssh in your github account:
\
cd catkin_workspace/src
git clone --recursive git@github.com:leggedrobotics/darknet_ros.git
catkin_make -DCMAKE_BUILD_TYPE=Release
\
download weights from following link and add it to the repo at location:
\
catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights/
\
yolo tiny weights:
\
https://drive.google.com/file/d/1yW7MvsKhlJg3W4dD9YL6cU6GtF_iVoP0/view?usp=sharing
\
download the file 'yolov3-tiny-obj.cfg' from this folder and add it to the location:
\
catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/cfg/
\
replace the files yolo_v3.launch and darknet_ros.launch from this folder to :
\
catkin_workspace/src/darknet_ros/darknet_ros/launch/

