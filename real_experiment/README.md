# Motomanの使い方～ShadowHandの使用まで

## Motoman PC how to launch
terminator -l motoman

cd catkin_ws/

cd src/

git clone git@github.com:mlkakram/sda5f_shl.git

catkin build

cd ../

source devel/setup.bash

///////////
roslaunch motoman_sda5f_support robot_interface_streaming_sda5f.launch robot_ip:=192.168.255.1 controller:=fs100
///////////
rosservice call robot_enable
///////////
roslaunch sda5fshl_moveit_config demo_fake.launch
///////////


## ShadowHandの制御
