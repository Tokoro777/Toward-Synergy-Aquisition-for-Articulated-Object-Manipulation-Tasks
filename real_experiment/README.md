# Motomanの使い方～ShadowHandの使用まで

## Motoman PC how to launch
1. ターミナル表示切替
```
terminator -l motoman
```

2. motomanの電源 ON

3. コントローラ 背面をhalfPressしながら, Connectをタッチし起動させる

4. key:teachに鍵を回す

5. サーボオンボタンを押しながら, ロボットを作業基点に動かす

6. key:remoteに鍵を回す(PCで所望の位置にmotomanを動かしたい場合. motomanとPCをつなぐ.)

7. (wiki参照)
```
roslaunch motoman_sda5f_support robot_interface_streaming_sda5f.launch robot_ip:=192.168.255.1 controller:=fs100
```

8. Turn on servo カチッとmotomanから音が鳴る(wiki参照)
```
rosservice call robot_enable
```

9. motomanの動きをシミュレーションで確認できる(only sim). demo.launchにすると, シミュレーションだけでなく実機も動くので注意(sim＆real).
```
roslaunch sda5fshl_moveit_config demo_fake.launch
```



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
