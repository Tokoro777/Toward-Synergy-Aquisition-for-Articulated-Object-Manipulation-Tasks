# Motomanの使い方～ShadowHandの使用まで

## Motomanの起動
1. motomanの電源 ON

2. コントローラ 背面をhalfPressしながら, Connectをタッチし起動させる

3. key:teachに鍵を回す

4. サーボオンボタンを押しながら, ロボットを作業基点に動かす

## motomanをPCと繋ぎ, PCから動かす
1. key:remoteに鍵を回す(PCで所望の位置にmotomanを動かしたい場合. motomanとPCをつなぐ.)

2. ターミナル表示切替(分割されたterminatorが表示される)
```
terminator -l motoman
```

3. 一番左のターミナルで記述すること
```
cd catkin_ws/
```

```
cd src/
```

これは不要の可能性あり
```
git clone git@github.com:mlkakram/sda5f_shl.git
```

```
catkin build
```

```
cd ../
```

```
source devel/setup.bash
```

4. 右側の別のターミナルに記述(wiki参照)
```
roslaunch motoman_sda5f_support robot_interface_streaming_sda5f.launch robot_ip:=192.168.255.1 controller:=fs100
```

5. また別のターミナルに記述. Turn on servo. カチッとmotomanから音が鳴る！！(wiki参照)
```
rosservice call robot_enable
```

6. motomanの動きをシミュレーションで確認できる(only sim). demo.launchにすると, シミュレーションだけでなく実機も動くので注意！！(sim＆real). あるいは, Planでシミュレーション確認をして, その後Executeで実機を動かすでも可.
```
roslaunch sda5fshl_moveit_config demo_fake.launch
```


## ShadowHandの制御
