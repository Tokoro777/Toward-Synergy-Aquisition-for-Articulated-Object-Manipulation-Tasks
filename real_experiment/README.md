# Motomanの使い方

### Motomanの起動
1. motomanの電源 ON

2. コントローラ 背面をhalfPressしながら, Connectをタッチし起動させる

3. key:teachに鍵を回す

4. ロボットR2orR1から作業基点選択, サーボオンボタンを押して, コントローラ背面とreadyボタンを押しながら, ロボットを作業基点に動かす

### motomanをPCと繋ぎ, PCから動かす
1. key:remoteに鍵を回す. (PCで所望の位置にmotomanを動かしたい場合. motomanとPCをつなぐ.)

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

6. また別のターミナルに記述. gazeboが起動. motomanの動きをシミュレーションで確認できる(only sim). demo.launchにすると, シミュレーションだけでなく実機も動くので注意！！(sim＆real). あるいは, Planでシミュレーション確認をして, その後Executeで実機を動かすでも可.
```
roslaunch sda5fshl_moveit_config demo_fake.launch
```

7. ハンドの7関節の値をあらかじめ保存したpythonコードで, motomanを動かす(私の場合). Planでシミュレーション確認をして, その後Executeで実機を動かすでも可.


# ShadowHandの使い方

### ShadowHandの起動
1. ShadowHandの文字が印字された黒い電源箱に, 電源ケーブルを刺して, 電源 Onに. 

2. ShadowHand本体から垂れているコード(先が丸い金属)と, もう一つの先が金属のコードとを接続. この時, 2つのコードのそれぞれの赤い印が揃うように接続する. 奥まで差し込むと, ハンドの指ががちゃがちゃと動く. これで接続完了.

### ShadowHandとコードで接続されたPCを用いて, 実際にShadowHandを制御する
1. PCデスクトップの Launcher を開ける.

2. Launcherファイル中の, launch_shadow_hand_right desktopを押し, dockerを起動させる. 4つのターミナルとgazeboが起動する. gazeboの起動まで待機.

3. これもPlanとExecuteがある. 壊れないように, Simで確認.

4. ターミナルで, 用意したpythonファイルを実行し, ShadowHandを制御する.

   例えばvscodeの, attach con... dextrous.....sr_graspにある, grasplite.pyを実行する. この時, ターミナルで, grasplite.pyのあるディレクトリまで, roscdで指定. vscodeの左でディレクトリ構造は分かる.

### ShadowHand終了の仕方
1. 指が閉じた状態に戻して終了する必要がある. gazeboで, Goal state:finger_packにして, Planする. Simで指が閉じたことが確認できたら, Executeで実機を動かす.

2. ターミナルをすべて, ctrl+何かで, 閉じる.

3. すべて閉じたら, Launcherファイル中にある, ShadowHand Close desktopをクリックすることで, dockerを終了.

4. ShadowHand本体の金属のコード接続を抜く.

5. ShadowHandの文字が印字された黒い電源箱の, 電源ボタンで電源 OFFにする.

6. 電源箱から, 電源ケーブルを抜く.
