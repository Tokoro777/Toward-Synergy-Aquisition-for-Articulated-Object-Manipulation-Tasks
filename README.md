# HumanWareFundemental: Sysngy Team
This repository is cloned from [openai/baselines](https://github.com/openai/baselines) and modifided for our reseach. Don't make PR for ofiginal repogitory.


## Train model with DDPG
以下のコマンドで学習済みモデルを作成する. tensorflowのモデルを保存するディレクトリを`--lodir_tf` で指定する.

例
```
python -m baselines.her.experiment.train \
       --env GraspBlock-v0 \
       --num_cpu 1 \
       --n_epochs 100 \
       --logdir_tf < Dierctory path to save tensorflow model>
```


## Action and Q-value Generation
以下のコマンドで学習モデルをロードし, 指定したディレクトリにアクションなどを書き出す. `--logdir_tf`で学習済みのモデルを指定し, `--logdir_aq`でactionやQ-valueなどを出力するディレクトリを指定する.


```
python -m baselines.her.experiment.test \
       --env GraspBlock-v0 \
       --num_cpu 1 --n_epochs 5 \
       --logdir_tf < path to saved model > \
       --logdir_aq < path to save actions etc... >
```

### Log File
ログファイルには以下の項目が記述されている.

+ `goal/desired`: ゴール (`g`)
+ `goal/achieved`: 到達点 (`ag`)
+ `observation`: 観測 (`o`)
+ `action`: action, shape=[EpisodeNo, Batch, Sequence, env.action_space]
+ `Qvalue`: Q-value, shape=[EpisodeNo, Batch, Sequence, env.action_space]
+ `fc`: Critic Networkの中間出力 (fc2), shape=[EpisodeNo, Batch, Sequence, n_unit(=256)]





--------------------------------------
## Memo
TBA


----------------------------------------
## Initial Setup
### Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 


### Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

- Install original environment

```bash
cd gym-grasp
pip install -e .
```



### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)




# 実行コードの手順
## シミュレーション
1. はさみ操作の学習
```
python3 grasp_object.py
``` 
2. PC1やPC2主成分軸に沿った, ハンドの動きを可視化
```
python3 see_pca_axis_Lite_14_elements_scissors_updown.py
```
3. 姿勢データセットを元に, はさみを操作し, その時の角度を取得することで, 姿勢と角度こみの新しいデータセットを作成.(PC1と角度の関係を調べるため)
```
python3 get_new_dataset_for_ag_with_sci_updown_no_zslider.py
```
4. PC1とはさみ角度の相関をプロット
```
python3 see_plot_correlation_pc1_ag_Lite.py
```
5. 新しいデータセット(姿勢と角度)を使用して,277個の操作を再生. この時, どのような姿勢が, どの主成分と角度に当たるのかが確認できる
```
python3 replay_dataset_Lite_pc1_ag_with_sci_updown_no_zslider_new_ag.py
```
6. 相関から同定したランプ関数で, 所望の角度に対応するPC1値で制御し,達成角度を保存
```
python3 change_posture_by_ag_pc1_ramp_function_sci_updown.py
```
7. 保存した達成角度と目標角度との誤差を箱ひげ図で評価
```
python3 evaluate_error_with_ag_in_boxplot.py
```
8. ROS用にジョイントの値を取得
```
python3 get_joint_value_for_ros.py
```
9. ピクルファイルを作成
```
python3 create_pickle.py
```
10. Shadow Hand Liteの動作を制御
```
python3 create_pickle.py
```


## 実機
1. はさみ操作の初期姿勢にShadow Hand Liteを移動
```
python3 operate_lite_initial_config_jointtrajectory.py
```
2. 単一の目標値に対して, フィードバック制御(PID)を用いたジョイント軌道で操作
```
python3 operate_lite_feedback_control_jointtrajectory.py
```
3. 連続的に変化する目標値に対して, フィードバック制御(PID)によるジョイント軌道で操作
```
python3 operate_lite_feedback_control_jointtrajectory_sequence.py
```
4. サイン波の目標値に対して, フィードバック制御(PID)を使用して操作
```
python3 operate_lite_feedback_control_jointtrajectory_sinwave.py
```

## ディレクトリ構造
リポジトリTOP
│
├ README.md .. 目次を書く
│
├ ○○.md
├ ○○.md
├ ○○.md
│
├ 第１章 .. フォルダを作成
│　├ ○○.md
│　├ ○○.md
│　└ ○○.md
│
├ 第２章 .. フォルダを作成





