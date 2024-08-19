#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import os
import argparse
from sklearn.decomposition import PCA
import time
import matplotlib
matplotlib.use('TkAgg')  # Tkinterバックエンドを使用
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)

# motoda ---------------------------------------

#file_npy = "total_grasp_dataset.npy"
file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"

## Axis 3
#folder_name = "axis_3/Last5_Off_Init_grasp"
#folder_name = "axis_3/Last5_On_Init_grasp"
#folder_name = "axis_3/Sequence5_Off_Init_grasp"
#folder_name = "axis_3/Sequence5_On_Init_grasp"

## Axis 5
#folder_name = "axis_5/Last5_Off_Init_grasp"
#folder_name = "axis_5/Last5_Off_On_grasp"
#folder_name = "axis_5/Sequence5_Off_Init_grasp"
#folder_name = "axis_5/Sequence5_On_Init_grasp"

# ----------------------------------------------
dataset_path = args.dir + "/policy_sci_updown_no_zslider/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)   # (20,)で、はじめの19個は姿勢。最後の１つはachieve_goal。
print(postures.shape)


# achieved_goalの値を取得
achievedgoal_values = postures[:, -1]

# PCAを実行
pca = PCA(n_components=2)
# postures = postures[:, 1:-2]  # 17個から14個に要素を減らす(☓WRJ0, zslider, ag)
postures = postures[:, :-1]
print(postures.shape)
postures_pca = pca.fit_transform(postures)

# PC1とPC2を取得
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]

# 相関係数を計算
correlation = np.corrcoef(pc1, achievedgoal_values)[0, 1]
print(f"PC1とachieved_goalの相関係数: {correlation}")

# ランプ関数の定義
def ramp_function(x, x0, a):
    return np.piecewise(x, [x < x0, x >= x0],
                        [0, lambda x: a * (x - x0)])

# 初期パラメータの設定
initial_x0 = np.percentile(pc1, 25)  # データの25パーセンタイルを初期値とする
initial_a = (achievedgoal_values.max() - achievedgoal_values.min()) / (pc1.max() - pc1.min())  # 傾きをデータの範囲で推定
initial_params = [initial_x0, initial_a]

# フィッティング
params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)

# フィッティング結果をプロット
# plt.scatter(pc1, achievedgoal_values, label='Data')
# plt.plot(np.sort(pc1), ramp_function(np.sort(pc1), *params), label='Fitted ramp function', color='red', linewidth=2)
# plt.xlabel('PC1')
# plt.ylabel('achieved_goal')
# plt.title('PC1 vs achieved_goal with Ramp Function Fit')
# plt.legend()
# plt.show()

# フィッティングパラメータを表示
print(f"Fitted parameters: x0 = {params[0]}, a = {params[1]}")


x0, a = params

# 目標の achieved_goal (ag) 値
desired_ag = 0.6

# 対応するPC1の値を計算
pc1_value = (desired_ag / a) + x0
print(f"PC1 value for ag = {desired_ag}: {pc1_value}")

# PC1 の逆変換
pc1_vector = np.zeros((1, 2))
pc1_vector[0, 0] = pc1_value

# 逆変換で姿勢を取得
inverse_posture = pca.inverse_transform(pc1_vector)
print("Inverse posture:", inverse_posture)
print(inverse_posture.shape)


# ハンドモデルの初期関節位置を設定する関数(ランダム姿勢)
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        # sim.data.qpos[joint_idx] = joint_angle  # もともとのコード
        perturbation = math.radians(random.uniform(-0, 0))  # ros実験用ランダムなし, 修正後のコード, ロバスト性の確認用
        sim.data.qpos[joint_idx] = joint_angle + perturbation  # 各関節に±1°のランダムなラジアンを加える

# 関節名と初期角度の定義
joint_names = [#"robot0:zslider",
                "robot0:rollhinge",
                "robot0:WRJ1", "robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                # "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [# 0.04,  # はさみの穴を狭めたver
                1.57,
                0.0, 0.0,
                0.0, 1.44, 0.0, 1.57,
                0.0, 1.53, 0.0, 1.57,
                0.0, 1.44, 0.0, 1.57,
                # 0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.209, 0.0, -1.57]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])  # はさみの初期位置

# actuation_centerとactuation_rangeを使用して, PCA入力前のpostureに修正をかける！
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

def _get_achieved_goal():
    # はさみのhingeの角度の取得
    hinge_joint_angle_2 = sim.data.get_joint_qpos("scissors_hinge_2:joint")  # 正の値(はさみが開く場合の時)
    return hinge_joint_angle_2

# 記録の初期化
recorded_ags = []

# 所望のagとなる姿勢をシミュレーションで表示する
while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if t > 500:  # 500step表示
        ag_value = _get_achieved_goal()
        recorded_ags.append(ag_value)  # actual_agを記録する
        print(f"actual_ag = {ag_value}")

        if len(recorded_ags) == 10:
            # 関節の値を取得
            ffj0_angle = sim.data.get_joint_qpos("robot0:FFJ0")
            mfj0_angle = sim.data.get_joint_qpos("robot0:MFJ0")
            rfj0_angle = sim.data.get_joint_qpos("robot0:RFJ0")
            # 制御信号の値を取得
            control_values = sim.data.ctrl
            print(control_values)
            # 最終的な16個の関節値を組み合わせる
            final_joint_values = np.zeros(16)
            # 制御信号の14個の値をセット
            final_joint_values[0] = control_values[0]    # FFJ3
            final_joint_values[1] = control_values[1]    # FFJ2
            final_joint_values[2] = control_values[2]    # FFJ1
            final_joint_values[3] = ffj0_angle           # FFJ0
            final_joint_values[4] = control_values[3]    # MFJ3
            final_joint_values[5] = control_values[4]    # MFJ2
            final_joint_values[6] = control_values[5]    # MFJ1
            final_joint_values[7] = mfj0_angle           # MFJ0
            final_joint_values[8] = control_values[6]    # RFJ3
            final_joint_values[9] = control_values[7]    # RFJ2
            final_joint_values[10] = control_values[8]   # RFJ1
            final_joint_values[11] = rfj0_angle          # RFJ0
            final_joint_values[12] = control_values[9]   # THJ4
            final_joint_values[13] = control_values[10]  # THJ3
            final_joint_values[14] = control_values[12]  # THJ1
            final_joint_values[15] = -control_values[13]  # THJ0 Liteでは符号逆
            print("Final joint values:", final_joint_values)  # 最終的なハンドの姿勢, 3つ以外はactuator値
            # sim.data.get_joint_qpos(joint_names, joint_angles)# 最終的なハンドの姿勢, 全てjoint値を得る

        t = 0
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

        if len(recorded_ags) >= 32:
            recorded_ags = recorded_ags[2:32]  # はじめの2つを無視して残りの30個を取得
            break

    sim.data.ctrl[:] = actuation_center[:] + inverse_posture[0] * actuation_range[:]
    sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    # if t == 500:
    #     print(sim.data.ctrl[:])  # 目標として与えた制御信号
    #     print(sim.data.get_joint_qpos("robot0:FFJ2")) # 500step目のactuatorの値

    # time.sleep(0.005)

# # 結果をファイルに保存
# print(recorded_ags)
#
# output_dir = os.path.join(args.dir, "policy_sci_updown_no_zslider_thre1/test/")
# os.makedirs(output_dir, exist_ok=True)
# file_name = os.path.join(output_dir, f"error_with_desired_ag={desired_ag}_in_ramdom_hand_2degree.txt")
#
# with open(file_name, 'w') as file:
#     for ag in recorded_ags:
#         file.write(f"{ag}\n")
#
# print(f"Saved actual_ag values to {file_name}")
