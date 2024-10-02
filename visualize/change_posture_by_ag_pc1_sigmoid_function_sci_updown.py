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

# データセットパス設定
file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)

viewer = MjViewer(sim)
t = 0
postures = np.load(dataset_path)
print(postures.shape)

# achieved_goalの値を取得
achievedgoal_values = postures[:, -1]

# PCAを実行
pca = PCA(n_components=2)
postures = postures[:, :-1]
postures_pca = pca.fit_transform(postures)

# PC1とPC2を取得
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]

# 相関係数を計算
correlation = np.corrcoef(pc1, achievedgoal_values)[0, 1]
print(f"PC1とachieved_goalの相関係数: {correlation}")

# シグモイド関数の定義
def sigmoid_function(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

# 初期パラメータの設定
initial_x0 = np.median(pc1)  # 中央値を初期値とする
initial_k = 1  # 初期勾配
initial_params = [initial_x0, initial_k]

# 初期フィッティング
params, params_covariance = curve_fit(sigmoid_function, pc1, achievedgoal_values, p0=initial_params)

# 残差の計算
residuals = achievedgoal_values - sigmoid_function(pc1, *params)
std_dev = np.std(residuals)

# 外れ値を除外 (残差が2標準偏差以内のデータを使用)
mask = np.abs(residuals) < 2 * std_dev
pc1_filtered = pc1[mask]
achievedgoal_values_filtered = achievedgoal_values[mask]

# フィルタリングされたデータで再フィッティング
params_filtered, params_covariance_filtered = curve_fit(
    sigmoid_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
)

# フィッティングパラメータを表示
print(f"初期フィットパラメータ: x0 = {params[0]}, k = {params[1]}")
print(f"フィルタリング後のフィットパラメータ: x0 = {params_filtered[0]}, k = {params_filtered[1]}")

x0, k = params_filtered

# 目標の achieved_goal (ag) 値
desired_ag = 1e-6

# 対応するPC1の値を計算
pc1_value = x0 - np.log((1 / desired_ag) - 1) / k
print(f"PC1 value for ag = {desired_ag}: {pc1_value}")

# PC1 の逆変換
pc1_vector = np.zeros((1, 2))
pc1_vector[0, 0] = pc1_value

# 逆変換で姿勢を取得
inverse_posture = pca.inverse_transform(pc1_vector)
print("Inverse posture:", inverse_posture)

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        perturbation = math.radians(random.uniform(-1, 1))
        sim.data.qpos[joint_idx] = joint_angle + perturbation

# 関節名と初期角度の定義
joint_names = [
    "robot0:rollhinge", "robot0:WRJ1", "robot0:WRJ0",
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
]

joint_angles = [
    1.57, 0.0, 0.0,
    0.0, 1.44, 0.0, 0.0,
    0.0, 1.53, 0.0, 0.0,
    0.0, 1.44, 0.0, 0.0,
    0.0, 1.22, 0.0, 0.0, 0.0
]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])

# actuation_centerとactuation_rangeを使用して, PCA入力前のpostureに修正をかける
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

def _get_achieved_goal():
    hinge_joint_angle_2 = sim.data.get_joint_qpos("scissors_hinge_2:joint")
    return hinge_joint_angle_2

# 記録の初期化
recorded_ags = []

# シミュレーションループ
while True:
    viewer.render()
    t += 1
    sim.step()

    if t > 500:
        ag_value = _get_achieved_goal()
        recorded_ags.append(ag_value)
        print(f"actual_ag = {ag_value}")
        t = 0
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

        if len(recorded_ags) >= 32:
            recorded_ags = recorded_ags[2:32]
            break

    sim.data.ctrl[:] = actuation_center[:] + inverse_posture[0] * actuation_range[:]
    sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

# 結果をファイルに保存
output_dir = os.path.join(args.dir, "policy_sci_updown_no_zslider_only_third_bend/test/")
os.makedirs(output_dir, exist_ok=True)
file_name = os.path.join(output_dir, f"error_with_desired_ag={desired_ag}_in_ramdom_hand_1degree_sigmoid.txt")

with open(file_name, 'w') as file:
    for ag in recorded_ags:
        file.write(f"{ag}\n")

print(f"Saved actual_ag values to {file_name}")
