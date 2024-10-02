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

file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"

# ----------------------------------------------
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)   # (20,)で、はじめの19個は姿勢。最後の１つはachieve_goal。
print(postures.shape)

# achieved_goalの値を取得
achievedgoal_values = postures[:, -1]

# PCAを実行
pca = PCA(n_components=2)
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
initial_a = (achievedgoal_values.max() - achievedgoal_values.min()) / (pc1.max() - pc1.min())
initial_params = [initial_x0, initial_a]

# 最初のフィッティング
params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)

# 残差を計算
residuals = achievedgoal_values - ramp_function(pc1, *params)
std_dev = np.std(residuals)

# 外れ値を除外（残差が2標準偏差以内のデータのみ使用）
mask = np.abs(residuals) < 2 * std_dev
pc1_filtered = pc1[mask]
achievedgoal_values_filtered = achievedgoal_values[mask]

# 外れ値を除外したデータで再度フィッティング
params_filtered, params_covariance_filtered = curve_fit(
    ramp_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
)

# desired_goal = 0.6のときの姿勢に移動
desired_ag = 0.6
x0, a = params_filtered
pc1_value = (desired_ag / a) + x0
print(pc1_value)

pc1_vector = np.zeros((1, 2))
pc1_vector[0, 0] = pc1_value
inverse_posture_1 = pca.inverse_transform(pc1_vector)

# desired_goal = 0.0のときの姿勢に移動
desired_ag = 0.0
pc1_value = (desired_ag / a) + x0
print(pc1_value)
pc1_value = -1.5  # 完全にはさみを閉じきれる時のPC1値
print(pc1_value)
pc1_vector[0, 0] = pc1_value
inverse_posture_2 = pca.inverse_transform(pc1_vector)

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        perturbation = math.radians(random.uniform(-0, 0))  # ロバスト性確認用
        sim.data.qpos[joint_idx] = joint_angle + perturbation

# 関節名と初期角度の定義
joint_names = ["robot0:rollhinge",
               "robot0:WRJ1", "robot0:WRJ0",
               "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
               "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
               "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
               "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [1.57, 0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.53, 0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])

# actuation_centerとactuation_rangeを使用して, PCA入力前のpostureに修正をかける
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

def _get_achieved_goal():
    hinge_joint_angle_2 = sim.data.get_joint_qpos("scissors_hinge_2:joint")
    return hinge_joint_angle_2

# シミュレーションループ
while True:
    viewer.render()
    t += 1
    sim.step()
    print(_get_achieved_goal())

    if t <= 500:  # 500stepまでdesired_goal=0.6 の姿勢へ
        sim.data.ctrl[:] = actuation_center[:] + inverse_posture_1[0] * actuation_range[:]
        sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    elif t > 500 and t <= 1000:  # 500step超えたらdesired_goal=0.0 の姿勢へ
        sim.data.ctrl[:] = actuation_center[:] + inverse_posture_2[0] * actuation_range[:]
        sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    elif t > 1000:  # 1000step超えたら姿勢を初期化
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)
        t = 0  # リセット
