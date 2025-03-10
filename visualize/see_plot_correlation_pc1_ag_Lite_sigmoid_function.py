#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # Tkinterバックエンドを使用
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_vertical_Lite.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

# データセットのパス設定
file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)

# データの読み込み
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
def sigmoid_function(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# 初期パラメータの設定 (シグモイド関数のために調整)
initial_a = achievedgoal_values.max() - achievedgoal_values.min()
initial_b = 1  # スケール調整の初期値
initial_c = np.median(pc1)  # 中央値を初期値に設定
initial_params = [initial_a, initial_b, initial_c]

# シグモイド関数で最初のフィッティング
params, params_covariance = curve_fit(sigmoid_function, pc1, achievedgoal_values, p0=initial_params)

# 残差を計算
residuals = achievedgoal_values - sigmoid_function(pc1, *params)
std_dev = np.std(residuals)

# 外れ値を除外（残差が2標準偏差以内のデータのみ使用）
mask = np.abs(residuals) < 2 * std_dev
pc1_filtered = pc1[mask]
achievedgoal_values_filtered = achievedgoal_values[mask]

# 外れ値を除外したデータで再度フィッティング
params_filtered, params_covariance_filtered = curve_fit(
    sigmoid_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
)

# 元データとフィッティング結果をプロット
plt.scatter(pc1, achievedgoal_values, label='Original Hand posture data')
plt.scatter(pc1_filtered, achievedgoal_values_filtered, label='Filtered Hand posture data', color='green')
plt.plot(np.sort(pc1), sigmoid_function(np.sort(pc1), *params_filtered), label='Filtered fitted sigmoid function', color='red', linewidth=2)
plt.xlabel('PC1', fontsize=20)
plt.ylabel('Desired scissor angle [rad]', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(left=0.125, right=0.99, top=0.98, bottom=0.125)
plt.legend()
plt.show()

# フィッティングパラメータを表示
print(f"Initial fit parameters: a = {params[0]}, b = {params[1]}, c = {params[2]}")
print(f"Filtered fit parameters: a = {params_filtered[0]}, b = {params_filtered[1]}, c = {params_filtered[2]}")
