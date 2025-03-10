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
# dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)  # RSJ用
dataset_path = args.dir + "/policy_roundscissor/{}/{}".format(folder_name, file_npy)  # 修論前用

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
# # 符号反転（必要に応じて行う）
# pc1 *= -1  # PC2 の符号を反転

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

# 元データとフィッティング結果をプロット
plt.scatter(pc1, achievedgoal_values, label='Outlier Hand posture')
plt.scatter(pc1_filtered, achievedgoal_values_filtered, label='Filtered Hand posture', color='green')
plt.plot(np.sort(pc1), ramp_function(np.sort(pc1), *params_filtered), label='Fitted ramp function', color='red', linewidth=2)
plt.xlabel('PC1', fontsize=20)
plt.ylabel('Desired scissor angle [rad]', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(left=0.125, right=0.99, top=0.98, bottom=0.125)
plt.legend(fontsize=18)
plt.show()

# フィッティングパラメータを表示
print(f"Initial fit parameters: x0 = {params[0]}, a = {params[1]}")
print(f"Filtered fit parameters: x0 = {params_filtered[0]}, a = {params_filtered[1]}")
