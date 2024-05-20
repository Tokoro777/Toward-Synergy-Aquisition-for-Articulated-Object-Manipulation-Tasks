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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_vertical_Lite.xml")
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
dataset_path = args.dir + "/policy_without_WRJ1J0/{}/{}".format(folder_name, file_npy)
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
initial_params = [0.1, 1]  # 適切に初期値を設定

# フィッティング
params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)

# フィッティング結果をプロット
plt.scatter(pc1, achievedgoal_values, label='Data')
plt.plot(np.sort(pc1), ramp_function(np.sort(pc1), *params), label='Fitted ramp function', color='red', linewidth=2)
plt.xlabel('PC1')
plt.ylabel('achieved_goal')
plt.title('PC1 vs achieved_goal with Ramp Function Fit')
plt.legend()
plt.show()

# フィッティングパラメータを表示
print(f"Fitted parameters: x0 = {params[0]}, a = {params[1]}")