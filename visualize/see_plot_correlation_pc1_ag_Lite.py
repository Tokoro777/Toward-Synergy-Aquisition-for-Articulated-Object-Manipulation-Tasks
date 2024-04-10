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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_vertical_Lite.xml")
sim = MjSim(model)

# motoda ---------------------------------------

#file_npy = "total_grasp_dataset.npy"
file_npy = "grasp_dataset_on_best_policy.npy"
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
dataset_path = args.dir + "/policy_Lite/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)   # (20,)で、はじめの19個は姿勢。最後の１つはachieve_goal。
print(postures.shape)
# policy_pos_with_achievedgoal/test/grasp_dataset_290.npyの時
# # # 使用したいデータのみ選別
# # # 抜き取りたい行のインデックスリスト
# # desired_row_indices = [2, 3, 5, 6, 9, 10, 12, 14, 17, 21, 22, 23, 25, 26, 28, 29, 35, 38, 40, 43, 45, 56, 58, 59, 60, 62, 63, 64, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 93, 97, 98, 99, 100]  # Pythonのインデックスは0から始まるため、2行目はインデックス1、5行目はインデックス4
# # # 複数の行を抜き取る
# # postures = postures[desired_row_indices, :]
#
# # policy_damping=0.1/test/grasp_dataset_15.npyの時
# # desired_row_indices = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53]
# desired_row_indices = [6, 9, 10, 14, 17, 19, 20, 22, 23, 24, 27, 28, 29, 32, 34, 35, 36, 40, 43, 44, 46, 49, 50, 51, 54, 55, 59, 61, 63, 64, 67, 68, 69, 71, 72, 75, 82, 93, 96, 105, 107, 113, 114, 119, 121, 133, 138, 147, 151]
# postures = postures[desired_row_indices, :]
# print(postures.shape)



# PCAを実行
pca = PCA(n_components=2)
postures_pca = pca.fit_transform(postures)

# PC1とPC2を取得
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]

# achieved_goalの値を取得
achievedgoal_values = postures[:, -1]

# 相関係数を計算
correlation = np.corrcoef(pc2, achievedgoal_values)[0, 1]
print(f"PC2とachieved_goalの相関係数: {correlation}")

# PC1とachievedgoalの値に基づいて色分けしてプロット
plt.scatter(pc2, achievedgoal_values, cmap='viridis')
plt.xlabel('PC2')
plt.ylabel('achieved_goal')
plt.title('PC2 vs achieved_goal')
plt.colorbar(label='achieved_goal')
plt.show()
