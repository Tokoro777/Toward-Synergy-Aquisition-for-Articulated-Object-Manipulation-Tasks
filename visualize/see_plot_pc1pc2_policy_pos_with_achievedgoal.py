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

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_vertical.xml")
sim = MjSim(model)

# motoda ---------------------------------------

#file_npy = "total_grasp_dataset.npy"
file_npy = "grasp_dataset_290.npy"
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
dataset_path = args.dir + "/policy_pos_with_achievedgoal/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)   # (20,)で、はじめの19個は姿勢。最後の１つはachieve_goal。
print(postures.shape)
# 使用したいデータのみ選別
# 抜き取りたい行のインデックスリスト
desired_row_indices = [2, 3, 5, 6, 9, 10, 12, 14, 17, 21, 22, 23, 25, 26, 28, 29, 35, 38, 40, 43, 45, 56, 58, 59, 60, 62, 63, 64, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 93, 97, 98, 99, 100]  # Pythonのインデックスは0から始まるため、2行目はインデックス1、5行目はインデックス4
# 複数の行を抜き取る
postures = postures[desired_row_indices, :]
print(postures.shape)
# postures = postures[:5]  # さらに30


# PCAを実行
pca = PCA(n_components=2)
postures_pca = pca.fit_transform(postures)

# PC1とPC2を取得
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]

# achievedgoalの値に基づいて色分けしてプロット
achievedgoal_values = postures[:, -1]
plt.scatter(pc1, pc2, c=achievedgoal_values, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA(Color-coded by achieved_goal)')
plt.colorbar(label='achieved_goal')
plt.show()
