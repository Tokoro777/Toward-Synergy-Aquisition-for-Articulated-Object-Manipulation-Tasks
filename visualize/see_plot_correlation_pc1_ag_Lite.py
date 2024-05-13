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
from sklearn.linear_model import LinearRegression

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
dataset_path = args.dir + "/policy_without_WRJ1J0/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)   # (20,)で、はじめの19個は姿勢。最後の１つはachieve_goal。
print(postures.shape)

print(postures[0])


# achieved_goals = postures[:, -1]  # 最後の列がachieved_goal
#
# # achieved_goalを0.0-0.8の範囲で10個ずつ区切る
# bins = np.linspace(0.0, 0.8, 9)  # 0.0から0.8までを8つの区間に分割
# digitized = np.digitize(achieved_goals, bins)
#
# # 各区間から10個ずつ抽出して整理する
# selected_indices = []
# for i in range(1, 9):  # 区間1から8まで
#     indices_in_bin = np.where(digitized == i)[0][:10]  # 各区間から10個ずつ選択
#     selected_indices.extend(indices_in_bin)
#
# # 選択されたデータを取得
# postures = postures[selected_indices]
# selected_achieved_goals = achieved_goals[selected_indices]
#
# print("整理されたposturesの形状:", postures.shape)
# print("整理されたachieved_goalsの形状:", selected_achieved_goals.shape)


# new_posture = np.array([0, 1.44, 0, 0, 1.53, 0, 0, 1.44, 0, 0, 1.22, 0, 0, 0, 0])
# new_postures = np.tile(new_posture, (10, 1))  # 同じ姿勢を10個作成する
# # 新しい姿勢データを既存のデータに追加する
# postures = np.append(postures, new_postures, axis=0)

# agの値を取得
ag_values = postures[:, -1]

# agの範囲ごとにカウントするためのリストを作成
ag_ranges = [(i * 0.1, (i + 1) * 0.1) for i in range(8)]  # (0.0, 0.1), (0.1, 0.2), ..., (0.6, 0.7), (0.7, 0.8)
ag_counts = [0] * len(ag_ranges)

# agの範囲ごとにカウント
for ag in ag_values:
    for idx, (start, end) in enumerate(ag_ranges):
        if start <= ag < end:
            ag_counts[idx] += 1
            break

# 結果を出力
for idx, (start, end) in enumerate(ag_ranges):
    print(f"ag範囲 {start}-{end}: {ag_counts[idx]} postures")


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

# # 線形回帰
# slope, intercept, r_value, p_value, std_err = linregress(pc1, achievedgoal_values)
#
# # 回帰直線の描画
# plt.plot(pc1, intercept + slope * pc1, 'r', label='Regression line')
#
# # 相関係数を計算
# correlation = np.corrcoef(pc1, achievedgoal_values)[0, 1]
# print(f"PC1とachieved_goalの相関係数: {correlation}")
#
# # PC1とachievedgoalの値に基づいて色分けしてプロット
# plt.scatter(pc1, achievedgoal_values, cmap='viridis')
# plt.xlabel('PC1')
# plt.ylabel('achieved_goal')
# plt.title('PC1 vs achieved_goal')
# plt.colorbar(label='achieved_goal')
# plt.show()


# LinearRegressionモデルを作成
model = LinearRegression()

# PC1をreshapeしてfitメソッドに渡すための形状に整形
pc1_reshaped = pc1.reshape(-1, 1)

# モデルをPC1とachieved_goalのデータにフィットさせる
model.fit(pc1_reshaped, achievedgoal_values)

# 回帰係数と切片を取得
slope = model.coef_[0]
intercept = model.intercept_

# 回帰直線の式を表示
print(f"回帰直線の式: y = {slope:.2f} * PC1 + {intercept:.2f}")

# プロット
plt.scatter(pc1, achievedgoal_values, color='blue', label='Data Points')
plt.plot(pc1, model.predict(pc1_reshaped), color='red', linewidth=2, label='Regression Line')
plt.xlabel('PC1')
plt.ylabel('achieved_goal')
plt.title('Linear Regression: PC1 vs achieved_goal')
plt.legend()
plt.show()