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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_vertical.xml")
sim = MjSim(model)

# motoda ---------------------------------------

#file_npy = "total_grasp_dataset.npy"
file_npy = "grasp_dataset_15.npy"
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
dataset_path = args.dir + "/policy_damping=0.1/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)
print(postures.shape)
# policy_best_free_joint/test/grasp_dataset_290.npyの時
# # 使用したいデータのみ選別
# # 抜き取りたい行のインデックスリスト
# desired_row_indices = [2, 3, 5, 6, 9, 10, 12, 14, 17, 21, 22, 23, 25, 26, 28, 29, 35, 38, 40, 43, 45, 56, 58, 59, 60, 62, 63, 64, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 93, 97, 98, 99, 100]  # Pythonのインデックスは0から始まるため、2行目はインデックス1、5行目はインデックス4
# # 複数の行を抜き取る
# postures = postures[desired_row_indices, :]

# policy_damping=0.1/test/grasp_dataset_15.npyの時
# desired_row_indices = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53]
desired_row_indices = [6, 9, 10, 14, 17, 19, 20, 22, 23, 24, 27, 28, 29, 32, 34, 35, 36, 40, 43, 44, 46, 49, 50, 51, 54, 55, 59, 61, 63, 64, 67, 68, 69, 71, 72, 75, 82, 93, 96, 105, 107, 113, 114, 119, 121, 133, 138, 147, 151]
postures = postures[desired_row_indices, :]
print(postures.shape)


ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

pca = PCA(n_components=5)
pca.fit(postures)

# PCAの各主成分の寄与率を出力
explained_variance_ratios = pca.explained_variance_ratio_
for i, explained_variance_ratio in enumerate(explained_variance_ratios):
    print(f"PC{i+1} explained variance ratio: {explained_variance_ratio:.4f}")

pc_axis = 1
n = 0
scores = pca.transform(postures)
score_range = [(min(a), max(a)) for a in scores.T]
# print(score_range)
trajectory = []
trajectory_len = 500

# pc_axis=1なので, PC1での500個の軌道を生成. trajectoryはリストで, １つ目がPC1に関する非ゼロの値の(500,)の配列, 2~5は中身が全て0の(500,)の配列
for i in range(5):
    if i == pc_axis - 1:
        trajectory.append(np.arange(score_range[pc_axis-1][0], score_range[pc_axis-1][1],
                                    (score_range[pc_axis-1][1] - score_range[pc_axis-1][0])/float(trajectory_len))[:500])
    else:
        trajectory.append(np.zeros(trajectory_len))
    # print(trajectory[-1].shape)
trajectory = np.array(trajectory).transpose()

# print(trajectory)  # .shapeは(500, 5) (0,0)~(499,0)にのみ非ゼロのPC1に沿った軌道が格納

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

joint_names = [#"robot0:rollhinge",
                "robot0:WRJ1", "robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

# joint_angles = [#1.57,
#                 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 1.57, 0.0,
#                 0.0, 0.0, 1.57, 1.57, 0.0,
#                 0.0, 1.22, 0.0, 0.0, 0.0]

joint_angles = [#1.57,  # すべての指先が曲がっていて、はさみをfreejointにしても落とさず掴んでくれそうな姿勢
                0.0, 0.0,
                0.0, 1.44, 0.0, 1.57,
                0.0, 1.53, 0.0, 1.57,
                0.0, 1.44, 0.0, 1.57,
                0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.209, -0.524, -0.361]


# 関節位置を設定
set_initial_joint_positions(sim, joint_names, joint_angles)

p = 0
while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if n == 0 and t < 230:  # 1個目の軌道は, 与える制御信号に対して関節が追従するのに時間がかかるので230step 1個目を維持
        n = 0
        p += 1
        # print("p", p)

    if n == 0 and t == 230:  # 制御信号に関節が追従したので, 2個目の軌道に移る
        n += 1
        t = 0
        # print("trajectory[0] finish! trajectory[1] start.")

    if t > 1 and n < 499 and n > 0:  # 500個ある軌道の点にそってハンドを制御, 5stepごとに次の点に移動
        t = 0
        n += 1

    # if t > 1 and n < 499:  # 500個ある軌道の点にそってハンドを制御, 5stepごとに次の点に移動
    #     t = 0
    #     n += 1

    # print("trajectory[", n, "]")

    posture = pca.mean_ + pca.inverse_transform(trajectory[n])  # trajectory[?]=[* 0 0 0 0]

    sim.data.ctrl[:-1] = actuation_center[:-1] + posture * actuation_range[:-1]
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])
    # print(n, sim.data.ctrl[:-1][3])　　#  制御信号と関節の値が同じか比較
    # print(sim.data.get_joint_qpos("robot0:FFJ2"))

    # print("robot0:FFJ0", sim.data.get_joint_qpos("robot0:FFJ0"), "robot0:MFJ0", sim.data.get_joint_qpos("robot0:MFJ0"),
    #       "robot0:RFJ0", sim.data.get_joint_qpos("robot0:RFJ0"))

    time.sleep(0.001)

    # 500ステップ以上経過したら初期化し, 繰り返し再生
    if n == 499 and t == 1:
        n = 0
        t = 0
        set_initial_joint_positions(sim, joint_names, joint_angles)
