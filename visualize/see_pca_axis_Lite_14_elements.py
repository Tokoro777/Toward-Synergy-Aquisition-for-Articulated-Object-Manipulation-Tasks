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
dataset_path = args.dir + "/policy/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy/{}/{}".format("210215", "grasp_dataset_30.npy")

viewer = MjViewer(sim)

t = 0
postures = np.load(dataset_path)
print(postures.shape)



ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

pca = PCA(n_components=5)
postures = postures[:, 1:-2]  # 17個から14個に要素を減らす(☓WRJ0, zslider, ag)
print(postures.shape)
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
                # "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

#
# joint_angles = [#1.57,  # すべての指先が曲がっていて、はさみをfreejointにしても落とさず掴んでくれそうな姿勢
#                 0.0, 0.0,
#                 0.0, 1.44, 0.0, 1.57,
#                 0.0, 1.53, 0.0, 1.57,
#                 0.0, 1.44, 0.0, 1.57,
#                 # 0.0, 0.0, 1.32, 0.0, 1.57,
#                 0.0, 1.22, 0.209, -0.524, -0.361]

joint_angles = [#1.57,  # はさみの穴を狭めたver
                0.0, 0.0,
                0.0, 1.44, 0.0, 1.57,
                0.0, 1.53, 0.0, 1.57,
                0.0, 1.44, 0.0, 1.57,
                # 0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.209, 0.0, -1.57]


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

    sim.data.ctrl[2:-1] = actuation_center[2:-1] + posture * actuation_range[2:-1]
    sim.data.ctrl[2:-1] = np.clip(sim.data.ctrl[2:-1], ctrlrange[2:-1, 0], ctrlrange[2:-1, 1])
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
