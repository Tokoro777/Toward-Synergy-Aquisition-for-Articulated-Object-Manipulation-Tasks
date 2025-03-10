#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import argparse
from sklearn.decomposition import PCA
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

file_npy = "grasp_dataset_on_best_policy.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)
postures = np.load(dataset_path)
print(postures.shape)

# PCA setup
pca = PCA(n_components=5)
pca.fit(postures)

# PCAの寄与率を表示
explained_variance_ratios = pca.explained_variance_ratio_
for i, explained_variance_ratio in enumerate(explained_variance_ratios):
    print(f"PC{i+1} explained variance ratio: {explained_variance_ratio:.4f}")

# PC1を固定して、PC2の軌道を生成
pc_axis = 2  # PC2の軌道を生成
fixed_pc1_value = 0.8  # PC1の固定値
trajectory_len = 500

scores = pca.transform(postures)
score_range = [(min(a), max(a)) for a in scores.T]

trajectory = []
for i in range(5):
    if i == pc_axis - 1:
        # PC2に沿って軌道を生成
        trajectory.append(np.arange(score_range[pc_axis-1][0], score_range[pc_axis-1][1],
                                    (score_range[pc_axis-1][1] - score_range[pc_axis-1][0])/float(trajectory_len))[:500])
    elif i == 0:  # PC1は固定値を設定
        trajectory.append(np.full(trajectory_len, fixed_pc1_value))
    else:
        trajectory.append(np.zeros(trajectory_len))

trajectory = np.array(trajectory).transpose()

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

joint_names = [
    "robot0:rollhinge", "robot0:WRJ1", "robot0:WRJ0",
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
]
joint_angles = [
    1.57, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0,
    0.0, 1.53, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0,
    0.0, 1.22, 0.0, 0.0, 0.0
]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])

# 関節位置を設定
set_initial_joint_positions(sim, joint_names, joint_angles)

ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

n = 0
t = 0

while True:
    viewer.render()
    t += 1
    sim.step()

    if t > 1 and n < 499:  # 軌道に沿ってハンドを動かす
        t = 0
        n += 1

    # 軌道に従って制御信号を生成し、ハンドに適用
    posture = pca.mean_ + pca.inverse_transform(trajectory[n])
    sim.data.ctrl[:] = actuation_center[:] + posture * actuation_range[:]
    sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    hinge_joint_angle_2 = sim.data.get_joint_qpos("scissors_hinge_2:joint")
    # print(hinge_joint_angle_2)  # PC1を変化させた時の, achieved_goalの値を出力

    time.sleep(0.001)

    # 500ステップ以上経過したら初期化し, 繰り返し再生
    if n == 499 and t == 1:
        n = 0
        t = 0
        set_initial_joint_positions(sim, joint_names, joint_angles)  # ハンドの位置を初期化
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)  # はさみを初期位置にし、freejointで落下させる
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)  # はさみの回転角度の初期化
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)