#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import os
import time

model = load_model_from_path("/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_Lite.xml")
sim = MjSim(model)

dataset_path = "/home/tokoro/policy/test/{}"

viewer = MjViewer(sim)

t = 0
pos_num = 0
postures = np.load(dataset_path.format("grasp_dataset_on_best_policy.npy"))
print(postures.shape)

# policy_best_free_joint/test/grasp_dataset_290.npyの時
# # 使用したいデータのみ選別
# # 抜き取りたい行のインデックスリスト
# desired_row_indices = [2, 3, 5, 6, 9, 10, 12, 14, 17, 21, 22, 23, 25, 26, 28, 29, 35, 38, 40, 43, 45, 56, 58, 59, 60, 62, 63, 64, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 93, 97, 98, 99, 100]  # Pythonのインデックスは0から始まるため、2行目はインデックス1、5行目はインデックス4
# # 複数の行を抜き取る
# postures = postures[desired_row_indices, :]

# policy_damping=0.1/test/grasp_dataset_15.npyの時
# desired_row_indices = [6, 9, 10, 14, 17, 19, 20, 22, 23, 24, 27, 28, 29, 32, 34, 35, 36, 40, 43, 44, 46, 49, 50, 51, 54, 55, 59, 61, 63, 64, 67, 68, 69, 71, 72, 75, 82, 93, 96, 105, 107, 113, 114, 119, 121, 133, 138, 147, 151]
# postures = postures[desired_row_indices, :]
# print(postures.shape)


ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
# print(actuation_center)
# print(postures, postures[0])  # actionの値

for name in sim.model.actuator_names:
    print(name)

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

joint_names = ["robot0:rollhinge",
                "robot0:WRJ1", "robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                # "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [1.57,  # すべての指先が曲がっていて、はさみをfreejointにしても落とさず掴んでくれそうな姿勢
                0.0, 0.0,
                0.0, 1.44, 0.0, 1.57,
                0.0, 1.53, 0.0, 1.57,
                0.0, 1.44, 0.0, 1.57,
                # 0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.209, -0.524, -0.361]


# 関節位置を設定, 初期化
set_initial_joint_positions(sim, joint_names, joint_angles)

while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()


    if t > 500:  # あるグラスプポーズについて500step表示
        t = 0
        pos_num += 1
        set_initial_joint_positions(sim, joint_names, joint_angles)  # 500step過ぎたらposition初期化
        print(pos_num)  # 今から表示するposの番号
        print(postures[pos_num, -1])  # achieved_goalの値を出力

    if t == 499:
        # print(sim.data.ctrl[:])  # jointの値
        # 各アクチュエータの角度を表示
        for name in sim.model.actuator_names:
            joint_idx = sim.model.actuator_name2id(name)
            joint_angle = sim.data.qpos[joint_idx]
            # print(f"{name}: {joint_angle}")
        # print(sim.data.ctrl[:-1])

    sim.data.ctrl[:-1] = actuation_center[:-1] + postures[pos_num][:-1] * actuation_range[:-1]
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])

    time.sleep(0.005)

    # if t > 100 and os.getenv('TESTING') is not None:
    #     break
