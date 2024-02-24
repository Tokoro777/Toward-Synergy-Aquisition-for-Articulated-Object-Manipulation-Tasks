#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import os
import time

model = load_model_from_path("/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand.xml")
sim = MjSim(model)

dataset_path = "/home/tokoro/policy/test/{}"

viewer = MjViewer(sim)

t = 0
pos_num = 0
postures = np.load(dataset_path.format("grasp_dataset_10.npy"))
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
print(actuation_center)
print(postures, postures[0])  # actionの値

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
                "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [1.57,
                0.0, 0.0,
                0.0, 1.57, 0.0, 0.0,
                0.0, 1.57, 0.0, 0.0,
                0.0, 1.57, 0.0, 0.0,
                0.0, 0.0, 1.57, 0.0, 0.0,
                0.0, 1.22, 0.0, 0.0, 0.0]

# joint_angles = [1.57,
#                 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 1.57, 0.0,
#                 0.0, 0.0, 1.57, 1.57, 0.0,
#                 0.0, 1.22, 0.0, 0.0, 0.0]

# 関節位置を設定
set_initial_joint_positions(sim, joint_names, joint_angles)

while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if t > 500:  # あるグラスプポーズについて500step表示
        t = 0
        pos_num += 1
        set_initial_joint_positions(sim, joint_names, joint_angles)
        print(pos_num)

    if t == 499:
        print(sim.data.ctrl[:])  # jointの値
        # 各アクチュエータの角度を表示
        for name in sim.model.actuator_names:
            joint_idx = sim.model.actuator_name2id(name)
            joint_angle = sim.data.qpos[joint_idx]
            print(f"{name}: {joint_angle}")

    sim.data.ctrl[:-1] = actuation_center[:-1] + postures[pos_num] * actuation_range[:-1]
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])

    time.sleep(0.005)

    # if t > 100 and os.getenv('TESTING') is not None:
    #     break
