#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
import time

model = load_model_from_path("/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)

dataset_path = "/home/tokoro/policy_2finger/test/{}"

viewer = MjViewer(sim)

t = 0
pos_num = 0
postures = np.load(dataset_path.format("grasp_dataset_on_best_policy.npy"))
print(postures.shape)

# agの値をposturesに追加するために, 新しい列を追加して (277, 15) の形状に拡張
postures = np.hstack((postures, np.zeros((277, 1))))
print(postures.shape)

ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

for name in sim.model.actuator_names:
    print(name)

# ハンドモデルの初期関節位置を設定する関数
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

# 関節名と初期角度の定義
# joint_names = [#"robot0:zslider",
#                "robot0:rollhinge",
#                "robot0:WRJ1", "robot0:WRJ0",
#                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
#                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
#                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
#                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
# joint_angles = [#0.04,
#                 1.57,  # はさみの穴を狭めたバージョン
#                 0.0, 0.0,
#                 0.0, 1.44, 0.0, 1.57,
#                 0.0, 1.53, 0.0, 1.57,
#                 0.0, 1.44, 0.0, 1.57,
#                 0.0, 1.22, 0.209, 0.0, -1.57]

# # robot_for_grasp_obj_Lite_scissors_updown_no_rollhingeWRJ1J0THJ2
# # rollhingeやWRJ1なし(WRJ0はあり0.0~0.001)で, THJ2は0.0~0.0001でしか動かないver
# joint_names = ["robot0:WRJ0",
#                 "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
#                 "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
#                 "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
#                 "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
# joint_angles = [0.0,
#                 0.0, 1.44, 0.0, 1.57,
#                 0.0, 1.53, 0.0, 1.57,
#                 0.0, 1.44, 0.0, 1.57,
#                 0.0, 1.22, 0.0, 0.0, -1.57]

# scissors_updown_only_third_bend
# 第三関節しか曲がらない, 第一関節は全て曲がらないver
joint_names = [#"robot0:zslider",
               "robot0:rollhinge",
               "robot0:WRJ1", "robot0:WRJ0",
               "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
               "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
               "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
               "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
joint_angles = [#0.04,
                1.57,  # はさみの穴を狭めたバージョン
                0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])  # はさみの初期位置

# 関節位置を設定, 初期化
set_initial_joint_positions(sim, joint_names, joint_angles)

def _get_achieved_goal():
    # はさみのhingeの角度の取得
    hinge_joint_angle_2 = sim.data.get_joint_qpos("scissors_hinge_2:joint")  # 正の値(はさみが開く場合の時)
    return hinge_joint_angle_2

while pos_num < len(postures):
    t = 0  # ステップカウンタを初期化
    max_ag_value = -np.inf  # 非常に低い初期値から最大角度を見つける

    while t < 500:
        viewer.render()
        sim.step()
        t += 1

        # 現在の角度を取得し、最大値を更新
        current_ag_value = _get_achieved_goal()
        if current_ag_value > max_ag_value:
            max_ag_value = current_ag_value

    # 500ステップ経過後に最大角度を保存
    postures[pos_num][-1] = max_ag_value  # pos_num行目の最後の要素に最大角度を設定
    print(f"Updated posture {pos_num}: max ag = {max_ag_value}")

    # pos_numを増加して次のデータに移行
    pos_num += 1
    if pos_num < len(postures):  # pos_numが範囲内か確認
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

    # 制御信号を設定
    if pos_num < len(postures):  # pos_numが範囲内か確認
        sim.data.ctrl[:] = actuation_center[:] + postures[pos_num][:-1] * actuation_range[:]
        sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

# 新しいデータセットの保存
new_dataset_path = dataset_path.format("new_grasp_dataset_with_ag.npy")
np.save(new_dataset_path, postures)
print(f"New dataset saved to {new_dataset_path}")