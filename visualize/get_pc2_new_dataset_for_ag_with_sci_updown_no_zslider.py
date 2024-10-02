#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

model = load_model_from_path("/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)

dataset_path = "/home/tokoro/policy_sci_updown_no_zslider_only_third_bend/test/{}"

viewer = MjViewer(sim)

t = 0
pos_num = 0
postures = np.load(dataset_path.format("grasp_dataset_on_best_policy.npy"))
print(postures.shape)

# agの値をposturesに追加するために, 新しい列を追加して (277, 17) の形状に拡張
postures = np.hstack((postures, np.zeros((277, 3))))
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
joint_names = ["robot0:rollhinge",
               "robot0:WRJ1", "robot0:WRJ0",
               "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
               "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
               "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
               "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
joint_angles = [1.57,
                0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.53, 0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])  # はさみの初期位置

# 関節位置を設定, 初期化
set_initial_joint_positions(sim, joint_names, joint_angles)

def get_scissors_orientation(sim):
    # ハサミのグローバル座標系での姿勢（四元数）を取得
    scissors_quat = sim.data.get_body_xquat("scissors_part0")
    # 四元数をオイラー角に変換
    r = R.from_quat(scissors_quat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # オイラー角を度数で取得
    return roll, pitch, yaw

while pos_num < len(postures):
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    if t > 500:
        euler_value = get_scissors_orientation(sim)
        postures[pos_num][-3:] = euler_value  # postures(277,17)の最後の3要素(0の値)をオイラー角に更新
        print(f"Updated posture {pos_num}: euler = {euler_value}")
        t = 0
        pos_num += 1
        if pos_num < len(postures):  # pos_numが範囲内か確認
            set_initial_joint_positions(sim, joint_names, joint_angles)
            sim.data.set_joint_qpos("scissors:joint", initial_qpos)
            sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
            sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

    if pos_num < len(postures):  # pos_numが範囲内か確認
        sim.data.ctrl[:] = actuation_center[:] + postures[pos_num][:-3] * actuation_range[:]
        sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    # time.sleep(0.005)

print(postures.shape)
# 新しいデータセットの保存
new_dataset_path = dataset_path.format("new_grasp_dataset_with_rollpitchyaw.npy")
np.save(new_dataset_path, postures)
print(f"New dataset saved to {new_dataset_path}")
