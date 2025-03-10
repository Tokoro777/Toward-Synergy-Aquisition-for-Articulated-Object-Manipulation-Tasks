#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time

# グローバル変数として現在のポスチャのインデックスを保持
current_index = 0

def plot_scatter(pc2_values, roll_values, pitch_values, yaw_values):
    global current_index

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # PC2 vs Roll
    scatter = axs[0].scatter(pc2_values, roll_values, color='blue', label='Data Points')
    highlighted, = axs[0].plot([], [], 'ro', label='Current Posture')
    axs[0].set_xlabel('PC2 Values')
    axs[0].set_ylabel('Roll (degrees)')
    axs[0].set_title('PC2 vs Roll')
    axs[0].legend()

    # PC2 vs Pitch
    scatter = axs[1].scatter(pc2_values, pitch_values, color='blue', label='Data Points')
    highlighted, = axs[1].plot([], [], 'ro', label='Current Posture')
    axs[1].set_xlabel('PC2 Values')
    axs[1].set_ylabel('Pitch (degrees)')
    axs[1].set_title('PC2 vs Pitch')
    axs[1].legend()

    # PC2 vs Yaw
    scatter = axs[2].scatter(pc2_values, yaw_values, color='blue', label='Data Points')
    highlighted, = axs[2].plot([], [], 'ro', label='Current Posture')
    axs[2].set_xlabel('PC2 Values')
    axs[2].set_ylabel('Yaw (degrees)')
    axs[2].set_title('PC2 vs Yaw')
    axs[2].legend()

    def update(frame):
        highlighted.set_data(pc2_values[current_index], roll_values[current_index])
        highlighted.set_data(pc2_values[current_index], pitch_values[current_index])
        highlighted.set_data(pc2_values[current_index], yaw_values[current_index])
        return highlighted,

    ani = animation.FuncAnimation(fig, update, frames=len(pc2_values), interval=100, blit=True)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

file_npy = "new_grasp_dataset_with_euler_angle.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)

# データセットの読み込みと前処理
postures = np.load(dataset_path)
print(postures.shape)

# 主成分分析 (PCA)
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

pca = PCA(n_components=5)
postures_data = postures[:, :-3]  # 最後の3列 (euler_angle) を除いて PCA を適用
pca.fit(postures_data)

# PCAスコアの取得
scores = pca.transform(postures_data)
pc2_values = scores[:, 1]  # PC2の値を取得
roll_values = postures[:, -3]  # Rollの値を取得
pitch_values = postures[:, -2]  # Pitchの値を取得
yaw_values = postures[:, -1]  # Yawの値を取得

# 相関係数の計算
correlation_roll = np.corrcoef(pc2_values, roll_values)[0, 1]
correlation_pitch = np.corrcoef(pc2_values, pitch_values)[0, 1]
correlation_yaw = np.corrcoef(pc2_values, yaw_values)[0, 1]
print(f"Correlation Coefficient between PC2 and Roll: {correlation_roll:.4f}")
print(f"Correlation Coefficient between PC2 and Pitch: {correlation_pitch:.4f}")
print(f"Correlation Coefficient between PC2 and Yaw: {correlation_yaw:.4f}")

# 別スレッドでプロットを表示
plot_thread = threading.Thread(target=plot_scatter, args=(pc2_values, roll_values, pitch_values, yaw_values))
plot_thread.start()

# シミュレーション内で関節の位置を設定する関数
def set_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

# 関節名と初期角度の定義
joint_names = [#"robot0:zslider",
                "robot0:rollhinge",
                "robot0:WRJ1", "robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                # "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
joint_angles = [# 0.04,  # はさみの穴を狭めたver
                1.57,
                0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.53, 0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                # 0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])  # はさみの初期位置

t = 0
pos_num = 0

while pos_num < len(postures):
    viewer.render()
    t += 1
    sim.step()

    pc2 = pc2_values[pos_num]
    roll = roll_values[pos_num]
    pitch = pitch_values[pos_num]
    yaw = yaw_values[pos_num]

    if t == 1:  # 表示するポスチャの番号とPC2値とeuler値を表示, postureを500step表示
        print(f"Posture {pos_num}: PC2 = {pc2:.4f}, Roll = {roll:.4f}, Pitch = {pitch:.4f}, Yaw = {yaw:.4f}")

    if t > 500:  # 初期化部分
        t = 0
        pos_num += 1
        current_index = pos_num  # 現在のインデックスを更新
        if pos_num < len(postures):  # pos_numが範囲内か確認
            set_joint_positions(sim, joint_names, joint_angles)  # 500ステップ過ぎたらハンドのポジションを初期化
            sim.data.set_joint_qpos("scissors:joint", initial_qpos)  # はさみを初期位置にし、freejointで落下させる
            sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)  # はさみの回転角度の初期化
            sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

    if pos_num < len(postures):  # pos_numが範囲内か確認
        sim.data.ctrl[:] = actuation_center[:] + postures[pos_num][:-3] * actuation_range[:]  # アクチュエータが14個
        sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    time.sleep(0.005)
