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

model = load_model_from_path(args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/hand_Lite.xml")
sim = MjSim(model)

file_npy = "grasp_dataset_on_best_policy.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_without_WRJ1J0/{}/{}".format(folder_name, file_npy)
viewer = MjViewer(sim)

# Load and preprocess the dataset
postures = np.load(dataset_path)
print(postures.shape)

# Principal Component Analysis (PCA)
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

pca = PCA(n_components=5)
postures_data = postures[:, :-1]  # Remove the last column (ag) for PCA
pca.fit(postures_data)

# Get the PCA scores
scores = pca.transform(postures_data)
pc1_values = scores[:, 0]  # Get PC1 values
ag_values = postures[:, -1]  # Get ag values

# Function to set joint positions in the simulation
def set_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

# Define joint names and initial angles
joint_names = ["robot0:rollhinge","robot0:WRJ1", "robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [1.57,  # 指先曲げないver
                0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                0.0, 1.53, 0.0, 0.0,
                0.0, 1.44, 0.0, 0.0,
                # 0.0, 0.0, 1.32, 0.0, 1.57,
                0.0, 1.22, 0.0, 0.0, 0.0]

t = 0
pos_num = 0

while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    pc1 = pc1_values[pos_num]
    ag = ag_values[pos_num]

    if t == 1:  # 表示するpostureの番号とPC1値とag値を表示
        print(f"Posture {pos_num}: PC1 = {pc1:.4f}, ag = {ag:.4f}")

    if t > 500:  # あるグラスプポーズについて500step表示
        t = 0
        pos_num += 1
        set_joint_positions(sim, joint_names, joint_angles)  # 500step過ぎたらposition初期化


    # sim.data.ctrl[2:-1] = actuation_center[2:-1] + postures[pos_num][1:-2] * actuation_range[2:-1]  # WRJ0とzsliderとagを消すパターン
    # sim.data.ctrl[2:-1] = np.clip(sim.data.ctrl[2:-1], ctrlrange[2:-1, 0], ctrlrange[2:-1, 1])

    sim.data.ctrl[:-1] = actuation_center[:-1] + postures[pos_num][:-1] * actuation_range[:-1]  # actuatorが14個で, datasetからagを消すパターン
    sim.data.ctrl[:-1] = np.clip(sim.data.ctrl[:-1], ctrlrange[:-1, 0], ctrlrange[:-1, 1])


    time.sleep(0.005)