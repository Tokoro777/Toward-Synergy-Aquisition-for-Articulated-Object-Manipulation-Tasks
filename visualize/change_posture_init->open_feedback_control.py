#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import math
import argparse
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import time

# Argument parsing for directory paths
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

# Load MuJoCo model
model = load_model_from_path(
    # args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
    args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown_no_rollhingeWRJ1J0THJ2.xml")
sim = MjSim(model)

# Define the timestep (in seconds) based on the MuJoCo model settings
timestep = sim.model.opt.timestep
print(f"timestep: {timestep}s")

# Load dataset
file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"
# dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy_roundscissor/{}/{}".format(folder_name, file_npy)
dataset_path = args.dir + "/policy_round3finger/{}/{}".format(folder_name, file_npy)
# dataset_path = args.dir + "/policy_oldscissor/{}/{}".format(folder_name, file_npy)
postures = np.load(dataset_path)
achievedgoal_values = postures[:, -1]

# PCA and posture transformations
pca = PCA(n_components=2)
postures = postures[:, :-1]
postures_pca = pca.fit_transform(postures)
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]
# 符号反転（必要に応じて行う）
pc1 *= -1  # PC2 の符号を反転

# Calculate correlation coefficient
correlation = np.corrcoef(pc1, achievedgoal_values)[0, 1]
print(f"PC1とachieved_goalの相関係数: {correlation}")

# ランプ関数の定義
def ramp_function(x, x0, a):
    return np.piecewise(x, [x < x0, x >= x0],
                        [0, lambda x: a * (x - x0)])

# 初期パラメータの設定
initial_x0 = np.percentile(pc1, 25)  # データの25パーセンタイルを初期値とする
initial_a = (achievedgoal_values.max() - achievedgoal_values.min()) / (pc1.max() - pc1.min())
initial_params = [initial_x0, initial_a]

# 最初のフィッティング
params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)

# 残差を計算
residuals = achievedgoal_values - ramp_function(pc1, *params)
std_dev = np.std(residuals)

# 外れ値を除外（残差が2標準偏差以内のデータのみ使用）
mask = np.abs(residuals) < 2 * std_dev
pc1_filtered = pc1[mask]
achievedgoal_values_filtered = achievedgoal_values[mask]

# 外れ値を除外したデータで再度フィッティング
params_filtered, params_covariance_filtered = curve_fit(
    ramp_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
)

x0, a = params_filtered
print(f"a: {a}")
print(f"x0: {x0}")

# desired_goal = 0.4のときの姿勢に移動
# desired_ag = 0.4
# pc1_value_for_0_4rad = (desired_ag / a) + x0
# print(f"0.4 radのときのpc1_value: {pc1_value_for_0_4rad}")

# 初期のpc1_scoreとして、0.0 radのときのpc1_valueを計算
initial_ag = 0.0
pc1_value_for_0_0rad = (initial_ag / a) + x0
pc1_score = pc1_value_for_0_0rad  # 初期のpc1_scoreを0.0 radのときのpc1_valueに設定
print(f"0.0 radのときのpc1_value: {pc1_value_for_0_0rad}")

# Helper functions for setting joint positions
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle

# joint_names = ["robot0:rollhinge", "robot0:WRJ1", "robot0:WRJ0", "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1",
#                "robot0:FFJ0",
#                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0", "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1",
#                "robot0:RFJ0",
#                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
#
# joint_angles = [1.57, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0, 0.0, 1.53, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0, 0.0, 1.22, 0.0, 0.0, 0.0]

joint_names = ["robot0:WRJ0",
                "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
                "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
                "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
                "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
# joint_angles = [0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.0, 1.57, 0.0, 0.0,
#                 0.115, 1.22, 0.0, 0.0, 0.0]
joint_angles = [0.0,
                0.0, 1.57, 0.0, 0.0,
                0.0, 1.57, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.115, 1.22, 0.0, 0.0, 0.0]

# joint_angles = [0.0,
#                 0.0, 1.44, 0.0, 0.0,
#                 0.0, 1.53, 0.0, 0.0,
#                 0.0, 1.44, 0.0, 0.0,
#                 0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])

# Control ranges and actuation setup
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

def _get_achieved_goal():
    return sim.data.get_joint_qpos("scissors_hinge_2:joint")

# PID gains and state
Kp, Ki, Kd = 0.015, 0.0, 0.0
previous_error, integral = 0, 0
# Define maximum and minimum values for the integral to prevent windup
integral_max = 0.026
integral_min = -0.026

# Simulation loop with feedback control
viewer = MjViewer(sim)
t = 0
loop_count = 0
scissors_angles = []

set_initial_joint_positions(sim, joint_names, joint_angles)
sim.data.set_joint_qpos("scissors:joint", initial_qpos)
sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)

# Prepare pc1_vector
pc1_vector = np.zeros((1, 2))

while True:
    viewer.render()
    sim.step()
    t += 1

    # Get current achieved goal
    current_ag = _get_achieved_goal()
    print(f"Step: {t}, current_ag: {current_ag}")

    # Set desired goal
    desired_goal = 0.4

    # Calculate PID control
    error = desired_goal - current_ag  # 誤差を (目標 - 現在) に変更

    # Update integral with clamping to prevent windup
    integral += error
    # if integral > integral_max:
    #     integral = integral_max
    # elif integral < integral_min:
    #     integral = integral_min

    derivative = error - previous_error
    pid_output = (Kp * error) + (Ki * integral) + (Kd * derivative)

    # Update pc1_score with PID output
    pc1_score += pid_output
    # pc1_vector with updated pc1_score
    pc1_vector[0, 0] = pc1_score
    # 必要であれば、再反転
    pc1_vector[0, 0] *= -1

    # Transform updated pc1_score back to joint angles
    inverse_posture = pca.inverse_transform(pc1_vector)

    print(f"pc1_score: {pc1_score}")

    # Apply control input based on updated joint angles
    sim.data.ctrl[:] = actuation_center[:] + inverse_posture * actuation_range[:]
    sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])
    # print(sim.data.ctrl[:])

    # Update previous error
    previous_error = error

    # Record scissors angle only when loop_count == 2
    if loop_count == 1:
        scissors_angles.append(sim.data.get_joint_qpos("scissors_hinge_2:joint"))

    # Reset simulation after 2000 steps
    if t > 800:
        if loop_count == 1:
            # Convert steps to time in seconds
            time_values = np.array(range(len(scissors_angles))) * timestep
            # Plot the graph with time in seconds on x-axis
            plt.figure()
            plt.plot(time_values, scissors_angles, label="Current Scissors Angle")
            plt.axhline(y=desired_goal, color='r', linestyle='--', label="Target Scissors Angle (0.4rad)")
            plt.xlabel('Time [s]', fontsize=20)
            plt.ylabel('Scissors Angle [rad]', fontsize=20)
            plt.ylim(0.0, 0.6)
            plt.title(f'Scissors Angle over Time (Kp={Kp}, Ki={Ki}, Kd={Kd})', fontsize=16)
            plt.tick_params(axis='both', labelsize=16)  # x軸とy軸の目盛り文字サイズ
            plt.legend(fontsize=15)
            plt.grid(True)
            # 図の下に余白を追加
            plt.subplots_adjust(bottom=0.13)  # 下の余白を増やす（デフォルトより広げる）
            plt.show()

        # Reset the simulation
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)
        previous_error, integral = 0, 0  # Reset PID states
        loop_count += 1
        t = 0  # Reset the timer
