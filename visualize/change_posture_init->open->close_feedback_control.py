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

# Argument parsing for directory paths
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

# Load MuJoCo model
model = load_model_from_path(
    args.dir + "/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/grasp_object_remove_lf_scissors_updown.xml")
sim = MjSim(model)

# Load dataset
file_npy = "new_grasp_dataset_with_ag.npy"
folder_name = "test"
dataset_path = args.dir + "/policy_sci_updown_no_zslider_only_third_bend/{}/{}".format(folder_name, file_npy)
postures = np.load(dataset_path)
achievedgoal_values = postures[:, -1]

# PCA and posture transformations
pca = PCA(n_components=2)
postures = postures[:, :-1]
postures_pca = pca.fit_transform(postures)
pc1 = postures_pca[:, 0]
pc2 = postures_pca[:, 1]

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

# desired_goal = 0.4のときの姿勢に移動
desired_ag = 0.4
x0, a = params_filtered
pc1_value = (desired_ag / a) + x0
print(pc1_value)

pc1_vector = np.zeros((1, 2))
pc1_vector[0, 0] = pc1_value
inverse_posture_1 = pca.inverse_transform(pc1_vector)

# desired_goal = 0.0のときの姿勢に移動
desired_ag = 0.0
pc1_value = (desired_ag / a) + x0
print(pc1_value)
pc1_value = -1.5  # 完全にはさみを閉じきれる時のPC1値（ここは変える）
print(pc1_value)
pc1_vector[0, 0] = pc1_value
inverse_posture_2 = pca.inverse_transform(pc1_vector)


# Helper functions for setting joint positions
def set_initial_joint_positions(sim, joint_names, joint_angles):
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        joint_idx = sim.model.joint_name2id(joint_name)
        sim.data.qpos[joint_idx] = joint_angle


joint_names = ["robot0:rollhinge", "robot0:WRJ1", "robot0:WRJ0", "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1",
               "robot0:FFJ0",
               "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0", "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1",
               "robot0:RFJ0",
               "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]

joint_angles = [1.57, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0, 0.0, 1.53, 0.0, 0.0, 0.0, 1.44, 0.0, 0.0, 0.0, 1.22, 0.0, 0.0, 0.0]

initial_qpos = np.array([1.07, 0.892, 0.4, 1, 0, 0, 0])

# Control ranges and actuation setup
ctrlrange = sim.model.actuator_ctrlrange
actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.


def _get_achieved_goal():
    return sim.data.get_joint_qpos("scissors_hinge_2:joint")


# PID gains and state
Kp, Ki, Kd = 1, 1.2, 0.85  # Gain調整: Kpを少し小さく、Kdを増やして微分制御を強化
previous_error, integral = 0, 0
# Define maximum and minimum values for the integral to prevent windup
integral_max = 0.026
integral_min = -0.026

# Simulation loop with feedback control
viewer = MjViewer(sim)

# シミュレーションループ
t = 0
desired_goal = 0.4  # 初期目標角度は0.4rad

# Record scissors angles and step count when loop_count == 2
scissors_angles = []
steps = []
loop_count = 0

while True:
    viewer.render()
    t += 1
    sim.step()

    # 現在の角度を取得
    current_ag = _get_achieved_goal()
    print(f"Step: {t}, current_ag: {current_ag}, desired_goal: {desired_goal}")

    # Calculate PID control
    error = -(desired_goal - current_ag)

    # Update integral with clamping to prevent windup
    integral += error
    if integral > integral_max:
        integral = integral_max
    elif integral < integral_min:
        integral = integral_min

    derivative = error - previous_error
    pid_output = (Kp * error) + (Ki * integral) + (Kd * derivative)

    # Apply inverse posture and adjust control input with PID output
    if desired_goal == 0.4:
        sim.data.ctrl[:] = actuation_center[:] + (inverse_posture_1[0] + pid_output) * actuation_range[:]
    elif desired_goal == 0.0:
        sim.data.ctrl[:] = actuation_center[:] + (inverse_posture_2[0] + pid_output) * actuation_range[:]

    sim.data.ctrl[:] = np.clip(sim.data.ctrl[:], ctrlrange[:, 0], ctrlrange[:, 1])

    # Update previous error
    previous_error = error

    # Record scissors angle and step when loop_count == 2
    if loop_count == 2:
        scissors_angles.append(sim.data.get_joint_qpos("scissors_hinge_2:joint"))
        steps.append(t)

    # 500ステップを過ぎたら目標を切り替える
    if t == 500:
        print("500 steps completed at 0.4rad. Switching to 0.0rad.")
        desired_goal = 0.0
        previous_error, integral = 0, 0  # PIDの状態をリセット

    # 1000ステップを過ぎたらリセット
    if t == 1000:
        # Plot the graph only when loop_count == 2
        if loop_count == 2:
            plt.figure()
            plt.plot(steps, scissors_angles, label="Current Scissors Angle")
            plt.axhline(y=0.4, color='r', linestyle='--', label="Target Scissors Angle (0.4rad)")
            plt.xlabel('Steps')
            plt.ylabel('Scissors Angle (rad)')
            plt.title('Scissors Angle over Time')
            plt.legend()
            plt.grid(True)
            plt.show()
            break  # Exit the loop after displaying the plot

        print("1000 steps completed. Resetting simulation.")
        set_initial_joint_positions(sim, joint_names, joint_angles)
        sim.data.set_joint_qpos("scissors:joint", initial_qpos)
        sim.data.set_joint_qpos("scissors_hinge_1:joint", 0)
        sim.data.set_joint_qpos("scissors_hinge_2:joint", 0)
        desired_goal = 0.4  # 初期目標角度を0.4radにリセット
        t = 0  # タイマーリセット
        loop_count += 1  # Increase the loop count after each full cycle