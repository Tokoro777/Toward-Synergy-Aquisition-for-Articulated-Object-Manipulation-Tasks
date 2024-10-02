#!/usr/bin/env python3
"""
Displays dataset of grasp object successfully in the RL
"""
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Tkinterバックエンドを使用
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/tokoro")
args = parser.parse_args()

# ベースディレクトリの設定
# base_dir = os.path.join(args.dir, "policy_without_WRJ1J0", "test")
base_dir = os.path.join(args.dir, "policy_sci_updown_no_zslider_only_third_bend", "test")

# 各ファイルのパスと対応するdesired_agの値
files_with_ag = [
    (os.path.join(base_dir, "error_with_desired_ag=1e-06_in_ramdom_hand_1degree_sigmoid.txt"), 1e-6), #error_with_desired_ag=0.0_in_ramdom_hand_2degree.txt
    (os.path.join(base_dir, "error_with_desired_ag=0.2_in_ramdom_hand_1degree_sigmoid.txt"), 0.2),
    (os.path.join(base_dir, "error_with_desired_ag=0.4_in_ramdom_hand_1degree_sigmoid.txt"), 0.4),
    (os.path.join(base_dir, "error_with_desired_ag=0.6_in_ramdom_hand_1degree_sigmoid.txt"), 0.6),
    (os.path.join(base_dir, "error_with_desired_ag=0.7_in_ramdom_hand_1degree_sigmoid.txt"), 0.7)
]

data = []

for file, desired_ag in files_with_ag:
    with open(file, 'r') as f:
        actual_values = [float(line.strip()) for line in f.readlines()]
        # errors = [abs(actual - desired_ag) for actual in actual_values]  # ラジアンの場合
        # errors = [math.degrees(actual - desired_ag) for actual in actual_values]  # 度の場合, ラジアンから度に変換
        errors = [abs(math.degrees(actual - desired_ag)) for actual in actual_values]  # 絶対値の場合
        data.append(errors)

# 箱ひげ図の作成
# plt.figure(figsize=(10, 6))
# plt.boxplot(data, labels=['0.0', '0.2', '0.4', '0.6', '0.7'])
# plt.xlabel('Desired achieved_goal values [rad]')
# plt.ylabel('Error ( | actual_achieved_goal  -  desired_achieved_goal | ) [rad]')
# plt.title('Boxplot of Errors for different Desired achieved_goal values')
# plt.grid(True)
# plt.savefig(os.path.join(base_dir, "boxplot_radian.png"))
# plt.show()


# 箱ひげ図の作成

# actual-desiredの差の場合
# plt.figure(figsize=(10, 6))
# plt.boxplot(data, labels=['0.0', '0.2', '0.4', '0.6', '0.7'])
# desired = "desired"
# plt.xlabel(f"$A_{{{desired}}}$ [rad]", fontsize=40)
# plt.ylabel('$e$ [°]', fontsize=40)
# # plt.title('Boxplot of Errors for different Desired achieved_goal values')
# plt.ylim(-30, 10)  # 縦軸の範囲を設定
# plt.grid(True)
# # ラベルの文字サイズを設定
# plt.xticks(fontsize=25)  # x軸のラベルの文字サイズを設定
# plt.yticks(fontsize=25)  # y軸のラベルの文字サイズを設定
# # Adjusting subplot parameters to trim excess whitespace
# plt.subplots_adjust(left=0.148, right=0.98, top=0.975, bottom=0.17)

# |actual-desired| 絶対値の場合
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['0.0', '0.2', '0.4', '0.6', '0.7'])
desired = "desired"
plt.xlabel(f"$A_{{{desired}}}$ [rad]", fontsize=40)
plt.ylabel('$e$ [°]', fontsize=40)
# plt.title('Boxplot of Errors for different Desired achieved_goal values')
plt.ylim(0, 12)  # 縦軸の範囲を設定
# 縦軸のメモリを5度刻みに設定
plt.yticks(range(0, 15, 5))
plt.grid(True)
# ラベルの文字サイズを設定
plt.xticks(fontsize=25)  # x軸のラベルの文字サイズを設定
plt.yticks(fontsize=25)  # y軸のラベルの文字サイズを設定
# Adjusting subplot parameters to trim excess whitespace
plt.subplots_adjust(left=0.148, right=0.98, top=0.975, bottom=0.17)

plt.savefig(os.path.join(base_dir, "boxplot_degrees_ramdom_hand_1degree_abs_0-12_sigmoid.png")) # boxplot_degrees_ramdom_hand_2degree ,boxplot_degrees_-30-10.png"
plt.show()