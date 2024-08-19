#!/usr/bin/env python3 

from copy import deepcopy
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
from math import pi
import pickle
import sys
import rospy
from threading import Timer

class OperateExecution():

    def __init__(self):

        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1.0)

        self.targets = None
        self.start_position = None
        self.interpolation_rate = 20

        # 定期的に関節の状態をログに出力するためのタイマー
        # self.timer = rospy.Timer(rospy.Duration(1.0), self.log_joint_state_callback)

    def log_joint_state(self):
        # 現在の関節の状態を取得してログに出力
        current_state = self.hand_commander.get_current_state()
        rospy.loginfo("Current joint state:")
        for joint, value in current_state.items():
            rospy.loginfo(f"{joint}: {value}")

    def log_target_joint_state(self):
        rospy.loginfo("Target joint state:")
        for joint, value in self.targets.items():
            rospy.loginfo(f"{joint}: {value}")

    def run_posture(self, joint_goal):
        self.targets = joint_goal
        self.start_position = self.hand_commander.get_current_state()

        # デバッグ用に現在の状態と目標の状態をログに出力
        rospy.loginfo("Setting new targets.")
        self.log_target_joint_state()

        # ターゲットの関節名と値をログ出力
        for name, value in self.targets.items():
            rospy.loginfo(f"Target joint {name}: {value}")

        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)
        # self.hand_commander.move_to_joint_value_target(self.targets, wait=True)  # 同期的に動作を待つ

        # # ターゲットに近づいたかどうかを確認するために関節の状態をログに出力
        # rospy.loginfo("Monitoring joint states...")
        # # 現在の状態を確認
        # current_state = self.hand_commander.get_current_state()
        # self.log_joint_state()

    # def stop_finger(self, finger):
    #     current_state = self.hand_commander.get_current_state()
    #     prefix = "rh_" + finger.upper()
    #     newjoint = {}
    #     for name, j in current_state.items():
    #         if name.startswith(prefix):
    #             newjoint[name] = j
    #
    #     for name, j in self.targets.items():
    #         if name.startswith(prefix):
    #             self.targets[name] = newjoint[name]
    #
    #     rospy.loginfo(f"Stopping finger {finger}")
    #     self.hand_commander.move_to_joint_value_target(self.targets, wait=False)

if __name__ == "__main__":
    rospy.init_node("operate_scissor", anonymous=True)

    node = OperateExecution()

    # ファイルからデータを取得する場合
    operateconfig = sys.argv[1]

    with open('operate_configs/' + operateconfig + '.pkl', 'rb') as jc:
        joints_states = pickle.load(jc)

    print(joints_states)

    newjoint = joints_states

    # 直接データを定義
    # newjoint = {
    #     'rh_THJ1': 1.57,
    #     'rh_THJ2': 0,
    #     'rh_THJ4': 1.22,
    #     'rh_THJ5': 0,
    #
    #     'rh_FFJ1': 1.57,
    #     'rh_FFJ2': 0,
    #     'rh_FFJ3': 1.44,
    #     'rh_FFJ4': 0,
    #
    #     'rh_MFJ1': 1.57,
    #     'rh_MFJ2': 0,
    #     'rh_MFJ3': 1.53,
    #     'rh_MFJ4': 0,
    #
    #     'rh_RFJ1': 1.57,
    #     'rh_RFJ2': 0,
    #     'rh_RFJ3': 1.44,
    #     'rh_RFJ4': 0
    # }

    print(newjoint)

    node.run_posture(newjoint)

    # rospy.spin()  # ROSノードが停止しないようにする