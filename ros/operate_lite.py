#!/usr/bin/env python3 

from copy import deepcopy
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
from math import pi
import pickle
import sys
import select  # 新しく追加
import termios  # 新しく追加

class OperateExecution():

    def __init__(self):

        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1.0)

        self.targets = None
        self.start_position = None
        self.interpolation_rate = 20
        self.force_threshold = 0.06  # 力センサ[N]の閾値を設定
        self.stop_requested = False

        # キーボード入力のための設定
        self.old_term = termios.tcgetattr(sys.stdin)
        new_term = termios.tcgetattr(sys.stdin)
        new_term[3] = new_term[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_term)

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

        # 力センサの値を監視しながら関節の動きを制御
        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)
        while not self.stop_requested:
            rospy.sleep(0.1)  # 少し待機してから力センサの値をチェック
            self.monitor_force_sensor("rh_RFJ4")
            self.check_for_stop_request()  # 停止要求を確認

    def monitor_force_sensor(self, joint_name):
        # 力センサの値を取得してログに出力し、閾値を超えた場合に動作を停止
        force_value = self.get_force_sensor_value(joint_name)
        rospy.loginfo(f"Force sensor value for {joint_name}: {force_value} N")  # センサ値をログ出力

        if force_value > self.force_threshold:
            rospy.loginfo(f"Force threshold exceeded for {joint_name}. Stopping hand movement.")
            self.stop_hand()

    def get_force_sensor_value(self, joint_name):
        # 力センサの値を取得（ここでは仮に関節の現在の値を返すようにしているが、実際にはセンサから取得する）
        current_state = self.hand_commander.get_current_state()
        return current_state.get(joint_name, 0.0)

    def stop_hand(self):
        # ハンドの動作を停止するために現在の状態に設定
        current_state = self.hand_commander.get_current_state()
        self.hand_commander.move_to_joint_value_target(current_state, wait=True)
        self.stop_requested = True

    def check_for_stop_request(self):
        # キーボード入力を監視し、特定のキー（例: 's'）が押された場合に停止
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                rospy.loginfo("Stop request received. Stopping hand movement.")
                self.stop_hand()

    def __del__(self):
        # キーボード設定を元に戻す
        termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_term)

if __name__ == "__main__":
    rospy.init_node("operate_scissor", anonymous=True)

    node = OperateExecution()

    # ファイルからデータを取得する場合
    operateconfig = sys.argv[1]

    with open('operate_configs/' + operateconfig + '.pkl', 'rb') as jc:
        joints_states = pickle.load(jc)

    print(joints_states)

    newjoint = joints_states

    print(newjoint)

    node.run_posture(newjoint)

    # rospy.spin()  # ROSノードが停止しないようにする
