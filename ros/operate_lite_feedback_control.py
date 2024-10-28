#!/usr/bin/env python3

import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
import pickle
import sys
import select
import termios
import serial  # Arduinoと通信するためにpySerialを使用
import time


class OperateExecution():

    def __init__(self):
        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1.0)

        self.targets = None
        self.start_position = None
        self.interpolation_rate = 20
        self.force_threshold = 0.06  # 力センサ[N]の閾値
        self.angle_threshold = 28.65  # 角度の閾値（0.5rad）
        self.max_angle_change_per_second = 10.0  # 許容される角度の変化量/秒（例: 10度/秒）
        self.stop_requested = False

        # 予防停止のための急激な変化予測
        self.angle_history = []  # 直近の角度の履歴
        self.max_angle_history_length = 5  # 過去nステップの角度を記録
        self.angle_slope_threshold = 5.0  # 急激な角度変化の兆候とみなす角度変化率の閾値

        # PID制御用のパラメータ
        self.Kp = 1.0
        self.Ki = 1.2
        self.Kd = 0.85
        self.previous_error = 0.0
        self.integral = 0.0
        self.integral_max = 0.026
        self.integral_min = -0.026

        self.previous_angle = 0.0  # 前回の角度
        self.previous_time = time.time()  # 前回の時間

        # Arduinoとの通信設定
        self.arduino_port = '/dev/ttyUSB0'
        self.baudrate = 9600

        # キーボード入力のための設定
        self.old_term = termios.tcgetattr(sys.stdin)
        new_term = termios.tcgetattr(sys.stdin)
        new_term[3] = new_term[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_term)

    def get_angle_from_arduino(self, port='/dev/ttyUSB0', baudrate=9600):
        try:
            # Arduinoと接続
            ser = serial.Serial(port, baudrate, timeout=1)
            ser.flush()

            if ser.in_waiting > 0:
                # Arduinoからデータを読み取る
                angle_data = ser.readline().decode('utf-8').rstrip()
                angle = float(angle_data)
                return angle
        except Exception as e:
            rospy.logerr(f"Error reading from Arduino: {e}")
            return None

    def log_joint_state(self):
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

        rospy.loginfo("Setting new targets.")
        self.log_target_joint_state()

        # 力センサの値を監視しながら関節の動きを制御
        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)

        # PID制御の目標角度を設定
        desired_goal = 0.4  # 目標角度を0.4radに設定

        # PID制御のループ
        while not self.stop_requested:
            rospy.sleep(0.1)
            current_ag = self.get_current_scissors_angle()  # Arduinoからの現在角度を取得
            rospy.loginfo(f"Current scissors angle: {current_ag}, Desired: {desired_goal}")

            # 急激な角度変化の予測
            self.angle_history.append(current_ag)
            if len(self.angle_history) > self.max_angle_history_length:
                self.angle_history.pop(0)

            if len(self.angle_history) >= 2:
                slope = self.angle_history[-1] - self.angle_history[-2]
                rospy.loginfo(f"Angle slope: {slope}")

                if abs(slope) > self.angle_slope_threshold:
                    rospy.logwarn(
                        f"Predicted sudden angle change: slope {slope} exceeds threshold {self.angle_slope_threshold}. Stopping hand movement.")
                    self.stop_hand()
                    break

            # 角度が閾値を超えた場合に動作を停止
            if current_ag > self.angle_threshold:
                rospy.logwarn(
                    f"Angle threshold exceeded: {current_ag} > {self.angle_threshold}. Stopping hand movement.")
                self.stop_hand()
                break

            # PID制御の計算
            error = -(desired_goal - current_ag)
            self.integral += error
            if self.integral > self.integral_max:
                self.integral = self.integral_max
            elif self.integral < self.integral_min:
                self.integral = self.integral_min

            derivative = error - self.previous_error
            pid_output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

            # PID制御の出力を目標角度に加えて制御信号として設定
            adjusted_targets = {joint: value + pid_output for joint, value in self.targets.items()}
            self.hand_commander.move_to_joint_value_target(adjusted_targets, wait=False)

            self.previous_error = error

            # 力センサの値を監視して閾値を超えた場合に停止
            self.monitor_force_sensor("rh_RFJ4")
            self.check_for_stop_request()

    def monitor_force_sensor(self, joint_name):
        force_value = self.get_force_sensor_value(joint_name)
        rospy.loginfo(f"Force sensor value for {joint_name}: {force_value} N")

        if force_value > self.force_threshold:
            rospy.loginfo(f"Force threshold exceeded for {joint_name}. Stopping hand movement.")
            self.stop_hand()

    def get_force_sensor_value(self, joint_name):
        current_state = self.hand_commander.get_current_state()
        return current_state.get(joint_name, 0.0)

    def stop_hand(self):
        current_state = self.hand_commander.get_current_state()
        self.hand_commander.move_to_joint_value_target(current_state, wait=True)
        self.stop_requested = True

    def check_for_stop_request(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                rospy.loginfo("Stop request received. Stopping hand movement.")
                self.stop_hand()

    def get_current_scissors_angle(self):
        # Arduinoから角度を取得
        current_angle = self.get_angle_from_arduino(self.arduino_port, self.baudrate)
        if current_angle is not None:
            rospy.loginfo(f"Current scissors angle from Arduino: {current_angle}")
            return current_angle
        else:
            return 0.0  # エラー時はデフォルト値を返す

    def __del__(self):
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
