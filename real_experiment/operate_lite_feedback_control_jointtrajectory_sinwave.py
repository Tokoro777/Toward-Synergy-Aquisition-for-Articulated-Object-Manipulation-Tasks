#!/usr/bin/env python3

import rospy
import pickle
import sys
import select
import termios
import serial  # Arduinoと通信するためにpySerialを使用
import time
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from std_msgs.msg import Float64

import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from datetime import datetime, timedelta  # 日本時間取得のため追加
import os  # ディレクトリ操作のため追加


# 評価指標の関数（既存のまま）
def calculate_settling_time(time, response, target, tolerance=0.02, hold_time=2.0):
    lower_bound = target * (1 - tolerance)
    upper_bound = target * (1 + tolerance)

    within_bounds = np.logical_and(response >= lower_bound, response <= upper_bound)

    for i in range(len(within_bounds)):
        if within_bounds[i]:
            end_time = time[i] + hold_time
            end_idx = np.searchsorted(time, end_time)
            end_idx = min(end_idx, len(within_bounds))
            if np.all(within_bounds[i:end_idx]):
                return time[i]
    return None  # Settling Timeが見つからなかった場合


def calculate_overshoot(time, response, target, ignore_time=0.3):
    # 初期の `ignore_time` 秒を無視
    mask = time > ignore_time
    if not np.any(mask):
        return 0.0  # 無視後にデータがない場合、オーバーシュートは0とする

    response_after_ignore = response[mask]

    peak = np.max(response_after_ignore)
    overshoot = (peak - target) / target
    return overshoot if overshoot > 0 else 0.0


def calculate_peak_time(time, response, target, ignore_time=0.3):
    # 初期の `ignore_time` 秒を無視
    mask = time > ignore_time
    if not np.any(mask):
        return None  # 無視後にデータがない場合、ピークタイムは計算不能

    response_after_ignore = response[mask]

    overshoot = calculate_overshoot(time, response, target, ignore_time)
    if overshoot == 0.0:
        return None  # オーバーシュートがない場合、ピークタイムもなし

    peak_value = np.max(response_after_ignore)
    peak_index = np.argmax(response_after_ignore)
    peak_time = time[mask][peak_index]
    return peak_time


def calculate_steady_state_error(time, response, target, tolerance=0.02, hold_time=2.0):
    lower_bound = target * (1 - tolerance)
    upper_bound = target * (1 + tolerance)

    within_bounds = np.logical_and(response >= lower_bound, response <= upper_bound)

    for i in range(len(within_bounds)):
        if within_bounds[i]:
            end_time = time[i] + hold_time
            end_idx = np.searchsorted(time, end_time)
            end_idx = min(end_idx, len(within_bounds))
            if np.all(within_bounds[i:end_idx]):
                # 定常状態期間のデータを抽出
                steady_state_response = response[i:end_idx]
                steady_state_errors = np.abs(target - steady_state_response)
                steady_state_avg = np.mean(steady_state_errors)
                return steady_state_avg
    return None  # 定常状態に達していない場合


class OperateExecution():

    def __init__(self):
        rospy.sleep(1.0)

        # SrHandCommander削除
        # hand_commander関連メソッド削除（get_current_stateやmove_to_joint_value_target）

        self.targets = None
        self.start_position = None
        self.interpolation_rate = 20
        self.force_threshold = 0.1
        self.angle_threshold = 28.65
        self.max_angle_change_per_second = 10.0
        self.stop_requested = False

        # PID制御用のパラメータ
        self.Kp = 0.3
        self.Ki = 0.0
        self.Kd = 0.0
        self.previous_error = 0.0
        self.integral = 0.0
        self.integral_max = 0.026
        self.integral_min = -0.026

        self.previous_angle = 0.0
        self.previous_time = time.time()

        # Arduinoとの通信設定
        self.arduino_port = '/dev/ttyACM0'
        self.baudrate = 115200
        self.ser = self.initialize_serial(self.arduino_port, self.baudrate)

        # キーボード入力のための設定
        self.old_term = termios.tcgetattr(sys.stdin)
        new_term = termios.tcgetattr(sys.stdin)
        new_term[3] = new_term[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_term)

        # Arduinoデータ読み取りスレッドを開始
        self.start_angle_reading_thread()

        # 時間と角度データの記録用リスト
        self.time_history = []
        self.angle_history = []

        # 保存先ディレクトリを設定
        self.save_dir = "/home/user/target_sequence"  # 変更後: "/home/user/plots" -> "/home/user/target_sequence"
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                rospy.loginfo(f"Created directory: {self.save_dir}")
            except Exception as e:
                rospy.logerr(f"Failed to create directory {self.save_dir}: {e}")
                sys.exit(1)

        # joint_trajectory_controller用アクションクライアント初期化
        self.controller_action_name = "/rh_trajectory_controller/follow_joint_trajectory"
        rospy.loginfo(f"Waiting for {self.controller_action_name} action server...")
        self.client = actionlib.SimpleActionClient(self.controller_action_name, FollowJointTrajectoryAction)
        self.client.wait_for_server()
        rospy.loginfo("Action server found.")

        # 目標角度シーケンスの定義（現状では使用しないが、参考まで）
        self.target_sequence = [
            {"angle": 0.4, "duration": 6.0},
            {"angle": 0.2, "duration": 6.0},
            {"angle": 0.5, "duration": 6.0},
            {"angle": 0.1, "duration": 6.0},
        ]
        self.current_target_index = 0
        self.target_start_time = time.time()

        # 目標角度シーケンスに対するプロット用のマーカー
        self.sequence_markers = []

    def initialize_serial(self, port, baudrate):
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(1)
            ser.flush()
            return ser
        except Exception as e:
            rospy.logerr(f"Error initializing Arduino connection: {e}")
            return None

    def start_angle_reading_thread(self):
        self.angle_thread = threading.Thread(target=self.update_scissors_angle)
        self.angle_thread.daemon = True
        self.angle_thread.start()

    def update_scissors_angle(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            current_angle = self.get_current_scissors_angle(self.ser)
            if current_angle is not None:
                self.current_angle = current_angle
                print(f"Current Angle: {self.current_angle}", flush=True)
                current_time = time.time() - start_time
                self.time_history.append(current_time)
                self.angle_history.append(self.current_angle)
            if self.ser:
                self.ser.flushInput()
            rospy.sleep(0.1)

    def get_current_scissors_angle(self, ser):
        try:
            if ser and ser.in_waiting > 0:
                angle_data = ser.readline().decode('utf-8').strip()
                try:
                    angle_deg = float(angle_data)
                    angle = math.radians(angle_deg)
                    return angle
                except ValueError:
                    return None
        except Exception as e:
            rospy.logerr(f"Error reading from Arduino: {e}")
            return None

    def plot_angle_history_sequence(self, reports):
        """
        連続的な目標角度に対する時間と角度のプロットを生成し、保存します。
        PIDの評価指標は計測しません。
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_history, self.angle_history, label="Scissors Angle [rad]", color='blue')

        # 正弦波の目標角度をプロット
        period = 24.0  # 周期24秒（4つの6秒周期）
        amplitude = 0.25  # 振幅0.25 rad
        phase_shift = - math.pi / 2  # 位相シフト - pi/2
        time_array = np.array(self.time_history)
        desired_goals = amplitude * np.sin(2 * np.pi * time_array / period + phase_shift) + 0.25

        plt.plot(time_array, desired_goals, label="Target [rad]", color='red', linestyle='--')

        plt.xlim(0, max(self.time_history))
        plt.ylim(0.0, 0.6)
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [rad]")
        plt.title("Scissors Angle Over Time with Sinusoidal Target")
        plt.legend()
        plt.grid(True)

        # サブディレクトリ名を生成（例：Kp0.3_Ki0.0_Kd0.0）
        pid_dir_name = f"Kp{self.Kp}_Ki{self.Ki}_Kd{self.Kd}"
        sub_dir_path = os.path.join(self.save_dir, pid_dir_name)

        # サブディレクトリが存在しない場合は作成（念のため）
        if not os.path.exists(sub_dir_path):
            try:
                os.makedirs(sub_dir_path)
                rospy.loginfo(f"Created sub-directory: {sub_dir_path}")
            except Exception as e:
                rospy.logerr(f"Failed to create sub-directory {sub_dir_path}: {e}")
                return  # サブディレクトリの作成に失敗した場合、関数を終了

        # ファイル名にタイムスタンプを追加
        now_jst = datetime.utcnow() + timedelta(hours=9)  # UTCに9時間加算してJSTを取得
        timestamp = now_jst.strftime("%Y%m%d%H%M%S")  # 'YYYYMMDDHHMMSS'形式
        plot_filename = f"Plot_sinwave_{timestamp}.pdf"
        plot_path = os.path.join(sub_dir_path, plot_filename)

        # PDFファイルとして保存
        try:
            plt.savefig(plot_path, format='pdf')
            rospy.loginfo(f"Plot saved to {plot_path}")
            plt.show()
        except Exception as e:
            rospy.logerr(f"Failed to save plot to {plot_path}: {e}")
        finally:
            plt.close()

    def log_target_joint_state(self):
        rospy.loginfo("Target joint state:")
        for joint, value in self.targets.items():
            rospy.loginfo(f"{joint}: {value}")

    def evaluate_pid_performance(self, desired_goal):
        """
        単一の目標角度に対するPID制御の性能評価を行います。
        連続的な目標角度シーケンスに対しては評価しません。
        """
        time_array = np.array(self.time_history)
        response = np.array(self.angle_history)
        target = desired_goal

        ignore_time = 0.3

        settling_time = calculate_settling_time(time_array, response, target, tolerance=0.02, hold_time=1.0)
        overshoot = calculate_overshoot(time_array, response, target, ignore_time=ignore_time)
        peak_time = calculate_peak_time(time_array, response, target, ignore_time=ignore_time)
        steady_state_error = calculate_steady_state_error(time_array, response, target, tolerance=0.02, hold_time=1.0)

        rospy.loginfo("PID Performance Evaluation:")

        if settling_time is not None:
            rospy.loginfo(f"Settling Time (±2%): {settling_time:.2f} s")
        else:
            rospy.loginfo("Settling Time (±2%): Not achieved")

        rospy.loginfo(f"Overshoot: {overshoot * 100:.2f}%")

        if peak_time is not None:
            rospy.loginfo(f"Peak Time: {peak_time:.2f} s")
        else:
            rospy.loginfo("Peak Time: Not achieved")

        if steady_state_error is not None:
            rospy.loginfo(f"Steady-State Error: {steady_state_error:.4f} rad")
        else:
            rospy.loginfo("Steady-State Error: Not calculated (System did not reach steady-state)")

        # Settling Timeが見つかった場合のみレポートとプロットを生成
        if settling_time is not None:
            rospy.loginfo("Generating performance report and plot.")
            self.generate_performance_report(desired_goal, settling_time, overshoot, peak_time, steady_state_error)
            self.plot_angle_history(desired_goal)
        else:
            rospy.logwarn("Settling Time not achieved. Skipping report and plot generation.")

    def evaluate_pid_performance_sequence(self):
        """
        連続的な目標角度シーケンスに対するPID制御の性能評価をスキップし、時間と角度のプロットのみを生成します。
        """
        rospy.loginfo("Generating angle history plot for target sequence.")
        self.plot_angle_history_sequence(None)  # reportsは不要

    def generate_performance_report_sequence(self, reports):
        """
        連続的な目標角度シーケンスに対する性能評価レポートの生成をスキップします。
        """
        pass  # 連続的な目標角度に対する評価は行わない

    def generate_performance_report(self, desired_goal, settling_time, overshoot, peak_time, steady_state_error):
        # UTCに9時間を加算してJSTを取得
        now_jst = datetime.utcnow() + timedelta(hours=9)
        timestamp_str = now_jst.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_filename = now_jst.strftime("%Y%m%d%H%M%S")  # 'YYYYMMDDHHMMSS'形式

        report = f"""
PID Performance Evaluation Report
===============================
Timestamp: {timestamp_str} JST

PID Parameters:
- Kp: {self.Kp}
- Ki: {self.Ki}
- Kd: {self.Kd}
- Desired Goal: {desired_goal:.4f} rad

Evaluation Metrics:
"""

        if settling_time is not None:
            report += f"- Settling Time (±2%): {settling_time:.2f} s\n"
        else:
            report += "- Settling Time (±2%): Not achieved\n"

        report += f"- Overshoot: {overshoot * 100:.2f}%\n"

        if peak_time is not None:
            report += f"- Peak Time: {peak_time:.2f} s\n"
        else:
            report += "- Peak Time: Not achieved\n"

        if steady_state_error is not None:
            report += f"- Steady-State Error: {steady_state_error:.4f} rad\n"
        else:
            report += "- Steady-State Error: Not calculated (System did not reach steady-state)\n"

        # サブディレクトリ名を生成（例：Kp0.3_Ki0.0_Kd0.0）
        pid_dir_name = f"Kp{self.Kp}_Ki{self.Ki}_Kd{self.Kd}"
        sub_dir_path = os.path.join(self.save_dir, pid_dir_name)

        # サブディレクトリが存在しない場合は作成
        if not os.path.exists(sub_dir_path):
            try:
                os.makedirs(sub_dir_path)
                rospy.loginfo(f"Created sub-directory: {sub_dir_path}")
            except Exception as e:
                rospy.logerr(f"Failed to create sub-directory {sub_dir_path}: {e}")
                return  # サブディレクトリの作成に失敗した場合、関数を終了

        # ファイル名にタイムスタンプを追加
        report_filename = f"PID_Report_{timestamp_filename}.txt"
        report_path = os.path.join(sub_dir_path, report_filename)

        try:
            with open(report_path, 'w') as file:
                file.write(report)
            rospy.loginfo(f"Performance report saved to {report_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save performance report to {report_path}: {e}")

    def run_posture(self, joint_goal):
        self.targets = joint_goal

        rospy.loginfo("Setting new targets.")
        self.log_target_joint_state()

        # PCAの準備
        self.pca = PCA(n_components=2)
        dataset_path = "/home/user/projects/shadow_robot/base/src/sr_interface/sr_grasp/scripts/new_grasp_dataset_with_ag.npy"
        if not os.path.exists(dataset_path):
            rospy.logerr(f"Dataset file {dataset_path} does not exist.")
            return
        data = np.load(dataset_path)
        postures = data[:, :-1]
        achievedgoal_values = data[:, -1]
        postures_pca = self.pca.fit_transform(postures)
        pc1 = postures_pca[:, 0]

        def ramp_function(x, x0, a):
            return np.piecewise(x, [x < x0, x >= x0],
                                [0, lambda x: a * (x - x0)])

        initial_x0 = np.percentile(pc1, 25)
        initial_a = (achievedgoal_values.max() - achievedgoal_values.min()) / (pc1.max() - pc1.min())
        initial_params = [initial_x0, initial_a]

        try:
            params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)
        except RuntimeError as e:
            rospy.logerr(f"Curve fitting failed: {e}")
            return

        residuals = achievedgoal_values - ramp_function(pc1, *params)
        std_dev = np.std(residuals)

        mask = np.abs(residuals) < 2 * std_dev
        pc1_filtered = pc1[mask]
        achievedgoal_values_filtered = achievedgoal_values[mask]

        try:
            params_filtered, params_covariance_filtered = curve_fit(
                ramp_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
            )
        except RuntimeError as e:
            rospy.logerr(f"Curve fitting on filtered data failed: {e}")
            return

        x0, a = params_filtered
        rospy.loginfo(f"Ramp function parameters: a={a}, x0={x0}")

        initial_ag = 0.0
        self.pc1_score = (initial_ag / a) + x0
        self.pc1_score = -0.4
        rospy.loginfo(f"Initial pc1_score for 0.0rad: {self.pc1_score}")

        joint_limits = {
            'rh_FFJ4': (-0.349, 0.349),
            'rh_FFJ3': (0.0, 1.571),
            'rh_FFJ2': (0.0, 1.571),

            'rh_MFJ4': (-0.349, 0.349),
            'rh_MFJ3': (0.0, 1.571),
            'rh_MFJ2': (0.0, 1.571),

            'rh_RFJ4': (-0.349, 0.349),
            'rh_RFJ3': (0.0, 1.571),
            'rh_RFJ2': (0.0, 1.571),

            'rh_THJ5': (-1.047, 1.047),
            'rh_THJ4': (0.0, 1.222),
            'rh_THJ2': (-0.524, 0.524),
            'rh_THJ1': (0.0, 1.571),
        }

        pc1_vector = np.zeros((1, 2))

        actuation_center = np.array([(joint_min + joint_max) / 2 for joint_min, joint_max in joint_limits.values()])
        actuation_range = np.array([(joint_max - joint_min) / 2 for joint_min, joint_max in joint_limits.values()])
        joint_names = list(joint_limits.keys())

        # PID制御ループの開始
        while not self.stop_requested:
            if hasattr(self, 'current_angle') and self.current_angle is not None:
                # 現在の目標と時間を取得
                current_time = time.time()
                elapsed_time = current_time - self.target_start_time

                # 正弦波による目標角度の計算
                period = 24.0  # 周期24秒（4つの6秒周期）
                amplitude = 0.25  # 振幅0.25 rad
                phase_shift = - math.pi / 2  # 位相シフト - pi/2
                desired_goal = amplitude * np.sin(2 * math.pi * elapsed_time / period + phase_shift) + 0.25

                # PID制御の計算
                error = desired_goal - self.current_angle
                self.integral += error * 0.1  # タイムステップ0.1秒
                # Integral windup対策
                self.integral = max(self.integral_min, min(self.integral, self.integral_max))
                derivative = (error - self.previous_error) / 0.1  # タイムステップ0.1秒
                pid_output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
                self.pc1_score += pid_output
                self.pc1_score = np.clip(self.pc1_score, -0.6, 0.5)

                pc1_vector[0, 0] = self.pc1_score
                inverse_posture = self.pca.inverse_transform(pc1_vector)

                rospy.loginfo(f"PC1 score: {self.pc1_score}")

                # データの適切な調整
                try:
                    inverse_posture = np.delete(inverse_posture, 11)
                    inverse_posture[12] = - inverse_posture[12]
                except IndexError as e:
                    rospy.logerr(f"Index error during posture adjustment: {e}")
                    continue  # 次のループへ

                control_values = actuation_center + inverse_posture * actuation_range
                control_values = np.clip(control_values, [jmin for jmin, _ in joint_limits.values()],
                                         [jmax for _, jmax in joint_limits.values()])

                adjusted_targets = dict(zip(joint_names, control_values))
                rospy.loginfo(f"Moving!!!!!(Based on angle: {self.current_angle:.4f} rad)")

                # joint_trajectory_controllerへ送信
                goal = FollowJointTrajectoryGoal()
                trajectory = JointTrajectory()
                trajectory.joint_names = joint_names

                point = JointTrajectoryPoint()
                point.positions = [adjusted_targets[jn] for jn in joint_names]
                point.time_from_start = rospy.Duration(0.2)  # 0.2秒で到達想定

                trajectory.points.append(point)
                trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.05)

                goal.trajectory = trajectory

                self.client.send_goal(goal)

                self.previous_error = error

                # キーボード入力のチェック
                self.check_for_stop_request()
                rospy.sleep(0.1)

    def check_for_stop_request(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                rospy.loginfo("Stop request received. Stopping hand movement.")
                self.stop_requested = True
                rospy.loginfo("Evaluating PID performance.")
                self.evaluate_pid_performance(desired_goal=0.1)
                rospy.loginfo("Plotting angle history.")
                self.plot_angle_history(desired_goal=0.1)

    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_term)

if __name__ == "__main__":
    rospy.init_node("operate_scissor", anonymous=True)

    node = OperateExecution()

    operateconfig = sys.argv[1]
    with open('operate_configs/' + operateconfig + '.pkl', 'rb') as jc:
        joints_states = pickle.load(jc)

    print(joints_states)
    newjoint = joints_states
    print(newjoint)

    node.run_posture(newjoint)