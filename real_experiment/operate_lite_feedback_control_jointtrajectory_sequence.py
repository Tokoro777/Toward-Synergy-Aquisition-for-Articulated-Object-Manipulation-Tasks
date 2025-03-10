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

        # 目標角度シーケンスの定義
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

        # 目標角度シーケンスをプロット
        cumulative_time = 0.0
        for idx, target in enumerate(self.target_sequence):
            start_time = cumulative_time
            end_time = cumulative_time + target['duration']

            # 縦の破線（目標変更時点）
            plt.axvline(x=start_time, color='k', linestyle='--', alpha=0.5)
            # plt.text(start_time, 0.6, f"Start {idx+1}: {target['angle']} rad", rotation=90, verticalalignment='bottom')

            # 目標角度の期間限定横線
            plt.hlines(y=target['angle'], xmin=start_time, xmax=end_time, colors='r', linestyles='--', alpha=0.7,
                       label=f"Target Sequence [rad]" if idx == 0 else "")

            cumulative_time += target['duration']

        # 最後の目標の終了時点を示す縦の破線
        plt.axvline(x=cumulative_time, color='k', linestyle='--', alpha=0.5)

        plt.xlim(0, cumulative_time)
        plt.ylim(0, 0.7)
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [rad]")
        plt.title("Scissors Angle Over Time with Target Sequence")
        plt.legend()

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
        plot_filename = f"Plot_Sequence_{timestamp}.pdf"
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
        time = np.array(self.time_history)
        response = np.array(self.angle_history)
        target = desired_goal

        ignore_time = 0.3

        settling_time = calculate_settling_time(time, response, target, tolerance=0.02, hold_time=1.0)
        overshoot = calculate_overshoot(time, response, target, ignore_time=ignore_time)
        peak_time = calculate_peak_time(time, response, target, ignore_time=ignore_time)
        steady_state_error = calculate_steady_state_error(time, response, target, tolerance=0.02, hold_time=1.0)

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
- Desired Goal: {desired_goal} rad

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

    def plot_angle_history(self, desired_goal):
        """
        単一の目標角度に対する時間と角度のプロットを生成し、保存します。
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_history, self.angle_history, label="Scissors Angle [rad]", color='blue')
        plt.axhline(y=desired_goal, color='r', linestyle='--', label=f"Target Angle [{desired_goal} rad]")
        plt.xlim(0, max(self.time_history) + 1)
        plt.ylim(0, 0.6)
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [rad]")
        plt.title("Scissors Angle Over Time")
        plt.legend()
        plt.grid(True)

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
        now_jst = datetime.utcnow() + timedelta(hours=9)  # UTCに9時間加算してJSTを取得
        timestamp = now_jst.strftime("%Y%m%d%H%M%S")  # 'YYYYMMDDHHMMSS'形式
        plot_filename = f"Plot_{timestamp}.pdf"
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

    def run_posture(self, joint_goal):
        self.targets = joint_goal

        rospy.loginfo("Setting new targets.")
        self.log_target_joint_state()

        # PCAの準備
        self.pca = PCA(n_components=2)
        dataset_path = "/home/user/projects/shadow_robot/base/src/sr_interface/sr_grasp/scripts/new_grasp_dataset_with_ag.npy"
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

        params, params_covariance = curve_fit(ramp_function, pc1, achievedgoal_values, p0=initial_params)

        residuals = achievedgoal_values - ramp_function(pc1, *params)
        std_dev = np.std(residuals)

        mask = np.abs(residuals) < 2 * std_dev
        pc1_filtered = pc1[mask]
        achievedgoal_values_filtered = achievedgoal_values[mask]

        params_filtered, params_covariance_filtered = curve_fit(
            ramp_function, pc1_filtered, achievedgoal_values_filtered, p0=initial_params
        )

        x0, a = params_filtered
        rospy.loginfo(f"Ramp function parameters: a={a}, x0={x0}")

        initial_ag = 0.0
        self.pc1_score = (initial_ag / a) + x0
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
        while not self.stop_requested and self.current_target_index < len(self.target_sequence):
            if hasattr(self, 'current_angle') and self.current_angle is not None:
                # 現在の目標と時間を取得
                current_time = time.time()
                elapsed_time = current_time - self.target_start_time
                current_target = self.target_sequence[self.current_target_index]
                desired_goal = current_target["angle"]
                duration = current_target["duration"]

                # 目標の持続時間が経過したら次の目標に切り替え
                if elapsed_time >= duration:
                    self.current_target_index += 1
                    if self.current_target_index < len(self.target_sequence):
                        self.target_start_time = current_time
                        rospy.loginfo(
                            f"Switching to next target: {self.target_sequence[self.current_target_index]['angle']} rad")
                    else:
                        rospy.loginfo("All targets have been processed.")
                        break  # すべての目標を処理したらループを抜ける

                # 現在の目標に基づくPID計算
                current_target_angle = desired_goal
                error = current_target_angle - self.c