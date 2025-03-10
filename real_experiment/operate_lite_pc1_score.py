#!/usr/bin/env python3
import rospy
import pickle
import sys
import select
import termios
import serial
import time
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from datetime import datetime, timedelta # 日本時間取得のため追加
import os # ディレクトリ操作のため追加


class OperateExecution():
    def __init__(self):
        rospy.init_node("operate_scissor_line_fit", anonymous=True)
        rospy.sleep(1.0)
        # ファイルからデータを取得
        operateconfig = sys.argv[1]
        with open('operate_configs/' + operateconfig + '.pkl', 'rb') as jc:
            self.targets = pickle.load(jc)
        print(self.targets)
        self.start_position = None
        self.interpolation_rate = 20
        self.force_threshold = 0.1
        self.angle_threshold = 28.65
        self.max_angle_change_per_second = 10.0
        self.stop_requested = False
        # PID用パラメータ
        self.Kp = 1.0
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
        # キーボード入力設定
        self.old_term = termios.tcgetattr(sys.stdin)
        new_term = termios.tcgetattr(sys.stdin)
        new_term[3] = new_term[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, new_term)
        # Arduinoから角度読み取りスレッド開始
        self.time_history = []
        self.angle_history = []
        self.start_angle_reading_thread()
        self.pc1_scores_history = []
        self.scissors_angles_history = []
        self.pc1_score = None
        # joint_trajectory_controller用アクションクライアント
        self.controller_action_name = "/rh_trajectory_controller/follow_joint_trajectory"
        rospy.loginfo(f"Waiting for {self.controller_action_name} action server...")
        self.client = actionlib.SimpleActionClient(self.controller_action_name, FollowJointTrajectoryAction)
        self.client.wait_for_server()
        rospy.loginfo("Action server found.")

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

    def plot_correlation(self):
        plt.figure(figsize=(10, 5))
        # 最初のデータ点(pc1_score=0.0, angle=0.0)を除外
        if len(self.pc1_scores_history) > 1:
            pc1_array = np.array(self.pc1_scores_history[1:])
            angle_array = np.array(self.scissors_angles_history[1:])
            plt.scatter(pc1_array, angle_array, label="Angle vs PC1 Score")
            # 直線フィット
            m, b = np.polyfit(pc1_array, angle_array, 1)
            # 直線プロット用
            x_line = np.linspace(min(pc1_array), max(pc1_array), 100)
            y_line = m * x_line + b
            plt.plot(x_line, y_line, 'r-', label=f"Fit: angle={m:.4f}*pc1_score+{b:.4f}")
            # # グラフ上にテキスト表示
            # plt.text(0.5*(min(pc1_array)+max(pc1_array)),
            # 0.5*(min(angle_array)+max(angle_array)),
            # f"angle={m:.4f}*pc1_score+{b:.4f}",
            # color='red')
        else:
            # データ点が1点以下ならそのままプロットなし
            plt.scatter(self.pc1_scores_history, self.scissors_angles_history, label="Angle vs PC1 Score")
            plt.xlabel("PC1 Score", fontsize=20)
            plt.ylabel("Scissors Angle [rad]", fontsize=20)
            plt.title("Scissors Angle vs PC1 Score", fontsize=16)
            plt.tick_params(axis='both', labelsize=16)
            plt.legend()
            plt.grid()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.93, bottom=0.13)
            # 保存先ディレクトリの設定
            save_dir = "/home/user/pc1_score"
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                    rospy.loginfo(f"Created directory: {save_dir}")
                except Exception as e:
                    rospy.logerr(f"Failed to create directory {save_dir}: {e}")
                    return # ディレクトリの作成に失敗した場合、関数を終了

        # タイムスタンプ付きのファイル名を生成
        now_jst = datetime.utcnow() + timedelta(hours=9) # UTCに9時間加算してJSTを取得
        timestamp = now_jst.strftime("%Y%m%d%H%M%S") # 'YYYYMMDDHHMMSS'形式
        plot_filename = f"Correlation_Plot_{timestamp}.pdf"
        plot_path = os.path.join(save_dir, plot_filename)

        # プロットをPDFとして保存
        try:
            plt.savefig(plot_path, format='pdf')
            rospy.loginfo(f"Plot saved to {plot_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save plot to {plot_path}: {e}")
            # プロットを表示
        try:
            plt.show()
        except Exception as e:
            rospy.logerr(f"Failed to display plot: {e}")
        finally:
            plt.close()

    def log_target_joint_state(self):
        rospy.loginfo("Target joint state:")
        for joint, value in self.targets.items():
            rospy.loginfo(f"{joint}: {value}")


    def run_posture(self, joint_goal):
        self.targets = joint_goal
        rospy.loginfo("Setting new targets.")
        self.log_target_joint_state()
        # PCA準備
        dataset_path = "/home/user/projects/shadow_robot/base/src/sr_interface/sr_grasp/scripts/new_grasp_dataset_with_ag.npy"
        data = np.load(dataset_path)
        postures = data[:, :-1]
        achievedgoal_values = data[:, -1]
        self.pca = PCA(n_components=2)
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
        # initial_ag / a + x0 の計算を残した上で、pc1_scoreを0.0に設定
        tmp_pc1_score = (initial_ag / a) + x0
        rospy.loginfo(f"(initial_ag / a) + x0 = {tmp_pc1_score}")
        self.pc1_score = -0.4 # pc1_scoreを0.0に設定
        step = 0.05
        pc1_vector = np.zeros((1, 2))
        # 関節の制御範囲
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

        actuation_center = np.array([(j_min + j_max) / 2 for j_min, j_max in joint_limits.values()])
        actuation_range = np.array([(j_max - j_min) / 2 for j_min, j_max in joint_limits.values()])
        joint_names = list(joint_limits.keys())
        # pc1_scoreを増やしつつ計測ループ
        while not self.stop_requested:
            pc1_vector[0, 0] = self.pc1_score
            inverse_posture = self.pca.inverse_transform(pc1_vector)
            rospy.loginfo(f"PC1 score: {self.pc1_score}")
            # inverse_postureのインデックス操作
            inverse_posture = np.delete(inverse_posture, 11)
            inverse_posture[12] = - inverse_posture[12]
            # スケール変換
            control_values = actuation_center + inverse_posture * actuation_range
            control_values = np.clip(control_values, [j_min for j_min, _ in joint_limits.values()],
            [j_max for _, j_max in joint_limits.values()])
            adjusted_targets = dict(zip(self.targets.keys(), control_values))
            rospy.loginfo(f"Moving!!!!!(Base on pc1_score: {self.pc1_score:.4f})")
            # joint_trajectory_controllerへ送信
            goal = FollowJointTrajectoryGoal()
            trajectory = JointTrajectory()
            trajectory.joint_names = joint_names
            point = JointTrajectoryPoint()
            point.positions = [adjusted_targets[jn] for jn in joint_names]
            point.time_from_start = rospy.Duration(1.0)
            trajectory.points.append(point)
            trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
            goal.trajectory = trajectory
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        if result is not None:
            rospy.loginfo("Trajectory execution finished.")
        else:
            rospy.logwarn("Trajectory execution not finished properly.")
        # データ記録
        # 最初のpc1_score=0.0のときのcurrent_angleもここで記録されるが、
        # plotで除外するのでこのままappendして問題なし
        if hasattr(self, 'current_angle'):
            self.pc1_scores_history.append(self.pc1_score)
            self.scissors_angles_history.append(self.current_angle)
            self.pc1_score += step
            if self.pc1_score > 0.4:
                rospy.loginfo("Reached maximum PC1 score. Stopping.")
                self.stop_requested = True
                rospy.loginfo("plot_correlation.")
                self.plot_correlation()
                self.check_for_stop_request()
                rospy.sleep(1.0)

    def check_for_stop_request(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                rospy.loginfo("Stop request received. Stopping movement.")
                self.stop_requested = True
                rospy.loginfo("plot_correlation.")
                self.plot_correlation()

    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_term)

if __name__ == "__main__":
    node = OperateExecution()
    operateconfig = sys.argv[1]
    with open('operate_configs/' + operateconfig + '.pkl', 'rb') as jc:
        joints_states = pickle.load(jc)
    print(joints_states)
    newjoint = joints_states
    print(newjoint)
    node.run_posture(newjoint)