import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander

# ノードの初期化
rospy.init_node("robot_commander_examples", anonymous=True)

# Shadow Hand Liteのコマンダーを初期化
hand_commander = SrHandCommander(name="hand_g")

# 初期姿勢を取得（ここでは初期姿勢をそのまま使用する）
initial_pose = hand_commander.get_current_pose()

# パターン1のジョイント設定（ジョイント名と目標値の辞書）
pattern1 = {
    "rh_THJ1": 1.57, "rh_THJ2": 0, "rh_THJ4": 1.22, "rh_THJ5": 0,
    "rh_FFJ1": 1.57, "rh_FFJ2": 0, "rh_FFJ3": 1.44, "rh_FFJ4": 0,
    "rh_MFJ1": 1.57, "rh_MFJ2": 0, "rh_MFJ3": 1.53, "rh_MFJ4": 0,
    "rh_RFJ1": 1.57, "rh_RFJ2": 0, "rh_RFJ3": 1.44, "rh_RFJ4": 0,
    #"rh_LFJ1": 0, "rh_LFJ2": 0, "rh_LFJ3": 0, "rh_LFJ4": 0, "rh_LFJ5": 0,
    #"rh_WRJ1": 0, "rh_WRJ2": 0
}

# パターン2のジョイント設定（ジョイント名と目標値の辞書）
pattern2 = {
    "rh_THJ1": 0, "rh_THJ2": 0, "rh_THJ4": 0, "rh_THJ5": 0,
    "rh_FFJ1": 0, "rh_FFJ2": 0, "rh_FFJ3": 0, "rh_FFJ4": 0,
    "rh_MFJ1": 0, "rh_MFJ2": 0, "rh_MFJ3": 0, "rh_MFJ4": 0,
    "rh_RFJ1": 0, "rh_RFJ2": 0, "rh_RFJ3": 0, "rh_RFJ4": 0,
    #"rh_LFJ1": 0, "rh_LFJ2": 0, "rh_LFJ3": 0, "rh_LFJ4": 0, "rh_LFJ5": 0,
    #"rh_WRJ1": 0, "rh_WRJ2": 0
}

# 初期姿勢をそのまま保持するために、現在の状態を設定
hand_commander.move_to_joint_value_target(initial_pose)

# 状態遷移を手動で制御するためのインタラクション
def transition_to_pattern1():
    hand_commander.move_to_joint_value_target(pattern1)
    rospy.loginfo("Moved to pattern 1")

def transition_to_pattern2():
    hand_commander.move_to_joint_value_target(pattern2)
    rospy.loginfo("Moved to pattern 2")

# 手動で状態遷移を呼び出す
input("Press Enter to move to Pattern 1...")
transition_to_pattern1()
input("Press Enter to move to Pattern 2...")
transition_to_pattern2()
