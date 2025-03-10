#!/usr/bin/env python3 

from copy import deepcopy
from builtins import input
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
from math import pi
import pickle 
from sr_grasp.utils import mk_grasp
from actionlib import SimpleActionClient, GoalStatus
from sr_robot_msgs.msg import GraspAction, GraspGoal
from moveit_msgs.msg import Grasp
from trajectory_msgs.msg import JointTrajectoryPoint
from sr_hand.shadowhand_ros import ShadowHand_ROS
import numpy as np
import rospkg 
import struct 
import sys

def force_inc(jointgoal):
    
    prefix = "rh_"
    fingers = ["FF", "MF", "RF"]
    
    thumb_upperlimits = {1:1.56, 2:0.5, 4:1.15, 5:1.035}
    thumb_lowerlimits = {1:-0.257, 2:-0.515, 4:0, 5:-1.04}

    for finger in fingers:
        for i in range(2,4):  # 1.1倍とか1.3倍が本当に適切なのかどうかは検討すること！！！！！
            jointgoal[prefix + finger + 'J' + str(i)] = 1.1 * jointgoal[prefix + finger + 'J' + str(i)]
            if jointgoal[prefix + finger + 'J' + str(i)] > 1.571:
                jointgoal[prefix + finger + 'J' + str(i)] = 1.56
    
    for i in [1,2,4,5]:
        jointgoal[prefix + "THJ" + str(i)] = 1.3 * jointgoal[prefix + "THJ" + str(i)]
        if jointgoal[prefix + "THJ" + str(i)] > thumb_upperlimits[i]:
            jointgoal[prefix + "THJ" + str(i)] = thumb_upperlimits[i]
        if jointgoal[prefix + "THJ" + str(i)] < thumb_lowerlimits[i]:
            jointgoal[prefix + "THJ" + str(i)] = thumb_lowerlimits[i]


    return jointgoal


class Finger:

    def __init__(self, name, initial_tactile, tactile_threshold, active=True):
        self.name = name
        self.active = active
        self.tactile_threshold = tactile_threshold
        self.initial_tactile = initial_tactile
        
    def get_name(self):
        return self.name
    
    def get_tacticle_threshold(self):
        return self.tactile_threshold
    
    def get_initial_tactile(self):
        return self.initial_tactile 

class GraspExecution():

    def __init__(self):

        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1.0)

        self.tactile = None 
        self.start_position = None 
        self.joints = None 
        self.interpolation_rate = 20
        

    def run_posture(self, joint_goal):
        self.targets = joint_goal
        self.start_position = self.hand_commander.get_current_state()

        initial_tactile_data = self.get_tactile()

        self.fingers = [Finger('FF', initial_tactile_data['FF'], 0.01), 
                        Finger('MF', initial_tactile_data['MF'], 0.06), 
                        Finger('RF', initial_tactile_data['RF'], 0.07), 
                        Finger('TH', initial_tactile_data['TH'], 0.08)]
        self.active_fingers = ['FF','MF','RF','TH']

        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)

        while len(self.active_fingers) > 0 and not rospy.is_shutdown():
            for finger_iterator , value in self.get_tactile().items():
                for finger in self.fingers:
                    if finger.get_name() == finger_iterator:
                        tactile_percentage = (value - finger.get_initial_tactile())/finger.get_initial_tactile()
                        if tactile_percentage >= finger.get_tacticle_threshold():
                            self.stop_finger(finger_iterator)

    def get_tactile(self):
        
        tactile_data = {}
        handtac = self.hand_commander.get_tactile_state()
        
        tactile_data['FF'] = handtac.pressure[0]         
        tactile_data['MF'] = handtac.pressure[1] 
        tactile_data['RF'] = handtac.pressure[2] 
        tactile_data['TH'] = handtac.pressure[4]

        return tactile_data 

    # 指の腹にある触覚センサーから, 停止すべき指を判定し, それ以外は動かし続ける.
    # はさみ操作は指の裏側なので, このコードは特に役に立たない. でも, 悪さもしないので, 残す
    def stop_finger(self, finger):
        current_state = self.hand_commander.get_current_state()
        for i in self.fingers:
            if i.name == finger and i.active == True:
                i.active = False
                prefix = "rh_" + finger.upper()
                newjoint = {}
                for name, j in current_state.items():
                    if name.startswith(prefix):
                        newjoint[name] = j
                
                for name, j in self.targets.items():
                    if name.startswith(prefix):
                        self.targets[name] = newjoint[name]

                self.active_fingers.remove(finger)
                
                rospy.loginfo(f"Stopping finger {finger}")
                self.hand_commander.move_to_joint_value_target(self.targets, wait=False)

if __name__ == "__main__":

    rospy.init_node("functional_grasping", anonymous=True)

    node = GraspExecution()

    # file_path = rospkg.RosPack().get_path('sr_grasp') + '/objwrtrobot.npy'
    graspconfig = sys.argv[1]

    with open('grasp_configs/'+ graspconfig + '.pkl','rb') as jc:
        joints_states = pickle.load(jc)

    print(joints_states)

    # 物体をつかむ操作の場合には force_inc を使用
    # newjoint = force_inc(joints_states)

    # はさみを開く動作の場合、force_inc は不要
    newjoint = joints_states

    print(newjoint)

    node.run_posture(newjoint)
