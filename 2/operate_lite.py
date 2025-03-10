#!/usr/bin/env python3 

from copy import deepcopy
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
from math import pi
import pickle
import sys


class OperateExecution():

    def __init__(self):

        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1.0)

        self.targets = None
        self.start_position = None
        self.interpolation_rate = 20

    def run_posture(self, joint_goal):
        self.targets = joint_goal
        self.start_position = self.hand_commander.get_current_state()

        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)

    def stop_finger(self, finger):
        current_state = self.hand_commander.get_current_state()
        prefix = "rh_" + finger.upper()
        newjoint = {}
        for name, j in current_state.items():
            if name.startswith(prefix):
                newjoint[name] = j

        for name, j in self.targets.items():
            if name.startswith(prefix):
                self.targets[name] = newjoint[name]

        rospy.loginfo(f"Stopping finger {finger}")
        self.hand_commander.move_to_joint_value_target(self.targets, wait=False)


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