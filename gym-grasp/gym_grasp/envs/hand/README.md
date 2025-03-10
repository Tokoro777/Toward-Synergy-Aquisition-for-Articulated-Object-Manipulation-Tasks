# grasp_object.py
def _reset_sim(self):\
内に\
for joint_name, angle in zip(joint_names, joint_angles):  # 全てのjointを初期指定\
　　self.sim.data.set_joint_qpos(joint_name, angle)\
という, ハンドの姿勢を初期化する部分がある.
これは, joint_anglesの値に気をつける. 角ばった4本指はさみのバージョン, 丸い4本指はさみのバージョン, 3本指のバージョンの3種類ある.
