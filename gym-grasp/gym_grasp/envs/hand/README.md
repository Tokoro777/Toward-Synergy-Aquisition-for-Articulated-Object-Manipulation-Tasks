# grasp_object.py 注意ポイント

## ハンドの初期姿勢の指定
```
def _reset_sim(self):\
関数内に\
for joint_name, angle in zip(joint_names, joint_angles):  # 全てのjointを初期指定\
　　self.sim.data.set_joint_qpos(joint_name, angle)\
```
という, ハンドの姿勢を初期化する部分がある.
これは, joint_anglesの値に気をつける. 角ばった4本指はさみのバージョン, 丸い4本指はさみのバージョン, 3本指のバージョンの3種類あるので, コメントアウトして選択する.\
\\

同様に, \
def step(self, action):\
関数内にも\
for joint_name, angle in zip(joint_names, joint_angles):  # 全てのjointを初期指定
    self.sim.data.set_joint_qpos(joint_name, angle)  # 始めの50stepは手の初期位置を維持する\
という, 初期化部分がある. これも, joint_anglesをはさみに応じて選択する.
