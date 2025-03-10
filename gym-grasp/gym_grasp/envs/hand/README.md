# grasp_object.py 注意ポイント

## ハンドの初期姿勢の指定
角ばった4本指はさみのバージョン, 丸い4本指はさみのバージョン, 3本指のバージョンの3種類が学習では用いられる. はさみそれぞれで, 対応する初期姿勢が異なる. よって,学習の際は, どのはさみを使うかで,どの姿勢にするかコメントアウトによって選択する必要がある.

```
def _reset_sim(self):

    for joint_name, angle in zip(joint_names, joint_angles):
        self.sim.data.set_joint_qpos(joint_name, angle)
```
ハンドの姿勢を初期化する部分.
これは, joint_anglesの値に気をつける. 角ばった4本指はさみのバージョン, 丸い4本指はさみのバージョン, 3本指のバージョンの3種類あるので, コメントアウトして選択する.

\同様に

```
def step(self, action):

    for joint_name, angle in zip(joint_names, joint_angles):  # 全てのjointを初期指定
        self.sim.data.set_joint_qpos(joint_name, angle)  # 始めの50stepは手の初期位置を維持する
```
初期化部分. これも, joint_anglesをはさみに応じて選択する.
