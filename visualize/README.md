# see_pca_axis_Lite_14_elements_scissors_updown.py
### どの主成分の軌道を見たいかの選択
これは第一主成分
```
    pc_axis = 1
```
### ハンドの初期姿勢の指定
はさみに応じて, joint_anglesをコメントアウトで選択
```
    set_initial_joint_positions(sim, joint_names, joint_angles)
```
### 3本指のはさみのみ注意すること
3本指のはさみの時, PC1ではなく, 符号反転を行ったPC2がはさみ操作の主成分. 3本指のときは, pc1_axis=2にしてかつ, 以下をコメントアウトを外す.
```
    # # 符号反転を行う (pc_axis が2の場合、PC2を反転)
    # trajectory[:, pc_axis - 1] *= -1  # PC2に相当する軸を反転
```


# get_new_dataset_for_ag_with_sci_updown_no_zslider.py
### ハンドの初期姿勢の指定
はさみに応じて, joint_anglesをコメントアウトで選択
```
    set_initial_joint_positions(sim, joint_names, joint_angles)
```
### policyのファイルに, 角度含めたデータセットを作成
```
    new_dataset_path = dataset_path.format("new_grasp_dataset_with_ag.npy")
    np.save(new_dataset_path, postures)
```


# see_plot_correlation_pc1_ag_Lite.py
3本指のはさみのみ,PC2がはさみ操作に必要な主成分であった.また, そのベクトルの符号を反転させた場合にのみ, はさみ操作の動きであることを確認したため, 3本指のときは以下のようにコメントアウトを外す.
```
    # # 符号反転（必要に応じて行う）
    # pc1 *= -1  # PC2 の符号を反転
```
