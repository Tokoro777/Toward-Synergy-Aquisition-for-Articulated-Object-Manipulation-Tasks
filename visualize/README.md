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
