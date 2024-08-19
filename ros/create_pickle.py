import pickle
import os

# initial_config
sample_data = {
    'rh_FFJ4': 0,
    'rh_FFJ3': 1.44,
    'rh_FFJ2': 0,
    #'rh_FFJ1': 1.57,

    'rh_MFJ4': 0,
    'rh_MFJ3': 1.53,
    'rh_MFJ2': 0,
    #'rh_MFJ1': 1.57,

    'rh_RFJ4': 0,
    'rh_RFJ3': 1.44,
    'rh_RFJ2': 0,
    #'rh_RFJ1': 1.57,

    'rh_THJ5': 0,
    'rh_THJ4': 1.22,
    'rh_THJ2': 0,
    'rh_THJ1': 1.57
}

# sample_data = {
#     'rh_THJ1': 0,
#     'rh_THJ2': 0,
#     'rh_THJ4': 1.22,
#     'rh_THJ5': 0,
#
#     'rh_FFJ1': 0,
#     'rh_FFJ2': 0,
#     'rh_FFJ3': 1.44,
#     'rh_FFJ4': 0,
#
#     'rh_MFJ1': 0,
#     'rh_MFJ2': 0,
#     'rh_MFJ3': 1.53,
#     'rh_MFJ4': 0,
#
#     'rh_RFJ1': 0,
#     'rh_RFJ2': 0,
#     'rh_RFJ3': 1.44,
#     'rh_RFJ4': 0
# }

# # pc1_config_0.6
# sample_data = {
#     'rh_FFJ4': -0.00780057,
#     'rh_FFJ3': 0.61548911,
#     'rh_FFJ2': 0.77643357,
#     # 'rh_FFJ1': 1.03083925,
#     'rh_MFJ4': -0.01636032,
#     'rh_MFJ3': 0.8231932,
#     'rh_MFJ2': 0.88202607,
#     # 'rh_MFJ1': 1.1357188,
#     'rh_RFJ4': 0.04239293,
#     'rh_RFJ3': 0.6697904,
#     'rh_RFJ2': 0.82158622,
#     # 'rh_RFJ1': 1.08070623,
#     'rh_THJ5': -0.44577062,
#     'rh_THJ4': 0.66701107,
#     'rh_THJ2': 0.01576079,
#     'rh_THJ1': 0.7747529
# }


# 保存するディレクトリとファイルパス
directory = '/home/tokoro/.mujoco/synergy/ros/operate_configs'
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = os.path.join(directory, 'initial_config.pkl')  # 初期姿勢
# file_path = os.path.join(directory, 'pc1_config_0.6.pkl')  # PC1で, 動かした後

# Pickleファイルとして保存
with open(file_path, 'wb') as f:
    pickle.dump(sample_data, f)

print(f"Pickleファイルが作成されました: {file_path}")