import pickle
import os

# initial_config
# 全ての第一関節は伸ばし, 第三関節のみ曲げる
# sample_data = {
#     'rh_FFJ4': 0,
#     'rh_FFJ3': 1.44,
#     'rh_FFJ2': 0,
#     #'rh_FFJ1': 1.57,
#
#     'rh_MFJ4': 0,
#     'rh_MFJ3': 1.53,
#     'rh_MFJ2': 0,
#     #'rh_MFJ1': 1.57,
#
#     'rh_RFJ4': 0,
#     'rh_RFJ3': 1.44,
#     'rh_RFJ2': 0,
#     #'rh_RFJ1': 1.57,
#
#     'rh_THJ5': 0,
#     'rh_THJ4': 1.22,
#     'rh_THJ2': 0,
#     'rh_THJ1': 0
# }


# # pc1_config_0.6
sample_data = {
    'rh_FFJ4': 0.01178833,
    'rh_FFJ3': 0.75908477,
    'rh_FFJ2': 0.83604953,

    'rh_MFJ4': -0.04035195,
    'rh_MFJ3': 0.66659031,
    'rh_MFJ2': 0.8807316,

    'rh_RFJ4': 0.02728771,
    'rh_RFJ3': 0.78746654,
    'rh_RFJ2': 0.76326687,

    'rh_THJ5': -0.46356998,
    'rh_THJ4': 0.62059778,
    'rh_THJ2': -0.04694811,
    'rh_THJ1': 0.92638631
}


# 保存するディレクトリとファイルパス
directory = '/home/tokoro/.mujoco/synergy/ros/operate_configs'
if not os.path.exists(directory):
    os.makedirs(directory)

# file_path = os.path.join(directory, 'initial_config.pkl')  # 初期姿勢
file_path = os.path.join(directory, 'pc1_config_0.6.pkl')  # PC1で, 動かした後

# Pickleファイルとして保存
with open(file_path, 'wb') as f:
    pickle.dump(sample_data, f)

print(f"Pickleファイルが作成されました: {file_path}")