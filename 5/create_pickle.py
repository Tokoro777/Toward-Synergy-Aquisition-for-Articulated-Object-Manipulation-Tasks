import pickle
import os

# sample_data = {
#     'rh_THJ1': 1.57,
#     'rh_THJ2': 0,
#     'rh_THJ4': 1.22,
#     'rh_THJ5': 0,
#
#     'rh_FFJ1': 1.57,
#     'rh_FFJ2': 0,
#     'rh_FFJ3': 1.44,
#     'rh_FFJ4': 0,
#
#     'rh_MFJ1': 1.57,
#     'rh_MFJ2': 0,
#     'rh_MFJ3': 1.53,
#     'rh_MFJ4': 0,
#
#     'rh_RFJ1': 1.57,
#     'rh_RFJ2': 0,
#     'rh_RFJ3': 1.44,
#     'rh_RFJ4': 0
# }

sample_data = {
    'rh_THJ1': 1.57,
    'rh_THJ2': 0,
    'rh_THJ4': 1.22,
    'rh_THJ5': 0,

    'rh_FFJ1': 1.57,
    'rh_FFJ2': 0,
    'rh_FFJ3': 1.44,
    'rh_FFJ4': 0,

    'rh_MFJ1': 0,
    'rh_MFJ2': 1.57,
    'rh_MFJ3': 1.53,
    'rh_MFJ4': 0,

    'rh_RFJ1': 0,
    'rh_RFJ2': 1.57,
    'rh_RFJ3': 1.44,
    'rh_RFJ4': 0
}

# 保存するディレクトリとファイルパス
directory = '/home/tokoro/.mujoco/synergy/ros/operate_configs'
if not os.path.exists(directory):
    os.makedirs(directory)
s
file_path = os.path.join(directory, 'initial_config.pkl')

# Pickleファイルとして保存
with open(file_path, 'wb') as f:
    pickle.dump(sample_data, f)

print(f"Pickleファイルが作成されました: {file_path}")