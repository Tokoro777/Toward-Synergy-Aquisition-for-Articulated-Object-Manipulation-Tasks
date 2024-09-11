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

# pc1_config_0.0
# sample_data = {
#     'rh_FFJ4': -0.01443463,
#     'rh_FFJ3': 0.89064492,
#     'rh_FFJ2': 0.76176316,
#
#     'rh_MFJ4': 0.02662908,
#     'rh_MFJ3': 0.90161942,
#     'rh_MFJ2': 0.73510634,
#
#     'rh_RFJ4': -0.02421442,
#     'rh_RFJ3': 0.82578812,
#     'rh_RFJ2': 0.80105921,
#
#     'rh_THJ5': 0.36772262,
#     'rh_THJ4': 0.62025456,
#     'rh_THJ2': -0.01985513,
#     'rh_THJ1': 0.69743
# }

# pc1_config_0.2
# sample_data = {
#     'rh_FFJ4': -0.00569365,
#     'rh_FFJ3': 0.84679153,
#     'rh_FFJ2': 0.78652528,
#
#     'rh_MFJ4': 0.00430207,
#     'rh_MFJ3': 0.82327638,
#     'rh_MFJ2': 0.78364809,
#
#     'rh_RFJ4': -0.00704705,
#     'rh_RFJ3': 0.81301426,
#     'rh_RFJ2': 0.78846176,
#
#     'rh_THJ5': 0.09062508,
#     'rh_THJ4': 0.62036897,
#     'rh_THJ2': -0.02888612,
#     'rh_THJ1': 0.77374877
# }

# pc1_config_0.4
# sample_data = {
#     'rh_FFJ4': 0.00304734,
#     'rh_FFJ3': 0.80293815,
#     'rh_FFJ2': 0.81128741,
#
#     'rh_MFJ4': -0.01802494,
#     'rh_MFJ3': 0.74493334,
#     'rh_MFJ2': 0.83218985,
#
#     'rh_RFJ4': 0.01012033,
#     'rh_RFJ3': 0.8002404,
#     'rh_RFJ2': 0.77586431,
#
#     'rh_THJ5': -0.18647245,
#     'rh_THJ4': 0.62048337,
#     'rh_THJ2': -0.03791712,
#     'rh_THJ1': 0.85006754
# }

# pc1_config_0.6
# sample_data = {
#     'rh_FFJ4': 0.01178833,
#     'rh_FFJ3': 0.75908477,
#     'rh_FFJ2': 0.83604953,
#
#     'rh_MFJ4': -0.04035195,
#     'rh_MFJ3': 0.66659031,
#     'rh_MFJ2': 0.8807316,
#
#     'rh_RFJ4': 0.02728771,
#     'rh_RFJ3': 0.78746654,
#     'rh_RFJ2': 0.76326687,
#
#     'rh_THJ5': -0.46356998,
#     'rh_THJ4': 0.62059778,
#     'rh_THJ2': -0.04694811,
#     'rh_THJ1': 0.92638631
# }

# pc1_config_0.7
sample_data = {
    'rh_FFJ4': 0.01615882,
    'rh_FFJ3': 0.73715807,
    'rh_FFJ2': 0.8484306,

    'rh_MFJ4': -0.05151546,
    'rh_MFJ3': 0.62741879,
    'rh_MFJ2': 0.90500248,

    'rh_RFJ4': 0.0358714,
    'rh_RFJ3': 0.78107961,
    'rh_RFJ2': 0.75696814,

    'rh_THJ5': -0.60211875,
    'rh_THJ4': 0.62065498,
    'rh_THJ2': -0.05146361,
    'rh_THJ1': 0.9645457
}


# 保存するディレクトリとファイルパス
directory = '/home/tokoro/.mujoco/synergy/ros/operate_configs'
if not os.path.exists(directory):
    os.makedirs(directory)

# file_path = os.path.join(directory, 'initial_config.pkl')  # 初期姿勢
# file_path = os.path.join(directory, 'pc1_config_0.0.pkl')  # PC1で, 動かした後
# file_path = os.path.join(directory, 'pc1_config_0.2.pkl')  # PC1で, 動かした後
# file_path = os.path.join(directory, 'pc1_config_0.4.pkl')  # PC1で, 動かした後
# file_path = os.path.join(directory, 'pc1_config_0.6.pkl')  # PC1で, 動かした後
file_path = os.path.join(directory, 'pc1_config_0.7.pkl')  # PC1で, 動かした後

# Pickleファイルとして保存
with open(file_path, 'wb') as f:
    pickle.dump(sample_data, f)

print(f"Pickleファイルが作成されました: {file_path}")