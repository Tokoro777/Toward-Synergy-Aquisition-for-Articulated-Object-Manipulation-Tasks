import pickle
import os

# initial_configおよび8つのfinal joint valuesに基づくデータ
configs = {
    'initial_config': {
        'rh_FFJ4': 0,
        'rh_FFJ3': 1.44,
        'rh_FFJ2': 0,

        'rh_MFJ4': 0,
        'rh_MFJ3': 1.53,
        'rh_MFJ2': 0,

        'rh_RFJ4': 0,
        'rh_RFJ3': 1.44,
        'rh_RFJ2': 0,

        'rh_THJ5': 0,
        'rh_THJ4': 1.22,
        'rh_THJ2': 0,
        'rh_THJ1': 0
    },
    'initial_4finger_config': {
        'rh_FFJ4': 0,
        'rh_FFJ3': 1.57,
        'rh_FFJ2': 0,

        'rh_MFJ4': 0,
        'rh_MFJ3': 1.57,
        'rh_MFJ2': 0,

        'rh_RFJ4': 0,
        'rh_RFJ3': 1.57,
        'rh_RFJ2': 0,

        'rh_THJ5': 0.115,
        'rh_THJ4': 1.22,
        'rh_THJ2': 0,
        'rh_THJ1': 0
    },
    'initial_3finger_config': {
        'rh_FFJ4': 0,
        'rh_FFJ3': 1.57,
        'rh_FFJ2': 0,

        'rh_MFJ4': 0,
        'rh_MFJ3': 1.57,
        'rh_MFJ2': 0,

        'rh_RFJ4': 0,
        'rh_RFJ3': 0,
        'rh_RFJ2': 0,

        'rh_THJ5': 0.115,
        'rh_THJ4': 1.22,
        'rh_THJ2': 0,
        'rh_THJ1': 0
    },
    '0.0_config': {
        'rh_FFJ4': -0.01245162,
        'rh_FFJ3': 0.88069618,
        'rh_FFJ2': 0.76738078,

        'rh_MFJ4': 0.02156389,
        'rh_MFJ3': 0.88384624,
        'rh_MFJ2': 0.7461187,

        'rh_RFJ4': -0.02031977,
        'rh_RFJ3': 0.82289019,
        'rh_RFJ2': 0.79820131,

        'rh_THJ5': 0.30485928,
        'rh_THJ4': 0.62028052,
        'rh_THJ2': -0.02190393,
        'rh_THJ1': -0.71474395
    },
    '0.1_config': {
        'rh_FFJ4': -0.00868069,
        'rh_FFJ3': 0.86177746,
        'rh_FFJ2': 0.77806337,

        'rh_MFJ4': 0.01193183,
        'rh_MFJ3': 0.85004839,
        'rh_MFJ2': 0.76706002,

        'rh_RFJ4': -0.01291362,
        'rh_RFJ3': 0.81737944,
        'rh_RFJ2': 0.79276666,

        'rh_THJ5': 0.18531705,
        'rh_THJ4': 0.62032987,
        'rh_THJ2': -0.02579998,
        'rh_THJ1': 0.74766851
    },
    '0.2_config': {
        'rh_FFJ4': -0.00490975,
        'rh_FFJ3': 0.84285874,
        'rh_FFJ2': 0.78874596,

        'rh_MFJ4': 0.00229977,
        'rh_MFJ3': 0.81625054,
        'rh_MFJ2': 0.78800134,

        'rh_RFJ4': -0.00550747,
        'rh_RFJ3': 0.81186869,
        'rh_RFJ2': 0.78733202,

        'rh_THJ5': 0.06577482,
        'rh_THJ4': 0.62037923,
        'rh_THJ2': -0.02969603,
        'rh_THJ1': 0.78059308
    },
    '0.3_config': {
        'rh_FFJ4': -0.00113882,
        'rh_FFJ3': 0.82394002,
        'rh_FFJ2': 0.79942855,

        'rh_MFJ4': -0.00733229,
        'rh_MFJ3': 0.78245269,
        'rh_MFJ2': 0.80894267,

        'rh_RFJ4': 0.00189869,
        'rh_RFJ3': 0.80635794,
        'rh_RFJ2': 0.78189737,

        'rh_THJ5': -0.0537674,
        'rh_THJ4': 0.62042858,
        'rh_THJ2': -0.03359207,
        'rh_THJ1': 0.81351764
    },
    '0.4_config': {
        'rh_FFJ4': 0.00263212,
        'rh_FFJ3': 0.80502129,
        'rh_FFJ2': 0.81011115,

        'rh_MFJ4': -0.01696435,
        'rh_MFJ3': 0.74865483,
        'rh_MFJ2': 0.82988399,

        'rh_RFJ4': 0.00930484,
        'rh_RFJ3': 0.80084719,
        'rh_RFJ2': 0.77646272,

        'rh_THJ5': -0.17330963,
        'rh_THJ4': 0.62047794,
        'rh_THJ2': -0.03748812,
        'rh_THJ1': 0.84644221
    },
    '0.5_config': {
        'rh_FFJ4': 0.00640306,
        'rh_FFJ3': 0.78610257,
        'rh_FFJ2': 0.82079374,

        'rh_MFJ4': -0.02659642,
        'rh_MFJ3': 0.71485698,
        'rh_MFJ2': 0.85082532,

        'rh_RFJ4': 0.01671099,
        'rh_RFJ3': 0.79533644,
        'rh_RFJ2': 0.77102808,

        'rh_THJ5': -0.29285185,
        'rh_THJ4': 0.62052729,
        'rh_THJ2': -0.04138417,
        'rh_THJ1': 0.87936677
    },
    '0.6_config': {
        'rh_FFJ4': 0.01017399,
        'rh_FFJ3': 0.76718385,
        'rh_FFJ2': 0.83147633,

        'rh_MFJ4': -0.03622848,
        'rh_MFJ3': 0.68105913,
        'rh_MFJ2': 0.87176664,

        'rh_RFJ4': 0.02411714,
        'rh_RFJ3': 0.78982569,
        'rh_RFJ2': 0.76559343,

        'rh_THJ5': -0.41239408,
        'rh_THJ4': 0.62057665,
        'rh_THJ2': -0.04528022,
        'rh_THJ1': 0.91229134
    },
    '0.7_config': {
        'rh_FFJ4': 0.01394493,
        'rh_FFJ3': 0.74826513,
        'rh_FFJ2': 0.84215892,

        'rh_MFJ4': -0.04586054,
        'rh_MFJ3': 0.64726128,
        'rh_MFJ2': 0.89270796,

        'rh_RFJ4': 0.0315233,
        'rh_RFJ3': 0.78431494,
        'rh_RFJ2': 0.76015879,

        'rh_THJ5': -0.5319363,
        'rh_THJ4': 0.620626,
        'rh_THJ2': -0.04917626,
        'rh_THJ1': 0.9452159
    }
}

# 保存するディレクトリ
directory = '/home/tokoro/.mujoco/synergy/ros/operate_configs'
if not os.path.exists(directory):
    os.makedirs(directory)

# 各configをファイルに保存
for config_name, sample_data in configs.items():
    file_path = os.path.join(directory, f'{config_name}.pkl')

    # Pickleファイルとして保存
    with open(file_path, 'wb') as f:
        pickle.dump(sample_data, f)

    print(f"Pickleファイルが作成されました: {file_path}")
