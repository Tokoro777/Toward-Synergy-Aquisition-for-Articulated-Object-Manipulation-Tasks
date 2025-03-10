import matplotlib.pyplot as plt
import numpy as np

# データの定義
data_0_4rad = [125.48, 125.16, 124.19, 127.42, 120.00, 120.32, 121.94, 126.13, 128.39, 129.35]
data_0_2rad = [142.90, 142.26, 143.23, 142.26, 136.13, 142.58, 144.52, 141.94, 141.61, 140.00]
data_0_0rad = [159.03, 159.35, 156.77, 161.29, 157.74, 157.42, 156.77, 157.42, 158.06, 157.74]

# 基準角度
reference_angle = 158.39  # 基準角度[°]

# ラジアン値を度に変換
rad_to_deg = lambda rad: rad * (180 / np.pi)
deg_0_4 = rad_to_deg(0.4)
deg_0_2 = rad_to_deg(0.2)
deg_0_0 = rad_to_deg(0.0)

# ステップ 1: 基準角度との差を計算
differences_0_4rad = [abs(x - reference_angle) for x in data_0_4rad]
differences_0_2rad = [abs(x - reference_angle) for x in data_0_2rad]
differences_0_0rad = [abs(x - reference_angle) for x in data_0_0rad]

# ステップ 2: 絶対誤差を計算
errors_0_4rad = [abs(diff - deg_0_4) for diff in differences_0_4rad]
errors_0_2rad = [abs(diff - deg_0_2) for diff in differences_0_2rad]
errors_0_0rad = [abs(diff - deg_0_0) for diff in differences_0_0rad]

# 平均値を計算
means = [np.mean(errors_0_0rad), np.mean(errors_0_2rad), np.mean(errors_0_4rad)]

# 箱ひげ図を作成（透明に設定）
plt.figure(figsize=(10, 6))
box = plt.boxplot(
    [errors_0_0rad, errors_0_2rad, errors_0_4rad],
    labels=['0.0', '0.2', '0.4'],
    patch_artist=True,  # 箱の色を設定可能にする
    showmeans=False
)

# 各箱を無色に設定
for patch in box['boxes']:
    patch.set(facecolor='none', edgecolor='black')  # 無色かつ黒い枠線

# 中央値の線を太くする
median_line_width = 3  # 中央値の線の太さ
for median in box['medians']:
    median.set(color='red', linewidth=median_line_width)

# # 平均値を赤い線で表示
# line_width = 2.5  # 赤い線の太さを調整
# for i, mean in enumerate(means, start=1):
#     plt.plot([i - 0.15, i + 0.15], [mean, mean], color='red', linewidth=line_width)

# ラベルの設定
desired = "desired"
plt.xlabel(f"$A_{{{desired}}}$ [rad]", fontsize=40)
plt.ylabel('$e$ [°]', fontsize=40)

# 縦軸の範囲と目盛り設定
plt.ylim(0, 17)
plt.yticks(range(0, 20, 5))

# グリッド、ラベルサイズ、余白調整
plt.grid(True)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplots_adjust(left=0.148, right=0.98, top=0.975, bottom=0.17)

# 図の表示
plt.show()
