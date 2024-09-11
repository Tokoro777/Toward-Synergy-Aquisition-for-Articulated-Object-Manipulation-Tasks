import matplotlib.pyplot as plt

# 各角度でのデータ
data_0_4rad = [125.48, 125.16, 124.19, 127.42, 120.00, 120.32, 121.94, 126.13, 128.39, 129.35]
data_0_2rad = [142.90, 142.26, 143.23, 142.26, 136.13, 142.58, 144.52, 141.94, 141.61, 140.00]
data_0_0rad = [159.03, 159.35, 156.77, 161.29, 157.74, 157.42, 156.77, 157.42, 158.06, 157.74]

# 基準角度
reference_angle = 158.39

# 相対誤差の計算
relative_errors_0_4rad = [abs(reference_angle - angle) for angle in data_0_4rad]
relative_errors_0_2rad = [abs(reference_angle - angle) for angle in data_0_2rad]
relative_errors_0_0rad = [abs(reference_angle - angle) for angle in data_0_0rad]

# 箱ひげ図の作成
plt.figure(figsize=(12, 6))

# 箱ひげ図の描画（0.0 rad、0.2 rad、0.4 radの順）
plt.boxplot([relative_errors_0_0rad, relative_errors_0_2rad, relative_errors_0_4rad],
            labels=['0.0 rad (0.0 °)', '0.2 rad (11.46 °)', '0.4 rad (22.92 °)'],
            vert=True,
            widths=0.6)

# 赤い点の追加（サイズを大きく設定）
plt.scatter([1], [0], color='red', s=100, zorder=5)  # 0.0 rad の箱ひげ図に赤い点
plt.scatter([2], [11.46], color='red', s=100, zorder=5)  # 0.2 rad の箱ひげ図に赤い点
plt.scatter([3], [22.92], color='red', s=100, zorder=5)  # 0.4 rad の箱ひげ図に赤い点

# 軸ラベルとタイトルの設定
current = "current"
plt.ylabel(f"$A_{{{current}}}$ [°]", fontsize=30)
desired = "desired"
plt.xlabel(f"$A_{{{desired}}}$ [rad]", fontsize=30)

# 軸の目盛りのフォントサイズを設定
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 縦軸の範囲を設定（0度から）
plt.ylim(bottom=0)

# 上部と下部の余白を調整
plt.subplots_adjust(top=0.98, bottom=0.15)

plt.show()
