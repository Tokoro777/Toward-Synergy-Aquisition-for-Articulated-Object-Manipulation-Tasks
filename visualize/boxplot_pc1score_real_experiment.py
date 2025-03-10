import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams  # フォント設定用

import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

def main():
    # Times New Romanをフォントとして設定
    rcParams['font.family'] = 'Times New Roman'

    # 基本ディレクトリ
    base_dir = "/home/tokoro/angle"

    # フォルダリスト
    folders = [
        "0.0_data",
        "0.1_data",
        "0.2_data",
        "0.3_data",
        "0.4_data",
        "0.5_data"
    ]

    # 読み取った角度の誤差（度）を格納するためのリスト
    errors_in_deg = {}

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        nominal_angle_str = folder.replace("_data", "")
        nominal_angle_rad = float(nominal_angle_str)

        file_pattern = os.path.join(folder_path, "*.txt")
        file_list = glob.glob(file_pattern)

        def extract_timestamp(fp):
            filename = os.path.basename(fp)
            match = re.search(r"(\d{14})(?=\.txt$)", filename)
            if match:
                return int(match.group(1))
            else:
                return 0

        file_list.sort(key=extract_timestamp, reverse=True)
        target_files = file_list[:10]

        error_list_deg = []

        for txt_file in target_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "Current Angle:" in line:
                        match_line = re.search(r"Current Angle:\s*([-\d.]+)\s*rad", line)
                        if match_line:
                            current_angle_rad = float(match_line.group(1))
                            diff_rad = abs(current_angle_rad - nominal_angle_rad)
                            diff_deg = diff_rad * 180.0 / np.pi
                            error_list_deg.append(diff_deg)
                        break

        errors_in_deg[nominal_angle_str] = error_list_deg

    labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    data_for_boxplot = [errors_in_deg[label] for label in labels]

    plt.figure(figsize=(10, 4))
    box = plt.boxplot(data_for_boxplot, labels=labels)

    median_line_width = 3
    for median in box['medians']:
        median.set(color='red', linewidth=median_line_width)

    desired = "desired"
    plt.xlabel(f"$A_{{{desired}}}$ [rad]", fontsize=40)
    plt.ylabel('$e$ [°]', fontsize=40)

    plt.ylim(0, 8)
    plt.yticks(range(0, 10, 5))

    plt.grid(True)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.975, bottom=0.26)

    plt.show()


if __name__ == "__main__":
    main()
