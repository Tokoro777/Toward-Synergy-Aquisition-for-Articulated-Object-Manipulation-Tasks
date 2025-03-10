import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from trimesh.creation import box

# 処理済みのジオメトリ名を追跡するセット
processed_geoms = set()

# 全ジオメトリを格納するリスト
meshes = []

def compute_absolute_position_and_rotation(geom, parent_position, parent_rotation):
    """
    親の位置と回転を考慮して、ジオメトリの絶対位置 (3D) と絶対回転行列 (4x4) を計算する。
    """
    if parent_position is None:
        parent_position = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(4)

    # geomタグから位置とオイラー角を取得
    geom_pos = np.array(list(map(float, geom.get('pos', '0 0 0').split())))
    geom_euler = np.array(list(map(float, geom.get('euler', '0 0 0').split())))

    # オイラー角を回転行列へ変換 (順番: sxyz)
    geom_rotation = trimesh.transformations.euler_matrix(*geom_euler, axes='sxyz')

    # 親の回転を掛け合わせ、絶対回転を生成
    absolute_rotation = np.dot(parent_rotation, geom_rotation)

    # 親から見た位置を、親の回転を考慮して変換
    absolute_position = np.dot(parent_rotation[:3, :3], geom_pos) + parent_position

    return absolute_position, absolute_rotation

def process_body(body, parent_position=None, parent_rotation=None, parent_name="root"):
    """
    再帰的に <body> タグとその中の <geom> を処理し、メッシュを作成してグローバルに蓄積する。
    """
    if parent_position is None:
        parent_position = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(4)

    # body タグ自身の pos, euler からボディのローカル変換を計算
    body_pos = np.array(list(map(float, body.get('pos', '0 0 0').split())))
    body_euler = np.array(list(map(float, body.get('euler', '0 0 0').split())))
    body_rotation = trimesh.transformations.euler_matrix(*body_euler, axes='sxyz')

    # 親座標系から見たこのボディの絶対位置＆回転
    current_position = np.dot(parent_rotation[:3, :3], body_pos) + parent_position
    current_rotation = np.dot(parent_rotation, body_rotation)

    # <body> タグ内に含まれる <geom> を処理
    for geom in body.findall("geom"):
        geom_name = geom.get('name', 'unknown')
        geom_type = geom.get('type', 'box')  # デフォルトを box とする
        geom_size = np.array(list(map(float, geom.get('size', '0 0 0').split())))

        # すでに処理済みかどうかを確認
        if geom_name in processed_geoms:
            print(f"Skipping already processed geom: {geom_name}")
            continue

        # ジオメトリ名を記録して重複を防ぐ
        processed_geoms.add(geom_name)

        # 絶対位置＆絶対回転を計算
        geom_pos, geom_rot = compute_absolute_position_and_rotation(geom, current_position, current_rotation)

        print(f"Processing geom: {geom_name}, Type: {geom_type}, Position: {geom_pos}, Size: {geom_size}")

        # box 型ジオメトリを想定
        if geom_type == "box":
            # size は半径的扱いなので 2倍して extents にする
            mesh = box(extents=2 * geom_size)

            # 回転と位置を適用
            mesh.apply_transform(geom_rot)
            mesh.apply_translation(geom_pos)

            # グローバルなリストに追加
            meshes.append(mesh)
        else:
            print(f"Unsupported geom type: {geom_type}. Skipping...")

    # 子ボディ (<body> タグ) がある場合は再帰的に処理
    for child_body in body.findall("body"):
        child_name = child_body.get('name', 'unknown_body')
        process_body(child_body, current_position, current_rotation, parent_name=f"{parent_name}_{child_name}")

def main():
    # 読み込みたい MJCF ファイルのパスを指定
    xml_file = "/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/scissors_model_4finger.xml"

    # MJCF をパース
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # グローバル変数を初期化
    global processed_geoms, meshes
    processed_geoms.clear()
    meshes.clear()

    # ルートから全 <body> を走査してメッシュ作成
    for body in root.findall(".//body"):
        process_body(body)

    # メッシュをすべて結合して1つのメッシュに統合
    if len(meshes) == 0:
        print("No meshes were created. Check your MJCF file.")
        return

    combined_mesh = trimesh.util.concatenate(meshes)
    print("All geoms combined into a single mesh.")

    # ----------------------------------------------
    #  メッシュの z最小値/最大値から中間を計算し、
    #  そこを plane_origin として断面を取得する
    # ----------------------------------------------
    bounds = combined_mesh.bounds
    min_z = bounds[0][2]
    max_z = bounds[1][2]

    # メッシュの z範囲をチェック
    print(f"Mesh Z-bounds: {min_z} ~ {max_z}")

    # メッシュが完全に平面的でない限り、この中間で切れば断面が得られるはず
    mid_z = 0.5 * (min_z + max_z)
    print(f"Cut plane at z = {mid_z}")

    # mid_z の平面で断面を取得
    plane_origin = [0, 0, mid_z]
    plane_normal = [0, 0, 1]

    section = combined_mesh.section(plane_origin=plane_origin,
                                    plane_normal=plane_normal)

    if section is None:
        print("No cross section was found at the mid_z plane. "
              "Possibly the mesh doesn't intersect even that plane.")
        return

    # 断面 (3Dの線分) を 2Dポリゴンに射影
    planar_section, _ = section.to_planar()

    # DXFファイルとして保存する
    output_dxf = "/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/stls/target/scissors_4finger_other.dxf"
    planar_section.export(output_dxf)
    print(f"DXFの2D輪郭を保存しました: {output_dxf}")

if __name__ == "__main__":
    main()
