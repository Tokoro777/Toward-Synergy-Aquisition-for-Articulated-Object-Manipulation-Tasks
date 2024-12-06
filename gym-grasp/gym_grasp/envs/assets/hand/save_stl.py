import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from trimesh.creation import box

# 処理済みのジオメトリを追跡するセット
processed_geoms = set()

def compute_absolute_position_and_rotation(geom, parent_position, parent_rotation):
    """
    親の位置と回転を考慮して、絶対位置と絶対回転を計算
    """
    if parent_position is None:
        parent_position = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(4)

    geom_pos = np.array(list(map(float, geom.get('pos', '0 0 0').split())))
    geom_euler = np.array(list(map(float, geom.get('euler', '0 0 0').split())))

    geom_rotation = trimesh.transformations.euler_matrix(*geom_euler, axes='sxyz')

    absolute_rotation = np.dot(parent_rotation, geom_rotation)
    absolute_position = np.dot(parent_rotation[:3, :3], geom_pos) + parent_position

    return absolute_position, absolute_rotation

def process_body(body, parent_position=None, parent_rotation=None, parent_name="root"):
    """
    再帰的にボディとジオメトリを処理
    """
    if parent_position is None:
        parent_position = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(4)

    body_pos = np.array(list(map(float, body.get('pos', '0 0 0').split())))
    body_euler = np.array(list(map(float, body.get('euler', '0 0 0').split())))
    body_rotation = trimesh.transformations.euler_matrix(*body_euler, axes='sxyz')

    current_position = np.dot(parent_rotation[:3, :3], body_pos) + parent_position
    current_rotation = np.dot(parent_rotation, body_rotation)

    for geom in body.findall("geom"):
        geom_name = geom.get('name', 'unknown')
        geom_type = geom.get('type', 'box')
        geom_size = np.array(list(map(float, geom.get('size', '0 0 0').split())))

        # すでに処理済みのジオメトリかを確認
        if geom_name in processed_geoms:
            print(f"Skipping already processed geom: {geom_name}")
            continue

        # ジオメトリを処理済みとして追加
        processed_geoms.add(geom_name)

        geom_pos, geom_rot = compute_absolute_position_and_rotation(geom, current_position, current_rotation)

        print(f"Processing geom: {geom_name}, Type: {geom_type}, Absolute Position: {geom_pos}, Size: {geom_size}")

        if geom_type == "box":
            mesh = box(extents=2 * geom_size)
            mesh.apply_transform(geom_rot)
            mesh.apply_translation(geom_pos)
            meshes.append(mesh)
        else:
            print(f"Unsupported geom type: {geom_type}. Skipping...")

    for child_body in body.findall("body"):
        child_name = child_body.get('name', 'unknown_body')
        process_body(child_body, current_position, current_rotation, parent_name=f"{parent_name}_{child_name}")

# メインスクリプト
xml_file = "/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/hand/scissors_model.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

meshes = []

for body in root.findall(".//body"):
    process_body(body)

combined_mesh = trimesh.util.concatenate(meshes)

output_stl = "/home/tokoro/.mujoco/synergy/gym-grasp/gym_grasp/envs/assets/stls/target/scissors_model.stl"
combined_mesh.export(output_stl)
print(f"STLファイルが保存されました: {output_stl}")
