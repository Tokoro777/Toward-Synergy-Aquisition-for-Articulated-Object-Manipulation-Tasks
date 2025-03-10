import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# XMLから情報を取得する代わりに、手動で情報を定義する
bodies = [
    {"name": "scissors_part1_mirror", "pos": [0, 0, 0], "size": [0.1, 0.008, 0.004]},
    {"name": "scissors_part2_mirror", "pos": [0, 0.024, 0], "size": [0.02, 0.008, 0.004]},
    {"name": "scissors_part3_mirror", "pos": [-0.045, 0.027, 0], "size": [0.053, 0.008, 0.004]},
    {"name": "scissors_part4_mirror", "pos": [-0.058, -0.025, 0], "size": [0.034, 0.008, 0.004]},
    {"name": "scissors_part5_mirror", "pos": [0.045, -0.018, 0], "size": [0.05, 0.008, 0.004]}
]

# STLファイルへの書き出し関数
def write_stl_file(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write("solid Scissors\n")
        for face in faces:
            normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            normal /= np.linalg.norm(normal)
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            for vertex_id in face:
                vertex = vertices[vertex_id]
                f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid Scissors\n")

# 各ボディのSTLデータを生成する関数
def generate_stl_data(bodies):
    vertices = []
    faces = []
    vertex_id = 0
    for body in bodies:
        size = np.array(body["size"]) / 2  # ボディのサイズの半分を取得
        pos = np.array(body["pos"])  # ボディの位置を取得
        # ボディの頂点を定義
        body_vertices = [
            pos + [-size[0], -size[1], -size[2]],
            pos + [size[0], -size[1], -size[2]],
            pos + [size[0], size[1], -size[2]],
            pos + [-size[0], size[1], -size[2]],
            pos + [-size[0], -size[1], size[2]],
            pos + [size[0], -size[1], size[2]],
            pos + [size[0], size[1], size[2]],
            pos + [-size[0], size[1], size[2]]
        ]
        vertices.extend(body_vertices)
        # ボディの面を定義
        body_faces = [
            [vertex_id, vertex_id + 1, vertex_id + 2],
            [vertex_id, vertex_id + 2, vertex_id + 3],
            [vertex_id + 4, vertex_id + 7, vertex_id + 6],
            [vertex_id + 4, vertex_id + 6, vertex_id + 5],
            [vertex_id, vertex_id + 4, vertex_id + 5],
            [vertex_id, vertex_id + 5, vertex_id + 1],
            [vertex_id + 1, vertex_id + 5, vertex_id + 6],
            [vertex_id + 1, vertex_id + 6, vertex_id + 2],
            [vertex_id + 2, vertex_id + 6, vertex_id + 7],
            [vertex_id + 2, vertex_id + 7, vertex_id + 3],
            [vertex_id + 4, vertex_id, vertex_id + 3],
            [vertex_id + 4, vertex_id + 3, vertex_id + 7]
        ]
        faces.extend(body_faces)
        vertex_id += 8
    return vertices, faces

# STLデータを生成
vertices, faces = generate_stl_data(bodies)

# STLファイルに書き出し
write_stl_file("scissors.stl", vertices, faces)

# 3Dプロットして確認
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(*zip(*vertices), triangles=faces, cmap='viridis')
plt.show()
