import os
import trimesh
import open3d as o3d
import numpy as np
from PIL import Image
def barycentric_sample(triangle, n_samples):
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)

    # 위 삼각형 영역을 벗어나는 샘플 조정
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v

    # 샘플 좌표 계산
    samples = (
        u[:, None] * triangle[0] +
        v[:, None] * triangle[1] +
        w[:, None] * triangle[2]
    )

    return samples, u, v, w

def sample_from_faces_with_color(mesh, texture_img, total_sample_count=40):
    face_vertices = mesh.vertices[mesh.faces]
    face_uvs = mesh.visual.uv[mesh.faces]

    face_areas = mesh.area_faces
    area_ratio = face_areas / face_areas.sum()
    samples_per_face = (area_ratio * total_sample_count * len(face_areas)).astype(int)

    sampled_points = []
    sampled_colors = []
    height, width = texture_img.size

    for i, (verts, uvs) in enumerate(zip(face_vertices, face_uvs)):
        pts, u, v, w = barycentric_sample(verts, samples_per_face[i])
        uv_coords = u[:, None] * uvs[0] + v[:, None] * uvs[1] + w[:, None] * uvs[2]
        # 이미지 좌표계로 변환 (UV는 [0,1])
        uv_coords = np.clip(uv_coords, 0, 1)
        px = (uv_coords[:, 0] * (width - 1)).astype(int)
        py = ((1 - uv_coords[:, 1]) * (height - 1)).astype(int)  # y축 뒤집기

        colors = np.array(texture_img)[py, px] / 255.0

        sampled_points.append(pts)
        sampled_colors.append(colors)

    return np.vstack(sampled_points), np.vstack(sampled_colors)

def load_texture_color_for_vertices(mesh, texture_img):
    # vertex마다 color를 주기 위해 uv 정보를 이용
    uv_coords = mesh.visual.uv
    height, width = texture_img.size
    uv_coords = np.clip(uv_coords, 0, 1)
    px = (uv_coords[:, 0] * (width - 1)).astype(int)
    py = ((1 - uv_coords[:, 1]) * (height - 1)).astype(int)
    colors = np.array(texture_img)[py, px] / 255.0
    return colors

def textured_mesh_to_colored_pointcloud(obj_path, texture_path, output_ply_path, samples_per_face=5):
    mesh = trimesh.load(obj_path, force='mesh', process=False)
    texture_img = Image.open(texture_path).convert("RGB")

    # 1. vertex 가져오기
    vertices = mesh.vertices
    vertex_colors = load_texture_color_for_vertices(mesh, texture_img)

    # 2. face에서 샘플링
    face_points, face_colors = sample_from_faces_with_color(mesh, texture_img, samples_per_face)

    # 3. 병합
    all_points = np.vstack((vertices, face_points))
    all_colors = np.vstack((vertex_colors, face_colors))

    # 4. Open3D 포인트클라우드 저장
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"컬러 PLY 파일 저장 완료: {output_ply_path}")

# === 예시 사용 ===
obj_dir = "logs/20250709_chair_chair/"      # .obj 파일 경로 입력
obj_file_path = os.path.join(obj_dir, "mesh.obj")      # .obj 파일 경로 입력
texture_path = os.path.join(obj_dir, "texture_kd.png")  # 텍스처 이미지 경로 입력
ply_output_path = os.path.join(obj_dir, "pointcloud_from_mesh.ply")  # 저장할 .ply 경로
textured_mesh_to_colored_pointcloud(obj_file_path, texture_path, ply_output_path, samples_per_face=100)

print(f"Point cloud saved as: {ply_output_path}")

# Step 4: 저장한 .ply 파일 불러와 렌더링
pcd_loaded = o3d.io.read_point_cloud(ply_output_path)



# 시각화
o3d.visualization.draw_geometries(
    [pcd_loaded],
    window_name="Rendered Point Cloud (Downsampled)",
    width=800,
    height=600,
)