import torch
import numpy as np
import os
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
import open3d as o3d

# === 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "projected_views_ply"
os.makedirs(output_dir, exist_ok=True)

image_size = 512
fov = 60.0

# === 포인트 클라우드 불러오기 ===
pcd = o3d.io.read_point_cloud("logs/20250709_chair_chair/pointcloud_from_mesh.ply")
points_np = np.asarray(pcd.points)
colors_np = np.asarray(pcd.colors)

points = torch.tensor(points_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
colors = torch.tensor(colors_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
point_cloud = Pointclouds(points=points, features=colors)

# === 렌더러 설정 ===
raster_settings = PointsRasterizationSettings(
    image_size=image_size,
    radius=0.01,
    points_per_pixel=1,  # 각 픽셀당 가장 가까운 100개 포인트
)

rasterizer = PointsRasterizer(raster_settings=raster_settings)
renderer = PointsRenderer(
    rasterizer=rasterizer,
    compositor=AlphaCompositor(),
)

# === 시점 정의 ===
elevations = [0, -60, 60]
azimuths = np.arange(0, 360, 30)

all_point_indices = []

view_idx = 0


        
for elev in elevations:
    for azim in azimuths:
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        # Rasterization 결과만 별도로 얻기
        fragments = rasterizer(point_cloud, cameras=cameras)
        point_indices = fragments.idx[0, ..., 0].cpu().numpy()  # (H, W)

        all_point_indices.append(point_indices)

        # RGB 이미지 렌더링 및 저장
        images = renderer(point_cloud, cameras=cameras)
        img = images[0, ..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        save_path = os.path.join(output_dir, f"view_{view_idx:02d}_e{elev}_a{azim}.png")
        imageio.imwrite(save_path, img)
        print(f"Saved {save_path}")

        view_idx += 1

# === npz 저장 ===
all_point_indices = np.stack(all_point_indices, axis=0)  # (36, H, W)
np.savez_compressed(os.path.join(output_dir, "point_index_map_all.npz"), point_indices=all_point_indices)
print("Saved all point index maps to point_index_map_all.npz")