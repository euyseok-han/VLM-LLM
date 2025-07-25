import trimesh
import pyrender
import numpy as np
import cv2
import os
from PIL import Image
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d

# ======== [0] Setup Output Directory ========
output_dir = "projected_views"
os.makedirs(output_dir, exist_ok=True)

# ======== [1] Load Textured Mesh ========
mesh_path = "logs/20250709_chair_chair/mesh.obj"
mesh_trimesh = trimesh.load(mesh_path, force='mesh', process=False)

# ======== [2] Center & Scale ========
center = mesh_trimesh.centroid
mesh_trimesh.apply_translation(-center)
scale = 1.0 / np.max(mesh_trimesh.bounding_box.extents)
mesh_trimesh.apply_scale(scale)

# ======== [3] Convert to pyrender.Mesh with texture ========
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)

# ======== [4] Camera Parameters ========
viewport_width, viewport_height = 512, 512
fov = np.pi / 3
camera = pyrender.PerspectiveCamera(yfov=fov)
radius = np.linalg.norm(mesh_trimesh.bounding_box.extents) / 2
cam_distance = radius / np.tan(fov / 2) * 1.2

fx = fy = 0.5 * viewport_height / np.tan(camera.yfov / 2)
cx = viewport_width / 2
cy = viewport_height / 2

r = pyrender.OffscreenRenderer(viewport_width, viewport_height)

# ======== [5] Generate Multiple Views ========
view_idx = 0
camera_poses = []
elevations = [-80, -60, -30, -15, 0, 15, 30, 60, 80]
azimuths = np.arange(0, 360, 30) #np.arange(0, 360, 30)

merged_points = []
merged_colors = []

for elev in elevations:
    for azim in azimuths:
        # New scene for each render
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0])
        scene.add(mesh)
        
        # Camera pose using spherical coordinates
        theta = np.radians(azim)
        phi = np.radians(90 - elev)
        x = cam_distance * np.sin(phi) * np.cos(theta)
        y = cam_distance * np.sin(phi) * np.sin(theta)
        z = cam_distance * np.cos(phi)

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [x, y, z]

        # Look at origin
        direction = camera_pose[:3, 3]  # Camera looks towards origin
        up = np.array([0, 1, 0]) if elev != -90 else np.array([0, 0, -1])  # Special case for looking straight down
        
        z_axis = direction / np.linalg.norm(direction)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        camera_pose[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T
        
        z_axi = -direction / np.linalg.norm(direction)
        x_axi = np.cross(up, z_axi)
        x_axi /= np.linalg.norm(x_axi)
        y_axi = np.cross(z_axi, x_axi)
        R_prime = np.vstack([x_axi, y_axi, z_axi]).T

        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=camera_pose)

        # ===== Render =====
        color, depth = r.render(scene)

        # ===== Save rendered image as .png =====
        image_save_path = os.path.join(output_dir, f"view_{view_idx:02d}_e{elev}_a{azim}.png")
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_save_path, color_bgr)

        # ===== Back-project =====
        i, j = np.meshgrid(np.arange(viewport_width), np.arange(viewport_height))
        z = depth
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        points_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(-1, 3)

        valid = (z > 0).reshape(-1)
        points_valid = points_cam[valid]
        colors_valid = colors[valid]

        # Transform points from camera coordinates to world coordinates
        # Note: camera_pose is camera-to-world transformation
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        points_world = (R_prime @ points_valid.T).T + t

        merged_points.append(points_world)
        merged_colors.append(colors_valid)

        camera_poses.append((elev, azim, camera_pose.copy()))
        view_idx += 1

# ====== Save merged PLY ======
all_points = np.vstack(merged_points)
all_colors = np.vstack(merged_colors)

# # Create point cloud with colors
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(all_points)
# pcd.colors = o3d.utility.Vector3dVector(all_colors / 255.0)

# # Apply voxel downsampling to reduce noise
# voxel_size = 0.01
# down_pcd = pcd.voxel_down_sample(voxel_size)

# # Save the point cloud
# merged_path = os.path.join(output_dir, "merged_pointcloud.ply")
# o3d.io.write_point_cloud(merged_path, down_pcd)

# # ======== [6] Save All Camera Poses ========
# pose_save_path = os.path.join(output_dir, "camera_poses.npz")
# np.savez(pose_save_path, poses=np.array([p[2] for p in camera_poses]), 
#          elevations=np.array([p[0] for p in camera_poses]), 
#          azimuths=np.array([p[1] for p in camera_poses]))
# print(f"Saved {view_idx} views and camera poses to {output_dir}")

# # Visualize the point cloud
# o3d.visualization.draw_geometries(
#     [down_pcd],
#     window_name="Rendered Point Cloud",
#     width=800,
#     height=600,
# )