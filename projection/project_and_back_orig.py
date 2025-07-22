import trimesh
import pyrender
import numpy as np
import cv2
import os
from PIL import Image
from pyntcloud import PyntCloud
import pandas as pd

# ======== [1] Load Textured Mesh ========
mesh_path = "data/13a66af840ee4e3c8426b7302e219963/mesh.obj"
mesh_trimesh = trimesh.load(mesh_path, force='mesh', process=False)

# ======== [2] Center & Scale ========
center = mesh_trimesh.centroid
mesh_trimesh.apply_translation(-center)
scale = 1.0 / np.max(mesh_trimesh.bounding_box.extents)
mesh_trimesh.apply_scale(scale)

# ======== [3] Convert to pyrender.Mesh with texture ========
scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
scene.add(mesh)

# ======== [4] Camera Setup ========
fov = np.pi / 3
camera = pyrender.PerspectiveCamera(yfov=fov)
camera_pose = np.eye(4)
bbox = mesh_trimesh.bounding_box_oriented
radius = np.linalg.norm(bbox.extents) / 2
cam_z = radius / np.tan(fov / 2) * 1.2
print(cam_z)
camera_pose[2, 3] = cam_z
scene.add(camera, pose=camera_pose)

# ======== [5] Lighting ========
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=camera_pose)

# ======== [6] Render ========
viewport_width, viewport_height = 512, 512
r = pyrender.OffscreenRenderer(viewport_width, viewport_height)
color, depth = r.render(scene)

print(depth>0)
# ======== [7] Save Rendered Image (optional) ========
cv2.imshow("Rendered Mesh", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# # ======== [8] Back-project depth to 3D point cloud ========
# fx = fy = 0.5 * viewport_height / np.tan(camera.yfov / 2)
# cx = viewport_width / 2
# cy = viewport_height / 2

# i, j = np.meshgrid(np.arange(viewport_width), np.arange(viewport_height))
# z = depth
# x = (i - cx) * z / fx
# y = (j - cy) * z / fy

# points_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)
# colors = color.reshape(-1, 3)

# # Remove background (depth==0)
# valid = (z > 0).reshape(-1)
# points_valid = points_cam[valid]
# colors_valid = colors[valid]

# # ======== [9] Save to .ply ========
# df = pd.DataFrame(np.hstack([points_valid, colors_valid]), columns=["x", "y", "z", "red", "green", "blue"])
# cloud = PyntCloud(df)
# cloud.to_file("backprojected_pointcloud.ply")
