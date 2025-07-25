import open3d as o3d


ply_filename = "logs/20250709_chair_chair/pointcloud_with_property_color.ply"

# Step 4: 저장한 .ply 파일 불러와 렌더링
pcd_loaded = o3d.io.read_point_cloud(ply_filename)
o3d.visualization.draw_geometries(
    [pcd_loaded],
    window_name="Rendered Point Cloud",
    width=800,
    height=600,
)