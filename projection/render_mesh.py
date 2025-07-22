import argparse
import os
import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    camera_position_from_spherical_angles,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
import trimesh
import numpy as np

# 디바이스 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_camera_distance_from_obj(obj_path, scale_factor=2.4):
    # mesh 로드
    mesh = trimesh.load(obj_path, force='mesh')

    # AABB bounds
    bounds = mesh.bounds  # shape (2, 3)
    min_corner, max_corner = bounds

    # 대각선 길이 (전체 사이즈)
    diag_len = np.linalg.norm(max_corner - min_corner)

    # 카메라 거리 결정
    distance = diag_len * scale_factor
    return distance


# 메시 파일 로딩
# 렌더러 생성 함수
def get_renderer_with_light(elev = 0, azim = 0, dist=5):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    T_reshaped = T.unsqueeze(2)
    R_transpose = R.transpose(1, 2)
    # C = -torch.bmm(R_transpose, T_reshaped).squeeze(2)
    C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=True, device=device
            )
        )
    lights = PointLights(device=device, location=C)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    return renderer


def render(data_dir):
    obj_filename = os.path.join(data_dir, "mesh.obj")
    dist = get_camera_distance_from_obj(obj_path=obj_filename)
    print("dist: ####", dist)
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # 렌더링 설정
    global raster_settings
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # 시점 목록

    # 출력 폴더 생성
    output_dir = "./2d_output"
    os.makedirs(output_dir, exist_ok=True)

    # 렌더링 및 저장
    for elev in [0, 60, -60]:
        for azim in range(0, 360, 90):
            renderer = get_renderer_with_light(elev, azim, dist)
            image = renderer(mesh)[0, ..., :3].cpu().numpy()
            # 저장
            save_path = os.path.join(output_dir, f"view_{elev}_{azim}.png")
            plt.imsave(save_path, image)
            if elev == 90:
                break




# # 화면 출력도 함께 하고 싶다면 아래 코드 사용
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# for ax, elev in zip(axes.flatten(), range(-90, 91, 30)):
#     image = plt.imread(os.path.join(output_dir, f"view_{elev}.png"))
#     ax.imshow(image)
#     ax.set_title(f"View {elev}")
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "render 3d to 2d image and save")
    parser.add_argument('--input_dir', type=str, default="./data/9ce8ab24383c4c93b4c1c7c3848abc52")
    args = parser.parse_args()
    render(args.input_dir)