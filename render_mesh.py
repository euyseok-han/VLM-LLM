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

# 디바이스 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 메시 파일 로딩
DATA_DIR = "./data/9ce8ab24383c4c93b4c1c7c3848abc52"
obj_filename = os.path.join(DATA_DIR, "mesh.obj")
mesh = load_objs_as_meshes([obj_filename], device=device)

# 렌더링 설정
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 시점 목록

# 출력 폴더 생성
output_dir = "./2d_output"
os.makedirs(output_dir, exist_ok=True)

# 렌더러 생성 함수
def get_renderer_with_light(elev = 0, azim = 0):
    dist = 5
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

# 렌더링 및 저장
for elev in [0, 90]:
    for azim in range(0, 360, 90):
        renderer = get_renderer_with_light(elev, azim)
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