# GUIDE

projected_views_blender: RGB 이미지 파일 넣는 곳
projected_views_blender_nir: NIR 이미지 파일 넣는 곳.
RGB이미지와 NIR이미지 개수는 동일해야 함

이미지 넣은 후 
* process_vlm_image.py: 이미지 전처리
* query_vlm.py: VLM query(open router key 사용자에 맞추어 업데이트 필요)

projected_views_ply: 3D back projection pcd 파일 저장처
