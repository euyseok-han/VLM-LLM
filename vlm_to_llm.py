import os
import time
from paint_it.paint_it import parse_args, main, StableDiffusion
from gaussian_property_main.folder_organizer import process_images
from gaussian_property_main.sam_preprocess import sam_model_registry, sam_image, save_gpt_input
from gaussian_property_main.vlm_predict import run_vlm

if __name__ == '__main__':

    # 1. Paint_it 
    args = parse_args()

    # 직접 설정 추가 (필요 시)
    args.objaverse_id = "be8fb616a5fb4b03b5e5f5e391d3c8b6"
    args.identity = "a red sofa with wooden legs"

    args.exp_name = '_'.join((args.identity.split(' ')[1:] + [args.objaverse_id[:6]]))

    # # 1-1. guidance model 불러오기
    # guidance = StableDiffusion("cuda", min=args.sd_min, max=args.sd_max)
    # guidance.eval()
    # for p in guidance.parameters():
    #     p.requires_grad = False

    # # 1-2. main 실행
    # main(args, guidance)

    # 2. Folder_organizer
    # 2-2. preprocess_images
    directory = "projected_views_ply"
    
    path_to_preprocess = directory
    preprocessed_save_path = path_to_preprocess + "_dirs"
    print("process image...")
    process_images(path_to_preprocess, True)

    # # 3. Sam_preprocess
    sam = sam_model_registry["vit_h"](checkpoint="/local_data_2/urp25su_hanuiseok/sam_vit_h_4b8939.pth").to('cuda')
    print("sam image...")
    sam_image(sam, preprocessed_save_path)
    print("save gpt input...")
    save_gpt_input(preprocessed_save_path)

    # # 4. vlm_predict
    # print("Feed images to VLM...")
    # run_vlm(preprocessed_save_path, vlm_type="gemini_flash")