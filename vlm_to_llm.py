import os
from time import time
from paint_it.paint_it import parse_args, main, StableDiffusion
from gaussian_property_main.folder_organizer import process_images
from gaussian_property_main.sam_preprocess import sam_model_registry, sam_image, save_gpt_input
from gaussian_property_main.vlm_predict import run_vlm
if __name__ == '__main__':

    # 1. Paint_it 

    # 필요한 설정 (args) 가져오기
    args = parse_args()

    # 직접 설정 추가 (필요 시)
    args.objaverse_id = "85819bbcdfee44f8b6f525eb89dd19bd"
    args.identity = "A chair with tools on it"
    args.exp_name = '_'.join((args.identity.split(' ')[1:] + [args.objaverse_id[:6]]))

    # guidance model 불러오기
    guidance = StableDiffusion("cuda", min=args.sd_min, max=args.sd_max)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False

    # main 실행
    main(args, guidance)

    # 2. Folder_organizer
    exp_name = time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name
    path_to_preprocess = os.path.join('./logs', exp_name)
    preprocessed_save_path = path_to_preprocess + "_dirs"
    process_images(path_to_preprocess, True)

    # 3. Sam_preprocess
    sam = sam_model_registry["vit_h"](checkpoint="/local_data_2/urp25su_hanuiseok/sam_vit_h_4b8939.pth").to('cuda')
    sam_image(sam, preprocessed_save_path)
    save_gpt_input(preprocessed_save_path)

    # 4. vlm_predict
    run_vlm(preprocessed_save_path, vlm_type="qwen")