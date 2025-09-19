import os
import time
# from paint_it.paint_it import parse_args, main, StableDiffusion
from gaussian_property_main.vlm_predict import run_vlm, run_vlm_nir

if __name__ == '__main__':

    # 2-2. preprocess_images
    directory = "projected_views_blender"

    preprocessed_save_path = directory + "_dirs"

    # # 4. vlm_predict
    print("Feeding images to VLM...")
    # run_vlm(preprocessed_save_path, False, vlm_type="gemini_flash")
    run_vlm_nir(preprocessed_save_path, vlm_type="gemini_flash")


