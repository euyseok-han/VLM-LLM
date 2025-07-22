import os
import argparse
from PIL import Image
from rembg import remove
from gaussian_property_main.utils.sam_utils import resize_image


def process_images(base_path, remove_bg):
    all_images = sorted((x for x in os.listdir(base_path) if x.endswith('.png')))
    save_base_path = base_path + '_dirs'
    l = len(all_images)
    for i in range(0, l):
        image_name = all_images[i]
        base_name, _ = os.path.splitext(image_name) # name without png extension

        # Create directories for saving processed images
        image_dir = os.path.join(save_base_path, base_name, 'images')
        os.makedirs(image_dir, exist_ok=True)

        # Open the image
        image_path = os.path.join(base_path, image_name)

        with Image.open(image_path) as img_pil:
            if remove_bg:
                # Use Rembg to remove the background and get the mask
                img_pil = remove(img_pil)
            img_pil = resize_image(img_pil, 1280)

            # Save the processed image
            mask_save_path = os.path.join(image_dir, '001.png')
            img_pil.save(mask_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with optional background removal.")
    parser.add_argument('--remove', action='store_true',default=True, help="Remove background from images.")
    parser.add_argument('--folder_path', type=str, default='2d_output', help="Path to the folder containing the images.")
    
    args = parser.parse_args()
    # Constants for directory paths
    base_path = args.folder_path

    process_images(base_path, args.remove)