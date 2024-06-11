# this file packs images into a single npy file
# though eliminating any data augmentation, it is useful for fast data loading
import os
import sys
import numpy as np

from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
IM_SIZE = 256
def load_image(image_path,return_np,image_size:int = IM_SIZE):
    """Load an image and return the path and size of the image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize((image_size, image_size), Image.BICUBIC)
            if return_np:
                img = np.array(img).transpose(2, 0, 1)
            return (image_path,img)
    except Exception as e:
        raise Exception(f'Error loading image {image_path}: {e}')
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    suffix = ('.png','.jpg','jpeg')
    dirs = os.listdir(sample_dir)
    imgs = [dir for dir in dirs if dir.lower().endswith(suffix)]
    num = min(num, len(imgs))
    print(f"Found {len(imgs)} images in {sample_dir}.")
    print(f"[!] resizing all images to {IM_SIZE}x{IM_SIZE} pixels.")
    for i in tqdm(imgs, desc="Building .npz file from samples"):
        
        sample_pil = Image.open(os.path.join(sample_dir, i)).convert('RGB').resize((IM_SIZE, IM_SIZE), Image.BICUBIC)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}_{IM_SIZE}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
def read_images_with_executor(folder_path, max_workers=4, recursive=False,return_np:bool=False, max_size:int = -1):
    """Read all images in the given folder using ThreadPoolExecutor."""
    valid_img_suffix = ['.png', '.jpg', '.jpeg']
    if recursive:
        image_paths = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(tuple(valid_img_suffix)):
                    image_paths.append(os.path.join(root, f))
    else:
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_size > 0:
        image_paths = image_paths[:max_size]
    print(f'Compressing {len(image_paths)} images in {folder_path}')
    return_nps = [return_np]*len(image_paths)
    # use ThreadPoolExecutor to load images in parallel, with max_workers threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(load_image, image_paths,return_nps), total=len(image_paths)))
    # return a list of tuples, each containing the image path and the image
    return results

def main():
    if len(sys.argv) < 3:
        print('Usage: pack_images.py <input_folder> <output_file>|<mode>')
        sys.exit(1)
    input_folder = sys.argv[1]
    output_dir_or_mode = sys.argv[2]
    if output_dir_or_mode == 'npz':
        im_size = 256 if len(sys.argv) < 4 else int(sys.argv[3])
        globals()['IM_SIZE'] = im_size
        create_npz_from_sample_folder(input_folder)
        return
    else:
        output_dir = output_dir_or_mode
    assert os.path.isdir(input_folder), f'Invalid input folder: {input_folder}'
    os.makedirs(output_dir, exist_ok=True)
    image_paths = read_images_with_executor(input_folder,recursive=True,max_workers=64,return_np=True)
    images = []
    idx = 0
    info_dict = {}
    for path, img in image_paths:
        images.append(img)
        info_dict[path] = idx
        idx += 1
    #images = np.stack(images)
    #transpose to NCHW
    #images = images.transpose(0, 3, 1, 2)
    output_file = os.path.join(output_dir, 'images.npy')
    dict_file = os.path.join(output_dir, 'info_dict.npy')
    np.save(output_file, images)
    np.save(dict_file, info_dict)
    print(f'Saved {len(images)} images to {output_file}, info dict to {dict_file}')

if __name__ == '__main__':
    main()