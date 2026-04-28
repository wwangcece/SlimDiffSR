from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
import cv2


def process_image(hr_image_path, sr_image_path):
    hr_image = np.asarray(Image.open(hr_image_path).convert("RGB"))
    tar_size = 800
    sr_image = cv2.resize(
        hr_image,
        (tar_size, tar_size),
        interpolation=cv2.INTER_NEAREST,
    )

    # sr_image = imresize(hr_image, sizes=(512, 512))
    Image.fromarray(sr_image).save(sr_image_path)


def main():
    hr_path = "/mnt/massive/wangce/SGDM/BasicSR/results/RealESRGAN_x4_full/visualization/RefSR"
    sr_path = "/mnt/massive/wangce/SGDM/BasicSR/results/RealESRGAN_x4_full/visualization/RefSR_800"
    hr_image_names = os.listdir(hr_path)
    os.makedirs(sr_path, exist_ok=True)

    # 使用 thread_map 替代 ThreadPoolExecutor
    with tqdm(total=len(hr_image_names), desc=f"Processing...") as pbar:
        thread_map(
            lambda hr_image_name: process_image(
                os.path.join(hr_path, hr_image_name),
                os.path.join(sr_path, hr_image_name),
            ),
            hr_image_names,
            max_workers=os.cpu_count(),
        )
        pbar.update(len(hr_image_names))


if __name__ == "__main__":
    main()
