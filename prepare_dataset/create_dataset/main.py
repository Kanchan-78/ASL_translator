import warnings
warnings.filterwarnings("ignore")

import cv2
import os
import albumentations as A
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

augmenter = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.HueSaturationValue(p=0.4),
    A.RandomGamma(p=0.4),
    A.Rotate(limit=25, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=25, p=0.7),
    A.MotionBlur(p=0.3),
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5)
])

def generate_one(output_index, img, save_dir):
    augmented = augmenter(image=img)["image"]
    cv2.imwrite(os.path.join(save_dir, f"{output_index}.jpg"), augmented)

def process_letter(args):
    letter, img_path = args
    img = cv2.imread(img_path)

    save_dir = os.path.join("augmented", letter)
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(1, 4001), desc=f"{letter} (4000 images)", position=0, leave=True):
        generate_one(i, img, save_dir)

def main():
    input_dir = "letters"

    files = [
        (os.path.splitext(f)[0], os.path.join(input_dir, f))
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with Pool(cpu_count()) as p:
        p.map(process_letter, files)

if __name__ == "__main__":
    main()
