import os
import sys
import pickle
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hand_module = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

hand_detector = hand_module.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset_path = '../dataset'

landmark_data = []
dataset_labels = []

total_images = 0
if os.path.exists(dataset_path):
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            total_images += len(os.listdir(category_path))
else:
    print(f"Error: Folder {dataset_path} not found.")
    exit()

print(f"Found {total_images} images to process.", flush=True)

try:
    with tqdm(total=total_images, desc="Processing", unit="img", file=sys.stdout, ncols=80) as pbar:
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        pbar.update(1)
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    H, W, _ = image.shape
                    results = hand_detector.process(image_rgb)

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        x_coords = []
                        y_coords = []
                        
                        for landmark in hand_landmarks.landmark:
                            x_coords.append(landmark.x * W)
                            y_coords.append(landmark.y * H)

                        min_x, min_y = min(x_coords), min(y_coords)
                        normalized_landmarks = []

                        for i in range(len(hand_landmarks.landmark)):
                            normalized_landmarks.append(x_coords[i] - min_x)
                            normalized_landmarks.append(y_coords[i] - min_y)

                        max_value = max(list(map(abs, normalized_landmarks)))

                        if max_value > 0:
                            normalized_landmarks = [n / max_value for n in normalized_landmarks]

                        if len(normalized_landmarks) == 42:
                            landmark_data.append(normalized_landmarks)
                            dataset_labels.append(category)

                except Exception as e:
                    pass

                finally:
                    pbar.update(1)

    print("\nSaving dataset...", flush=True)
    with open('../data_model/dataset.pickle', 'wb') as file:
        pickle.dump({'data': landmark_data, 'labels': dataset_labels}, file)

    print("Dataset created successfully!", flush=True)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")