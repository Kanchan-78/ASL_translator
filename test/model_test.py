import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import cv2
import mediapipe as mp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = '../data_model/model.p'
TEST_DATA_DIR = './test_dataset'
CONFIDENCE_THRESHOLD = 0.3

valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

labels_dict = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
    19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'Delete'
}

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()

model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=CONFIDENCE_THRESHOLD)

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None, "Read Error"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_aux = []
            y_aux = []
            H, W, _ = img.shape
            
            for lm in hand_landmarks.landmark:
                x_aux.append(lm.x * W)
                y_aux.append(lm.y * H)
            
            min_x, min_y = min(x_aux), min(y_aux)
            data_aux = []
            
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_aux[i] - min_x)
                data_aux.append(y_aux[i] - min_y)
            
            max_value = max(list(map(abs, data_aux)))
            if max_value > 0:
                data_aux = [n / max_value for n in data_aux]
            
            if len(data_aux) != 42: return None, "Length Error"
            return np.asarray(data_aux).reshape(1, -1), "Success"
        
        return None, "No Hand Detected"
    except Exception:
        return None, "Processing Error"

y_true = []
y_pred = []
failed_files = []

if not os.path.exists(TEST_DATA_DIR):
    print(f"Error: Test folder '{TEST_DATA_DIR}' not found.")
    exit()

print("Starting Evaluation...")

classes = sorted(os.listdir(TEST_DATA_DIR))

for class_name in classes:
    class_dir = os.path.join(TEST_DATA_DIR, class_name)
    if not os.path.isdir(class_dir): continue
    
    files = [f for f in os.listdir(class_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    for img_name in files:
        img_path = os.path.join(class_dir, img_name)
        features, status = extract_features(img_path)
        
        if status == "Success":
            prediction_idx = int(model.predict(features)[0])
            predicted_char = labels_dict.get(prediction_idx, str(prediction_idx))
            
            y_true.append(class_name)
            y_pred.append(predicted_char)
        else:
            failed_files.append(f"{class_name}/{img_name} ({status})")

print("\n" + "=" * 30)
print("PERFORMANCE REPORT")
print("=" * 30)

total_attempts = len(y_true) + len(failed_files)
det_rate = (len(y_true) / total_attempts * 100) if total_attempts > 0 else 0

print(f"Total Images:       {total_attempts}")
print(f"Hands Detected:     {len(y_true)}")
print(f"Detection Rate:     {det_rate:.2f}%")

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"Model Accuracy:     {acc:.2f}%")
    
    try:
        unique_labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Accuracy: {acc:.2f}%)')
        plt.show()
    except Exception:
        print("Could not generate heatmap.")
else:
    print("No hands detected.")

if failed_files:
    print("\nFailed Images:")
    for f in failed_files: print(f" - {f}")