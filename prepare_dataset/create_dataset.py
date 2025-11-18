import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe hand and drawing modules
hand_module = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

# Create Hand object for static image processing with minimum confidence threshold
hand_detector = hand_module.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset_path = '../dataset'

# Lists to store processed landmark data and corresponding labels
landmark_data = []
dataset_labels = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        
        normalized_landmarks = []  # List to store normalized landmark coordinates
        x_coords = []
        y_coords = []

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks in the image
        results = hand_detector.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and normalize landmarks
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)

                # Normalize landmarks by subtracting the minimum x and y values
                min_x, min_y = min(x_coords), min(y_coords)
                for i in range(len(hand_landmarks.landmark)):
                    normalized_landmarks.append(x_coords[i] - min_x)
                    normalized_landmarks.append(y_coords[i] - min_y)

            # Ensure the correct number of landmarks (42 points) are extracted
            if len(normalized_landmarks) == 42:
                landmark_data.append(normalized_landmarks)
                dataset_labels.append(category)

# Save the processed dataset to a pickle file
with open('../data_model/dataset.pickle', 'wb') as file:
    pickle.dump({'data': landmark_data, 'labels': dataset_labels}, file)