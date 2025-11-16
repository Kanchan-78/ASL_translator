import cv2
from ultralytics import YOLO
import time

MODEL_PATH = r"C:\Users\Hello Dell !!\Desktop\project-kanchan\runs\asl_project\asl_yolo_cls\weights\last.pt"

# Load YOLOv8 model
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam. Try changing webcam index, e.g., cv2.VideoCapture(1)")
    exit()

print("🎥 Webcam started! Press 'q' to quit.\n")

frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error capturing frame.")
        break

    frame_count += 1

    # Predict using YOLO
    results = model.predict(frame, verbose=False)

    if results and len(results) > 0:
        res = results[0]

        if hasattr(res, "probs") and res.probs is not None:
            pred_idx = res.probs.top1
            pred_label = model.names[pred_idx]
            conf = res.probs.top1conf.item()

            # Show prediction on video
            cv2.putText(frame, f"{pred_label}: {conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    # FPS calculation every 30 frames
    if frame_count % 30 == 0:
        current_time = time.time()
        fps = 30 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow("ASL Detection - Real-Time Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed. 👋 ASL detection stopped.")
