import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import time

from typing import Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model_dict = pickle.load(open('./data_model/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
    19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'Delete'
}

class TranslationRequest(BaseModel):
    sentence: str
    lang: str

session_data: Dict[str, Dict[str, any]] = {}

@app.get("/")
async def index():
    return HTMLResponse(open("static/index.html").read())

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon", headers={"Cache-Control": "public, max-age=31353600, immutable"})

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
         raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_aux = []
                y_aux = []
                
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

                if len(data_aux) != 42:
                     return {"prediction": None, "message": "Hand detected but features incomplete"}

                prediction = model.predict([np.asarray(data_aux)])
                prediction_proba = model.predict_proba([np.asarray(data_aux)])
                confidence = float(np.max(prediction_proba))

                predicted_index = int(prediction[0])
                predicted_character = labels_dict.get(predicted_index, "?")

                return {
                    "prediction": predicted_character,
                    "confidence": round(confidence, 4)
                }
        else:
            return {"prediction": None, "message": "No hand detected in image"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    if session_id not in session_data:
        session_data[session_id] = {
            "sentence": "",
            "last_detected_time": None,
            "current_character": None,
            "selection_effect_time": 0,
            "last_active": time.time()
        }

    user = session_data[session_id]
    frame = cv2.resize(frame, (640, 480))
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_aux = []
            y_aux = []

            for lm in hand_landmarks.landmark:
                x_aux.append(lm.x * W)
                y_aux.append(lm.y * H)
            
            min_x, min_y = min(x_aux), min(y_aux)
            data_aux = []

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_aux[i] - min_x)
                data_aux.append(y_aux[i] - min_y)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            max_value = max(list(map(abs, data_aux)))
            if max_value > 0:
                data_aux = [n / max_value for n in data_aux]
            
            if len(data_aux) != 42:
                return {"image": "", "sentence": user["sentence"]}

            x1 = int(min(x_aux)) - 10
            y1 = int(min(y_aux)) - 10
            x2 = int(max(x_aux)) + 10
            y2 = int(max(y_aux)) + 10

            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = float(np.max(prediction_proba))

            predicted_index = int(prediction[0])
            predicted_character = labels_dict.get(predicted_index, "?")

            if user["last_detected_time"] is None:
                user["last_detected_time"] = time.time()
            user["current_character"] = predicted_character

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            display_text = f"{predicted_character} {int(confidence * 100)}%"
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            if time.time() - user["last_detected_time"] >= 2:
                if predicted_character == "Delete":
                    user["sentence"] = user["sentence"][:-1]
                elif predicted_character == " ":
                    user["sentence"] += " "
                elif predicted_character != "?":
                    user["sentence"] += predicted_character
                
                user["last_detected_time"] = None
                user["selection_effect_time"] = time.time()
    else:
        user["last_detected_time"] = None

    if time.time() - user["selection_effect_time"] < 0.5:
        cv2.putText(frame, f'Selected: {user["current_character"]}', (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    img_str = base64.b64encode(buffer).decode('utf-8')

    user["last_active"] = time.time()

    return {
        "image": img_str,
        "sentence": user["sentence"],
        "confidence": round(confidence, 4) if 'confidence' in locals() else 0.0
    }