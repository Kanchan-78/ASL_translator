
### Folder structure

```
ASL_Translator/
├── data_model/
│   ├── dataset.pickle
│   └── model.p
│
├── model_train/
│   └── model_train.py
│
├── prepare_dataset/
│   ├── collection_img.py
│   └── create_dataset.py
│
├── static/
│   ├── favicon.ico
│   ├── index.html
│   ├── script.js
│   ├── signs.PNG
│   └── style.css
│
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
└── uv.lock
```

✨ **Notes:**
- `ASL_Translator/` is project directory.
- Each folder groups related functionality:
  - **data_model/** → stores serialized datasets and models.
  - **model_train/** → contains training scripts.
  - **prepare_dataset/** → scripts for dataset preparation and image collection.
  - **static/** → frontend assets (HTML, CSS, JS, images, favicon).
- Root-level files (`main.py`, `pyproject.toml`, `uv.lock`, etc.) handle execution, environment, and dependency management.

**Tools Used**
- HTML, CSS, JS
- Tkinter, FastAPI, Open-CV, Mediapipe, RandomForestClassifier