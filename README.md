# 🤚 MudraVision AI — Neural Gesture Analysis

> **Bharatiya Natya · Digital Preservation**  
> An AI-powered neural vision system that recognizes classical Indian dance hand gestures — preserving millennia of cultural heritage through deep learning, fully hand-gesture controlled.

![MudraVision Banner](https://img.shields.io/badge/MudraVision-Neural%20Gesture%20Analysis-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNetV2--M-red?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square&logo=fastapi)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FJS-yellow?style=flat-square&logo=html5)

---

## 📥 Downloads

| Resource | Description | Link |
|---|---|---|
| 🧠 **Trained Model** | `best_mudra_model.pth` — EfficientNetV2-M weights trained on 50 classical mudras | [**Download Model**](https://drive.google.com/file/d/1nl_80THFJO5SJ6mOiBYqfd1uv6rIeUXR/view?usp=sharing) |
| 💻 **Application Source** | Full source code including `mudra2.html` frontend + `main.py` backend | [**Download App**](https://drive.google.com/file/d/16qOCRMLuTWxFl8aHs-bZtw4c39MQbjX7/view?usp=sharing) |

---

## ✨ Features

- **🎯 50 Classical Mudra Recognition** — Identifies Asamyuta (single-hand) and Samyuta (double-hand) mudras from the Natya Shastra
- **📷 Real-time Gesture Camera Control** — Fully hands-free navigation using gestures detected via webcam
- **🎬 Video Analyzer** — Upload a classical dance video; the AI logs every mudra with timestamps automatically
- **📚 Mudra Library** — Rich database with Devanagari names, meanings, usage, and associated dance forms
- **🏛️ Cultural Context** — Covers Bharatanatyam, Kathak, Odissi, Kuchipudi, Mohiniyattam & more
- **⚡ Local Inference** — FastAPI backend runs entirely on your machine (CPU or CUDA)
- **🖥️ Single-File Frontend** — No frameworks or build step needed; just open `mudra2.html` in Chrome

---

## 🖼️ Screenshots

| Hero Page | Gesture Controls | Mudra Analyzer |
|---|---|---|
| "Identify Sacred Mudras" landing | Index-point, Fist, Pinch, Shaka & more | Drop image or use gesture cam |

| Video Analyzer | Architecture | Mobile Splash |
|---|---|---|
| Frame-by-frame mudra timestamping | Input → Preprocess → EfficientNetV2 → Classifier → Output | Mudra · Bharatiya Natya app |

---

## 🏗️ System Architecture

```
Image / Frame
     │
     ▼
Preprocess ──── Normalize · Crop · Landmark Extract (320×320 RGB)
     │
     ▼
EfficientNetV2-M ──── PyTorch Model · Local Inference (backbone feature_dim=1280)
     │
     ▼
Attention Layer ──── Linear(1280→512) → ReLU → Linear(512→1280) → Sigmoid
     │
     ▼
Classifier ──── FC(1024)→BN→ReLU→Drop(0.5) → FC(512)→BN→ReLU→Drop(0.4)
                → FC(256)→BN→ReLU→Drop(0.3) → FC(num_classes)
     │
     ▼
Output ──── Mudra Label · Confidence Score · Cultural Data
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip
- Google Chrome (for gesture camera features)
- CUDA GPU *(optional, CPU works too)*

### 1. Download the Source

Download the application source from the **[App Download link](#-downloads)** above and extract it.

### 2. Install Python Dependencies

```bash
pip install fastapi uvicorn torch torchvision pillow timm pydantic
```

### 3. Download the Trained Model

Download `best_mudra_model.pth` from the **[Model Download link](#-downloads)** above and place it in the project folder along with `mudra_names.txt`:

```
mudra/
├── main.py
├── mudra2.html
├── best_mudra_model.pth      ← place here
└── mudra_names.txt           ← class labels file
```

### 4. Update Model Path in `main.py`

Open `main.py` and update these two lines to match your system path:

```python
MODEL_PATH   = "path/to/your/best_mudra_model.pth"
CLASSES_PATH = "path/to/your/mudra_names.txt"
```

### 5. Start the Backend Server

```bash
python main.py
```

The API will start at `http://localhost:8000`. You should see:

```
INFO  Model ready ✅
INFO  Uvicorn running on http://0.0.0.0:8000
```

### 6. Open the Frontend

Open `mudra2.html` directly in **Google Chrome** and allow camera access when prompted.

---

## 🤌 Gesture Controls

The frontend is fully controllable via hand gestures through your webcam:

| Gesture | Action |
|---|---|
| ☝️ **Index Point** | Move cursor / hover |
| 🤏 **Pinch** | Click / select |
| ✊ **Fist** | Capture / analyze mudra |
| ✌️ **Two Fingers** | Scroll up |
| 🤟 **Three Fingers** | Scroll down |
| 🖐️ **Open Palm** | Toggle legend / close modal |
| 👌 **OK / Circle** | Confirm / save API key |
| 👍 **Thumbs Up** | Scroll to next section |
| 👎 **Thumbs Down** | Scroll to previous section |
| 🤙 **Shaka / Hang Loose** | Upload image trigger |

---

## 🌐 API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Returns model status, device, and loaded class names.

```json
{
  "status": "ready",
  "model_loaded": true,
  "device": "cpu",
  "num_classes": 50,
  "classes": ["Pataka", "Tripataka", "..."]
}
```

### `POST /analyze`
Upload an image to classify the mudra.

**Request:** `multipart/form-data` with field `file` (JPG / PNG / WEBP)

**Response:**
```json
{
  "success": true,
  "mudra": "Anjali",
  "devanagari": "अञ्जलि",
  "confidence": 94.7,
  "classification": "Samyuta Hasta",
  "dance_forms": "All classical dance forms",
  "meaning": "Prayer, salutation, respect",
  "usage": "Universal greeting; devotional prayer",
  "top_predictions": [
    { "name": "Kapota", "confidence": 3.1 },
    { "name": "Pushpaputa", "confidence": 1.2 }
  ]
}
```

### `POST /load-model`
Dynamically load a different model at runtime.

```json
{
  "model_path": "/path/to/model.pth",
  "classes_path": "/path/to/mudra_names.txt"
}
```

---

## 📖 Supported Mudras — 50 Classes

> All 50 mudras are drawn from the **Bharatiya Natya Mudra Kosha** and confirmed as captured in the training dataset.

### 🖐️ Asamyuta Hasta — Single-Hand Mudras (28)

| # | Mudra Name |
|---|---|
| 1 | Pataka |
| 2 | Tripataka |
| 3 | Ardhapataka |
| 4 | Kartarimukha |
| 5 | Mayura |
| 6 | Ardhachandra |
| 7 | Arala |
| 8 | Shukatunda |
| 9 | Mushti |
| 10 | Shikhara |
| 11 | Kapittha |
| 12 | Katakamukha |
| 13 | Suchi |
| 14 | Chandrakala |
| 15 | Padmakosha |
| 16 | Sarpashirsha |
| 17 | Mrigashirsha |
| 18 | Simhamukha |
| 19 | Kangula |
| 20 | Alapadma |
| 21 | Chatura |
| 22 | Bhramara |
| 23 | Hamsasya |
| 24 | Hamsapaksha |
| 25 | Sandamsha |
| 26 | Mukula |
| 27 | Tamrachuda |
| 28 | Trishula |

### 🙌 Samyuta Hasta — Double-Hand Mudras (22)

| # | Mudra Name |
|---|---|
| 29 | Anjali |
| 30 | Kapota |
| 31 | Karkata |
| 32 | Swastika |
| 33 | Pushpaputa |
| 34 | Shivalinga |
| 35 | Katakavardhana |
| 36 | Kartariswastika |
| 37 | Shakata |
| 38 | Shanku |
| 39 | Chakra |
| 40 | Samputa |
| 41 | Pasha |
| 42 | Kilaka |
| 43 | Matsya |
| 44 | Kurma |
| 45 | Varaha |
| 46 | Garuda |
| 47 | Nagabandha |
| 48 | Khatva |
| 49 | Bherunda |
| 50 | Gyan Mudra |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model Backbone** | EfficientNetV2-M (via `timm`) |
| **Deep Learning** | PyTorch |
| **Backend API** | FastAPI + Uvicorn |
| **Image Processing** | PIL / Pillow, torchvision |
| **Frontend** | Vanilla HTML + CSS + JavaScript |
| **Gesture Detection** | MediaPipe (browser-based) |
| **Particle Effects** | Custom canvas animation |

---

## 📊 Stats

| Metric | Value |
|---|---|
| Total Mudras Classified | **50** |
| Asamyuta Hasta (Single-Hand) | 28 |
| Samyuta Hasta (Double-Hand) | 22 |
| Dance Forms Covered | 8+ |
| Years of Heritage | 2000+ |
| Input Resolution | 320 × 320 px |
| Top-K Predictions | 5 |

---

## 🙏 Cultural Heritage

This project is dedicated to the preservation of **Bharatiya Natya Shastra** — the ancient Indian treatise on performing arts attributed to Sage Bharata Muni (200 BCE – 200 CE). Classical mudras are a precise language of the hands, expressing devotion, narrative, and cosmic symbolism across dance traditions including Bharatanatyam, Kathak, Odissi, Kuchipudi, and Mohiniyattam.

---

## 📄 License

This project is shared for educational and cultural preservation purposes.  
For commercial use, please contact the authors.

---

*मुद्रा पहचान प्रणाली — Mudra Recognition System*  
*🕉️*
