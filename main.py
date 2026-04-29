
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mudravision")



MODEL_PATH   = "C:/Users/91906/OneDrive/Desktop/mudra/best_mudra_model.pth"
CLASSES_PATH = "C:/Users/91906/OneDrive/Desktop/mudra/mudra_names.txt"
IMAGE_SIZE   = 320          
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K        = 5


MUDRA_META = {
    "Pataka":           {"dev": "पताक",          "cat": "Asamyuta", "dance": "Bharatanatyam, Kathak, Odissi",      "meaning": "Flag, cloud, forest, blessing",         "usage": "Used in greetings and ceremonial blessings"},
    "Tripataka":        {"dev": "त्रिपताक",      "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Crown, tree, lightning bolt",            "usage": "Dramatic gesture in Nritta sequences"},
    "Ardhapataka":      {"dev": "अर्धपताक",      "cat": "Asamyuta", "dance": "Bharatanatyam, Odissi",              "meaning": "Knife, leaf, writing",                  "usage": "Depicts writing, blade, leaves"},
    "Kartarimukha":     {"dev": "कर्तरीमुख",     "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Scissors, separation, deer",            "usage": "Separation of lovers; deer eyes"},
    "Mayura":           {"dev": "मयूर",           "cat": "Asamyuta", "dance": "Bharatanatyam, Mohiniyattam",        "meaning": "Peacock, wiping tears",                 "usage": "Peacock; anointing; wiping tears"},
    "Ardhachandra":     {"dev": "अर्धचंद्र",     "cat": "Asamyuta", "dance": "Bharatanatyam, Kathak",              "meaning": "Half moon, waist belt, prayer",         "usage": "Moon; prayer; apprehension"},
    "Mushti":           {"dev": "मुष्टि",         "cat": "Asamyuta", "dance": "Bharatanatyam, Kuchipudi",           "meaning": "Holding sword, courage",                "usage": "Courage; holding objects; battle"},
    "Shikhara":         {"dev": "शिखर",           "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Tower, Shiva's bow, lips",              "usage": "God; lover; tower; bow"},
    "Alapadma":         {"dev": "अलपद्म",         "cat": "Asamyuta", "dance": "Bharatanatyam, Odissi, Kuchipudi",  "meaning": "Full-bloomed lotus, full moon, beauty", "usage": "Lotus bloom; moon; beauty"},
    "Anjali":           {"dev": "अञ्जलि",         "cat": "Samyuta",  "dance": "All classical dance forms",         "meaning": "Prayer, salutation, respect",           "usage": "Universal greeting; devotional prayer"},
    "Kapota":           {"dev": "कपोत",           "cat": "Samyuta",  "dance": "Bharatanatyam, Kathak",              "meaning": "Dove, modesty, submission",             "usage": "Humble salutation; dove"},
    "Pushpaputa":       {"dev": "पुष्पपुट",      "cat": "Samyuta",  "dance": "Bharatanatyam, Odissi",              "meaning": "Offering flowers, cupped vessel",       "usage": "Flower offering; water offering"},
    "Hamsasya":         {"dev": "हंसास्य",        "cat": "Asamyuta", "dance": "Bharatanatyam, Mohiniyattam",        "meaning": "Swan's beak, painting, delicate touch", "usage": "Swan; artistic painting; touching gently"},
    "Suchi":            {"dev": "सूचि",           "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Needle, one, sun, town",                "usage": "Number one; needle; city"},
    "Padmakosha":       {"dev": "पद्मकोश",        "cat": "Asamyuta", "dance": "Bharatanatyam, Odissi",              "meaning": "Lotus bud, apple, ball",                "usage": "Flowers; fruits"},
    "Katakamukha":      {"dev": "कटकमुख",         "cat": "Asamyuta", "dance": "Bharatanatyam, Kuchipudi",           "meaning": "Plucking flowers, pulling bow",         "usage": "Stringing garland; arrow; flowers"},
    "Shivalinga":       {"dev": "शिवलिंग",        "cat": "Nritta",   "dance": "Bharatanatyam",                      "meaning": "The form of Shiva",                     "usage": "Depicting Shiva; cosmic pillar"},
    "Hamsapaksha":      {"dev": "हंसपक्ष",        "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Swan's wing",                           "usage": "Flight of the swan"},
    "Sandamsha":        {"dev": "सन्दंश",         "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Grasping with pincers",                 "usage": "Holding, grasping, picking up"},
    "Mukula":           {"dev": "मुकुल",           "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Bud, closed flower",                    "usage": "Eating, flower bud, offering"},
    "Kangula":          {"dev": "काङ्गुल",         "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Small bell",                            "usage": "Bell ringing, small objects"},
    "Sarpashirsha":     {"dev": "सर्पशीर्ष",      "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Snake's head",                          "usage": "Snake; waves; sprinkling water"},
    "Mrigashirsha":     {"dev": "मृगशीर्ष",       "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Deer's head",                           "usage": "Deer; cheeks; woman"},
    "Simhamukha":       {"dev": "सिंहमुख",        "cat": "Asamyuta", "dance": "Bharatanatyam",                      "meaning": "Lion's face",                           "usage": "Herbs; lion; attack"},
    "Garuda":           {"dev": "गरुड",           "cat": "Samyuta",  "dance": "Bharatanatyam",                      "meaning": "Eagle, divine bird Garuda",             "usage": "Flight; Garuda vehicle of Vishnu"},
    "Chakra":           {"dev": "चक्र",           "cat": "Samyuta",  "dance": "Bharatanatyam",                      "meaning": "Wheel, disc",                           "usage": "Sudarshana Chakra; Vishnu"},
    "Pasha":            {"dev": "पाश",            "cat": "Samyuta",  "dance": "Bharatanatyam",                      "meaning": "Noose, binding rope",                   "usage": "Binding; Yama's noose"},
    "Kilaka":           {"dev": "कीलक",           "cat": "Samyuta",  "dance": "Bharatanatyam",                      "meaning": "Bond, link",                            "usage": "Love bond; friendship"},
}




class EnhancedMudraModel(nn.Module):
    """
    Mirrors the training architecture 1-to-1:
      backbone  : tf_efficientnetv2_m  (feature_dim = backbone.num_features = 1280)
      attention : Linear(1280,512) → ReLU → Linear(512,1280) → Sigmoid
      classifier: FC(1024)→BN→ReLU→Drop(0.5)
                  FC(512) →BN→ReLU→Drop(0.4)
                  FC(256) →BN→ReLU→Drop(0.3)
                  FC(num_classes)
    """
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Install timm:  pip install timm")

        self.backbone = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=pretrained,
            num_classes=0,          
        )
        feature_dim = self.backbone.num_features   

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),   
            nn.BatchNorm1d(1024),            
            nn.ReLU(inplace=True),           
            nn.Dropout(0.5),                 
            nn.Linear(1024, 512),            
            nn.BatchNorm1d(512),             
            nn.ReLU(inplace=True),           
            nn.Dropout(0.4),                 
            nn.Linear(512, 256),             
            nn.BatchNorm1d(256),             
            nn.ReLU(inplace=True),           
            nn.Dropout(0.3),                 
            nn.Linear(256, num_classes),     
        )

    def forward(self, x):
        features         = self.backbone(x)
        attention_weights = self.attention(features)
        attended         = features * attention_weights
        return self.classifier(attended)



class AppState:
    model: Optional[nn.Module] = None
    classes: list              = []
    ready: bool                = False

state = AppState()


preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])



def load_classes(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_model_from_path(model_path: str, classes_path: str):
    log.info(f"Loading classes from: {classes_path}")
    classes = load_classes(classes_path)
    log.info(f"Found {len(classes)} classes: {classes}")

    log.info(f"Loading checkpoint from: {model_path}")
    raw = torch.load(model_path, map_location=DEVICE, weights_only=False)

    
    if not isinstance(raw, dict):
        
        log.info("Checkpoint is a full model object — using directly")
        model = raw
        model.to(DEVICE).eval()
        return model, classes

    
    if "model_state_dict" in raw:
        sd = raw["model_state_dict"]
        
        if "class_names" in raw and not classes:
            classes = raw["class_names"]
    elif "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    
    final_key = "classifier.12.weight"
    if final_key not in sd:
        
        for k in sd:
            if k.startswith("classifier.") and k.endswith(".weight"):
                final_key = k   
        final_key = sorted(
            [k for k in sd if k.startswith("classifier.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1])
        )[-1]

    inferred_classes = sd[final_key].shape[0]
    log.info(f"Checkpoint has {inferred_classes} output classes")

    if inferred_classes != len(classes):
        log.warning(
            f"Checkpoint head={inferred_classes} but class file has {len(classes)}. "
            f"Padding/trimming class list to {inferred_classes}."
        )
        if inferred_classes > len(classes):
            classes += [f"class_{i}" for i in range(len(classes), inferred_classes)]
        else:
            classes = classes[:inferred_classes]

    
    model = EnhancedMudraModel(num_classes=inferred_classes, pretrained=False)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        log.info("Weights loaded perfectly ✅")

    model.to(DEVICE).eval()
    log.info("Model ready ✅")
    return model, classes



@asynccontextmanager
async def lifespan(app):
    if Path(MODEL_PATH).exists() and Path(CLASSES_PATH).exists():
        try:
            state.model, state.classes = load_model_from_path(MODEL_PATH, CLASSES_PATH)
            state.ready = True
        except ImportError as e:
            log.error(f"Missing dependency: {e}")
        except Exception as e:
            log.error(f"Startup model load failed: {e}", exc_info=True)
    else:
        log.warning(
            f"Model or classes file not found at startup.\n"
            f"  Model  : {MODEL_PATH}\n"
            f"  Classes: {CLASSES_PATH}\n"
            f"POST to /load-model to load manually."
        )
    yield



app = FastAPI(
    title="MudraVision AI API",
    description="Classify classical Indian dance mudras using a PyTorch EfficientNetV2-M model",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class Prediction(BaseModel):
    name: str
    confidence: float

class AnalyzeResponse(BaseModel):
    success: bool
    mudra: str
    devanagari: str
    confidence: float
    classification: str
    dance_forms: str
    meaning: str
    usage: str
    top_predictions: list[Prediction]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    classes: list[str]
    model_config = {"protected_namespaces": ()}

class LoadModelRequest(BaseModel):
    model_path: str
    classes_path: str
    model_config = {"protected_namespaces": ()}




@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    return HealthResponse(
        status="ready" if state.ready else "no_model",
        model_loaded=state.ready,
        device=DEVICE,
        num_classes=len(state.classes),
        classes=state.classes,
    )


@app.post("/load-model", tags=["Utility"])
def load_model_endpoint(req: LoadModelRequest):
    try:
        state.model, state.classes = load_model_from_path(req.model_path, req.classes_path)
        state.ready = True
        return {
            "success": True,
            "message": f"Model loaded with {len(state.classes)} classes",
            "classes": state.classes,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Model load error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze(file: UploadFile = File(...)):
    if not state.ready or state.model is None:
        raise HTTPException(status_code=503,
            detail="Model not loaded. POST to /load-model first or check server logs.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
            detail=f"Expected an image file, got: {file.content_type}")

    
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    
    try:
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pre-processing failed: {e}")

    
    try:
        with torch.no_grad():
            logits = state.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
    except Exception as e:
        log.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    
    k = min(TOP_K, len(state.classes))
    top_probs, top_idx = torch.topk(probs, k)

    top_predictions = [
        Prediction(
            name=state.classes[i.item()],
            confidence=round(p.item() * 100, 2),
        )
        for p, i in zip(top_probs, top_idx)
    ]

    best       = top_predictions[0]
    clean_name = best.name.split("(")[0].strip()
    meta       = MUDRA_META.get(clean_name, MUDRA_META.get(best.name, {}))

    log.info(f"Predicted: {best.name} ({best.confidence}%)")

    return AnalyzeResponse(
        success=True,
        mudra=clean_name,
        devanagari=meta.get("dev", ""),
        confidence=best.confidence,
        classification=meta.get("cat", "Unknown"),
        dance_forms=meta.get("dance", "—"),
        meaning=meta.get("meaning", "—"),
        usage=meta.get("usage", "—"),
        top_predictions=top_predictions[1:],   
    )


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)