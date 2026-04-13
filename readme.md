# GroundingDINO Image Annotation (CPU Setup)

This project runs GroundingDINO on a single image and saves the annotated output.

---

## Requirements
- Python 3.10+ (CPU supported)
- Ubuntu/Linux recommended
- GroundingDINO repo cloned

---

## Setup

### 1. Create virtual environment
```bash
python3 -m venv .grounding_dino_env
source .grounding_dino_env/bin/activate
````

### 2. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install opencv-python
pip install transformers==4.36.2
pip install -e . --no-build-isolation
```

---

## Download Pre-trained Weights

### Option 1: Swin-T (lightweight)

```bash
mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

### Option 2: Swin-B (recommended accuracy)

```bash
mkdir -p weights
cd weights
wget -q https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
cd ..
```

---

## Run Inference

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

DEVICE = "cpu"

# For Swin-T
# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device=DEVICE)


# For Swin-B
model = load_model(
    "groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "weights/groundingdino_swinb_cogcoor.pth",
    device=DEVICE
)

IMAGE_PATH = "assets/nid_45.png"
TEXT_PROMPT = "head . wing . fuselage . engine . tail"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE
)

annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

cv2.imwrite("annotated_image_swinb.jpg", annotated_frame)
print("Saved successfully")
```

---

## Output

```bash
annotated_image_swinb.jpg
```
