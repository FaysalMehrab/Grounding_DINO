````md
# GroundingDINO Image Annotation

This project runs GroundingDINO on a single image and saves the annotated output.

## Requirements

- Python 3.10 or 3.12
- CPU is supported
- GroundingDINO installed in editable mode
- PyTorch, OpenCV, Transformers

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .grounding_dino_env
source .grounding_dino_env/bin/activate
````

Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install opencv-python
pip install transformers==4.36.2
pip install -e . --no-build-isolation
```

## Files

* `groundingdino/config/GroundingDINO_SwinB_cfg.py`
* `weights/groundingdino_swinb_cogcoor.pth`
* `assets/nid_45.png`

## Run

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

DEVICE = "cpu"

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

## Output

The annotated image will be saved as:

```bash
annotated_image_swinb.jpg
```
