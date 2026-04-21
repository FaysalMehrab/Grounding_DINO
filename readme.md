# GroundingDINO Image Annotation

Runs GroundingDINO on a single image and saves the annotated output. Supports both CPU-only and CUDA-accelerated environments.

---

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.10+ |
| OS | Ubuntu / Linux (recommended) |
| CUDA | Optional (CPU-only mode supported) |

---

## Installation

### ⚠️ Important Notes

- Follow **all steps strictly** in order. Skipping steps may cause:
  ```
  NameError: name '_C' is not defined
  ```
  If this error occurs, re-clone the repository and repeat the entire setup from scratch.

- GroundingDINO compiles under **CPU-only mode** if no CUDA is detected. If you have a GPU, configure `CUDA_HOME` **before** installing (see [CUDA Setup](#cuda-setup-optional) below).

---

### 1. Clone the Repository

```bash
git clone https://github.com/FaysalMehrab/Grounding_DINO.git
```

---

### 2. CUDA Setup (Optional)

Skip this section if you are running CPU-only.

**Check if `CUDA_HOME` is set:**
```bash
echo $CUDA_HOME
```
If nothing is printed, `CUDA_HOME` is not configured. Set it before proceeding.

**Find your CUDA path:**
```bash
which nvcc
# Example output: /usr/local/cuda/bin/nvcc
# → Your CUDA_HOME is: /usr/local/cuda
```

**Set `CUDA_HOME` for the current session:**
```bash
export CUDA_HOME=/usr/local/cuda   # Replace with your actual path
```
> Make sure the CUDA version matches your CUDA runtime. If multiple CUDA versions are installed, verify with `nvcc --version`.

**Set `CUDA_HOME` permanently:**
```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
echo $CUDA_HOME   # Confirm it prints correctly
```

---

### 3. Create a Virtual Environment

```bash
python3 -m venv .grounding_dino_env
source .grounding_dino_env/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install opencv-python
pip install transformers==4.36.2
pip install -e . --no-build-isolation
```

---

### 5. Download Pre-trained Weights

Choose one of the following model variants:

**Option A — Swin-T** *(lightweight, faster)*
```bash
mkdir -p weights && cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

**Option B — Swin-B** *(recommended for higher accuracy)*
```bash
mkdir -p weights && cd weights
wget -q https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
cd ..
```

---

## Running Inference

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

DEVICE = "cpu"  # Change to "cuda" if GPU is available

# Load model — choose one:
# Swin-T (lightweight)
# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth", device=DEVICE)

# Swin-B (recommended accuracy)
model = load_model(
    "groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "weights/groundingdino_swinb_cogcoor.pth",
    device=DEVICE
)

IMAGE_PATH = "assets/nid_45.png"
TEXT_PROMPT = "head . wing . fuselage . engine . tail"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

cv2.imwrite("annotated_image_swinb.jpg", annotated_frame)
print("Saved: annotated_image_swinb.jpg")
```

---

## Output

The annotated image is saved to:
```
annotated_image_swinb.jpg
```
# Grounding DINO via Autodistill (Optional Alternative)

This directory contains an implementation of **Grounding DINO** using the **Autodistill** framework.  
While a raw implementation of Grounding DINO is available, this version is provided as an optional alternative for users who prefer a high-level API integrated with the **Supervision** ecosystem.

---

## 🚀 Why Use This Version?

The Autodistill wrapper simplifies the *Teacher–Student* workflow and provides several quality-of-life improvements over the raw repository:

- **Simplified Ontology**  
  Easily map complex text prompts to clean class names using the `CaptionOntology` dictionary.

- **Supervision Integration**  
  Native support for `supervision (sv)` objects, enabling easier NMS, filtering, and professional-grade annotations.

- **Automatic Weight Handling**  
  Automatically manages model configuration and weight downloading — no manual path setup required.

- **Hallucination Filtering**  
  Built-in logic to filter out oversized detections ("giant boxes") based on image area ratios.

---

## 📦 Installation

Install the required packages:

```bash
pip install autodistill-grounding-dino supervision
```

---

## ⚡ Quick Start

The core logic revolves around the `CaptionOntology`, which defines what to detect and how to label it.

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# Define detection ontology
ontology = CaptionOntology({
    "the pointed front section of the airplane": "head",
    "airplane engine": "engine"
})

# Initialize model
model = GroundingDINO(ontology=ontology)

# Run inference
results = model.predict("image.jpg")
```

---

## 🔑 Key Features in This Implementation

- **2-Phase Detection (Animals)**  
  Prevents class confusion (e.g., detecting a "beak" on a dog) by:
  1. Classifying the animal  
  2. Applying a specific part-based ontology

- **Class-Aware NMS**  
  Ensures overlapping detections from different classes (e.g., eye inside head) are not suppressed.

- **Area-Based Filtering**  
  Removes detections exceeding a defined percentage of the image size to reduce false positives.

---

## ⚖️ When to Use This vs. Raw Grounding DINO?

| Feature         | Raw Implementation         | Autodistill Version        |
|----------------|---------------------------|----------------------------|
| **Control**     | Maximum (low-level access) | High (standardized API)    |
| **Setup**       | Manual config & weights    | Automatic                  |
| **Visualization** | Manual OpenCV             | Supervision annotators     |
| **Speed**       | Slightly faster runtime    | Better for batch labeling  |

---

## 📝 Note

This implementation is **optional**.

If your research or production pipeline requires **low-level control over the Grounding DINO architecture**, refer to the raw implementation available in the adjacent directory.

---

## 🙏 Acknowledgements

This project is built on top of the following open-source work:

- GroundingDINO by IDEA-Research  
  https://github.com/IDEA-Research/GroundingDINO
  
- Original paper:  
  Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection  
  https://arxiv.org/abs/2303.05499
  
Special thanks to the authors for releasing the code and pretrained models.
