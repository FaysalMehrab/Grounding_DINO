import os
import cv2
import json
import numpy as np
import supervision as sv
from pathlib import Path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# --- Paths & Config ---
INPUT_DIR = Path("Animal") 
OUTPUT_DIR = Path("animal_results_autodistill")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_PATH = OUTPUT_DIR / "grounding_dino_animal.json"

CONFIDENCE_THRESHOLD = 0.25
MAX_BOX_RATIO = 0.80 

# --- Phase 1: Classification Ontology ---
cls_ontology = CaptionOntology({
    "dog": "dog", "elephant": "elephant", "chicken": "chicken", 
    "cat": "cat", "cow": "cow", "sheep": "sheep", "squirrel": "squirrel"
})

# --- Phase 2: Part Ontologies ---
mammal_ontology = CaptionOntology({
    "animal head": "head", "animal eye": "eye", 
    "animal leg": "leg", "animal tail": "tail", "animal torso": "torso"
})

bird_ontology = CaptionOntology({
    "animal head": "head", "animal eye": "eye", "bird beak": "beak", 
    "bird wings": "wing", "animal leg": "leg", "animal tail": "tail", "animal torso": "torso"
})

# Initialize Model
model = GroundingDINO(ontology=cls_ontology) # Start with classification
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def class_aware_nms(detections, current_ontology, thresholds: dict):
    if len(detections) == 0: return detections
    kept_indices = []
    classes = current_ontology.classes()
    for class_name, iou_thresh in thresholds.items():
        if class_name not in classes: continue
        class_idx = classes.index(class_name)
        mask = detections.class_id == class_idx
        indices = np.where(mask)[0]
        if len(indices) == 0: continue
        class_det = detections[indices].with_nms(threshold=iou_thresh)
        for box in class_det.xyxy:
            for j in indices:
                if np.allclose(detections.xyxy[j], box, atol=1.0):
                    kept_indices.append(j)
                    break
    return detections[np.array(sorted(set(kept_indices)))] if kept_indices else detections

image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
results_list = []

for img_name in image_files:
    img_path = INPUT_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None: continue
    img_h, img_w = image.shape[:2]
    
    # --- PHASE 1: CLASSIFY ---
    model.ontology = cls_ontology
    cls_results = model.predict(str(img_path))
    
    is_bird = False
    if len(cls_results) > 0:
        best_cls_idx = np.argmax(cls_results.confidence)
        class_name = cls_ontology.classes()[cls_results.class_id[best_cls_idx]]
        if class_name == "chicken":
            is_bird = True

    # --- PHASE 2: DETECT PARTS ---
    current_ontology = bird_ontology if is_bird else mammal_ontology
    model.ontology = current_ontology
    results = model.predict(str(img_path))
    
    # NMS
    results = class_aware_nms(results, current_ontology, {
        "head": 0.3, "eye": 0.5, "leg": 0.5, "tail": 0.3, "torso": 0.3, "beak": 0.3, "wing": 0.4
    })

    # Filter
    mask = [results.confidence[i] >= CONFIDENCE_THRESHOLD and 
            (((results.xyxy[i][2]-results.xyxy[i][0])*(results.xyxy[i][3]-results.xyxy[i][1]))/(img_w*img_h)) <= MAX_BOX_RATIO 
            for i in range(len(results))]
    results = results[np.array(mask)]

    img_detections = []
    # Ensure all 7 keys exist in components even if 0
    components = {k: 0 for k in ["head", "eye", "leg", "tail", "torso", "beak", "wing"]}
    confidences = []

    if len(results) > 0:
        for i in range(len(results)):
            c_name = current_ontology.classes()[int(results.class_id[i])]
            conf = float(results.confidence[i])
            x1, y1, x2, y2 = results.xyxy[i]
            
            components[c_name] += 1
            confidences.append(conf)
            img_detections.append({
                "class": c_name, "confidence": round(conf, 4),
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
            })

        labels = [f"{current_ontology.classes()[class_id]} {conf:.2f}" for class_id, conf in zip(results.class_id, results.confidence)]
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=results)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=results, labels=labels)
        cv2.imwrite(str(OUTPUT_DIR / img_name), annotated_image)
    else:
        cv2.imwrite(str(OUTPUT_DIR / img_name), image)

    results_list.append({
        "image_id": img_name, "components": components, "leg_count": components.get("leg", 0),
        "average_confidence": round(float(np.mean(confidences)), 2) if confidences else 0.0,
        "total_detections": len(img_detections), "detections": img_detections
    })

final_output = {
    "model": "GroundingDINO", "confidence_threshold": CONFIDENCE_THRESHOLD,
    "total_images": len(image_files), "total_detections": sum(r["total_detections"] for r in results_list),
    "results": results_list
}
with open(JSON_OUTPUT_PATH, "w") as f: json.dump(final_output, f, indent=2)
