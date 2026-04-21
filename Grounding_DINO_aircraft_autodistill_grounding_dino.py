import os
import cv2
import json
import numpy as np
import supervision as sv
from pathlib import Path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

INPUT_DIR = Path("Aircraft") 
OUTPUT_DIR = Path("aircraft_results_autodistill")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_PATH = OUTPUT_DIR / "grounding_dino_aircraft.json"

CONFIDENCE_THRESHOLD = 0.25
MAX_BOX_RATIO = 0.80 

ontology = CaptionOntology({
    "the pointed front section of the airplane featuring the nose and cockpit windows": "head",
    "airplane main/front wing": "swept wing",
    "airplane engine": "engine",
    "airplane vertical stabilizer tail fin": "tail",
    "airplane horizontal stabilizer": "tail wing"
})

model = GroundingDINO(ontology=ontology, box_threshold=0.20, text_threshold=0.15)
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
    
    results = model.predict(str(img_path))
    results = class_aware_nms(results, ontology, {"head": 0.3, "swept wing": 0.25, "tail wing": 0.4, "tail": 0.3, "engine": 0.4})

    mask = [results.confidence[i] >= CONFIDENCE_THRESHOLD and 
            (((results.xyxy[i][2]-results.xyxy[i][0])*(results.xyxy[i][3]-results.xyxy[i][1]))/(img_w*img_h)) <= MAX_BOX_RATIO 
            for i in range(len(results))]
    results = results[np.array(mask)]

    img_detections = []
    components = {c: 0 for c in ontology.classes()}
    confidences = []

    if len(results) > 0:
        for i in range(len(results)):
            c_name = ontology.classes()[int(results.class_id[i])]
            conf = float(results.confidence[i])
            x1, y1, x2, y2 = results.xyxy[i]
            components[c_name] += 1
            confidences.append(conf)
            img_detections.append({"class": c_name, "confidence": round(conf, 4), "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]})

        labels = [f"{c_name} {conf:.2f}" for c_name, conf in zip([ontology.classes()[cid] for cid in results.class_id], results.confidence)]
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=results)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=results, labels=labels)
        cv2.imwrite(str(OUTPUT_DIR / img_name), annotated_image)
    else:
        cv2.imwrite(str(OUTPUT_DIR / img_name), image)

    results_list.append({"image_id": img_name, "components": components, "engine_count": components.get("engine", 0), "average_confidence": round(float(np.mean(confidences)), 2) if confidences else 0.0, "total_detections": len(img_detections), "detections": img_detections})

final_output = {"model": "GroundingDINO", "confidence_threshold": CONFIDENCE_THRESHOLD, "total_images": len(image_files), "total_detections": sum(r["total_detections"] for r in results_list), "results": results_list}
with open(JSON_OUTPUT_PATH, "w") as f: json.dump(final_output, f, indent=2)
