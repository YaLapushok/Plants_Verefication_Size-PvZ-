import os
import sys

# Append project root to sys.path so we can import core
project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

import math
import cv2
import numpy as np
from PIL import Image

# Import models
from core.models import arugula_seg_model, unet_arugula_model, UNET_TRANSFORM, DEVICE
from core.metrics import calculate_yolo_metrics, calculate_unet_metrics
from core.constants import YOLO_RU_MAP, UNET_CLASS_NAMES, BASE_SCALE_FACTOR

def test_models():
    img_path = r"d:\papka\2\IT\Python\AI_plants\check-twise\arugula_20260219162241005.jpg"
    if not os.path.exists(img_path):
        # Fallback to a wheat image
        img_path = r"d:\papka\2\IT\Python\AI_plants\check_segmentation\wheat_20260219161730827.jpg"
        
    print(f"Testing on {img_path}")
    img_pil = Image.open(img_path).convert('RGB')
    
    print(f"Original image size: {img_pil.size}")

    # YOLO
    print("\n--- YOLO ---")
    results = arugula_seg_model(img_pil, verbose=False)
    masks_xy = results[0].masks.xy if results[0].masks is not None else []
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    boxes = results[0].boxes.xywh.cpu().numpy()
    
    yolo_ru_map = YOLO_RU_MAP
    classes_to_show = {yolo_ru_map.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
    
    yolo_metrics, _ = calculate_yolo_metrics(masks_xy, classes, names, boxes, classes_to_show, yolo_ru_map)
    print("YOLO Metrics:", yolo_metrics)
    yolo_total_area = sum(m['area'] for m in yolo_metrics.values())
    print("YOLO Total Area:", yolo_total_area)

    # UNET
    print("\n--- U-NET ---")
    import torch
    img_np = np.array(img_pil)
    orig_w, orig_h = img_pil.size
    aug = UNET_TRANSFORM(image=img_np)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = unet_arugula_model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_orig = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    unet_metrics = {}
    for class_id in range(1, 4):
        mask = (pred_orig == class_id).astype(np.uint8) * 255
        cls_name = UNET_CLASS_NAMES[class_id]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_mm2 = 0.0
        length_mm = 0.0
        count = 0
        for cnt in contours:
            if len(cnt) >= 3:
                area_mm2 += cv2.contourArea(cnt) * (BASE_SCALE_FACTOR ** 2)
            x, y, w, h = cv2.boundingRect(cnt)
            length_mm += math.sqrt(w ** 2 + h ** 2) * BASE_SCALE_FACTOR
            count += 1
            
        if count > 0:
            unet_metrics[cls_name] = {'count': count, 'area': area_mm2, 'length': length_mm}
            
    print("UNET Metrics (Contour based):", unet_metrics)
    unet_total_area = sum(m['area'] for m in unet_metrics.values())
    print("UNET Total Area:", unet_total_area)

if __name__ == '__main__':
    test_models()
