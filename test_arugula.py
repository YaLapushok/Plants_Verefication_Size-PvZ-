import os
import sys
import math
import cv2
import numpy as np
from PIL import Image

project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

from core.models import arugula_seg_model, unet_arugula_model, UNET_TRANSFORM, DEVICE
from core.metrics import calculate_yolo_metrics
from core.constants import YOLO_RU_MAP, UNET_CLASS_NAMES, BASE_SCALE_FACTOR

def test_models():
    # specifically test on arugula_20260219162241005.jpg
    # try local check-twise directory, or check_segmentation
    paths = [
        r"d:\papka\2\IT\Python\AI_plants\check-twise\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\check_segmentation\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\YOLO_classification\val\arugula\arugula_20260219162241005.jpg"
    ]
    
    img_path = None
    for p in paths:
        if os.path.exists(p):
            img_path = p
            break
            
    if img_path is None:
        print("Arugula image not found!")
        # Fallback to simple walk
        for root, dirs, files in os.walk(r"d:\papka\2\IT\Python\AI_plants"):
            if "arugula_20260219162241005.jpg" in files:
                img_path = os.path.join(root, "arugula_20260219162241005.jpg")
                break
                
    if img_path is None:
        print("Still missing, aborting.")
        return

    print(f"Testing on {img_path}")
    img_pil = Image.open(img_path).convert('RGB')
    
    # YOLO
    print("\n--- YOLO ---")
    results = arugula_seg_model(img_pil, verbose=False)
    masks_xy = results[0].masks.xy if results[0].masks is not None else []
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    boxes = results[0].boxes.xywh.cpu().numpy()
    
    classes_to_show = {YOLO_RU_MAP.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
    
    yolo_metrics, _ = calculate_yolo_metrics(masks_xy, classes, names, boxes, classes_to_show, YOLO_RU_MAP)
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
        mask_uint8 = (pred_orig == class_id).astype(np.uint8) * 255
        
        # Test erosion
        if class_id == 1:
            erosion_kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask_uint8, erosion_kernel, iterations=1)
            contours_eroded, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            area_eroded = 0.0
            for cnt in contours_eroded:
                if len(cnt) >= 3:
                    area_eroded += cv2.contourArea(cnt) * (BASE_SCALE_FACTOR ** 2)
            print(f"Area with 5x5 erosion: {area_eroded}")

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
