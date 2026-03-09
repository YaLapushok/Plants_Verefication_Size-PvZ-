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
    paths = [
        r"d:\papka\2\IT\Python\AI_plants\dataset\arugula\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\check_segmentation\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\YOLO_classification\val\arugula\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\check-twise\arugula_20260219162241005.jpg"
    ]
    
    img_path = None
    for p in paths:
        if os.path.exists(p):
            img_path = p
            break
            
    if img_path is None:
        print("Image not found")
        return
        
    img_pil = Image.open(img_path).convert('RGB')
    
    print("\n--- YOLO ---")
    results = arugula_seg_model(img_pil, verbose=False)
    masks_xy = results[0].masks.xy if results[0].masks is not None else []
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    boxes = results[0].boxes.xywh.cpu().numpy()
    
    classes_to_show = {YOLO_RU_MAP.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
    yolo_metrics, _ = calculate_yolo_metrics(masks_xy, classes, names, boxes, classes_to_show, YOLO_RU_MAP)
    print("YOLO Metrics:", yolo_metrics)
    print("YOLO Total Area:", sum(m['area'] for m in yolo_metrics.values()))

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
    
    from skimage import morphology
    
    for class_id in range(1, 4):
        mask_uint8 = (pred_orig == class_id).astype(np.uint8) * 255
        cls_name = UNET_CLASS_NAMES[class_id]
        
        mask_bool = mask_uint8 > 0
        if cls_name in ["корень", "стебель"]:
            mask_bool = morphology.skeletonize(mask_bool)
            
        pixel_count = int(mask_bool.astype(np.uint8).sum())
        
        area_mm2 = pixel_count * (BASE_SCALE_FACTOR ** 2)
        
        coords = np.argwhere(mask_bool)
        if len(coords) > 0:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h_px = float(r1 - r0)
            w_px = float(c1 - c0)
            length_mm = math.sqrt(w_px ** 2 + h_px ** 2) * BASE_SCALE_FACTOR
        else:
            length_mm = 0.0

        if pixel_count > 0:
            unet_metrics[cls_name] = {
                'count': 1,
                'area': area_mm2,
                'length': length_mm
            }
            
    print("UNET Metrics (Pixel width-corrected):", unet_metrics)
    print("UNET Total Area:", sum(m['area'] for m in unet_metrics.values()))

if __name__ == '__main__':
    test_models()
