import os
import sys
import numpy as np
import cv2
from PIL import Image

project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

from core.models import arugula_seg_model
from core.constants import BASE_SCALE_FACTOR, YOLO_RU_MAP

def test():
    img_path = r"d:\papka\2\IT\Python\AI_plants\dataset\arugula\arugula_20260219162241005.jpg"
    image_pil = Image.open(img_path).convert('RGB')
    
    seg_results = arugula_seg_model(image_pil, verbose=False)
    masks = seg_results[0].masks.data.cpu().numpy()
    masks_xy = seg_results[0].masks.xy
    classes = seg_results[0].boxes.cls.cpu().numpy()
    names = seg_results[0].names
    
    print("YOLO Raw Mask Areas:")
    class_areas = {}
    for i, xy in enumerate(masks_xy):
        cls_name = names[int(classes[i])]
        ru_name = YOLO_RU_MAP.get(cls_name.lower(), cls_name)
        area_px = cv2.contourArea(xy) if len(xy) >= 3 else 0.0
        area_mm2 = area_px * (BASE_SCALE_FACTOR ** 2)
        
        if ru_name not in class_areas:
            class_areas[ru_name] = 0.0
        class_areas[ru_name] += area_mm2
        
    for k, v in class_areas.items():
        print(f" {k}: {v:.1f} mm2")

test()
