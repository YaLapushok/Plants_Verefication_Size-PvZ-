import os
import sys

# Append project root to sys.path so we can import core
project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

import cv2
import numpy as np
import io
from PIL import Image

# Import models
from core.models import arugula_seg_model, unet_arugula_model
from core.metrics import calculate_yolo_metrics, calculate_unet_metrics
from core.constants import YOLO_RU_MAP, UNET_CLASS_NAMES

def test_models():
    # Load an image (try to find one in the folder or just create a blank one to see shapes)
    image_files = [f for f in os.listdir('./app') if f.endswith('.jpg') or f.endswith('.png')]
    # Try any image
    if not image_files:
        image_np = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img_pil = Image.fromarray(image_np)
    else:
        img_pil = Image.open(os.path.join('./app', image_files[0]))

    print(f"Original image size: {img_pil.size}")

    # YOLO
    print("\n--- YOLO ---")
    results = arugula_seg_model(img_pil, verbose=False)
    masks_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else np.array([])
    
    print(f"YOLO masks.data shape: {masks_data.shape}")
    if len(masks_data) > 0:
        for i, mask in enumerate(masks_data):
            valid_pixels = np.count_nonzero(mask)
            print(f"  Mask {i}: {valid_pixels} pixels in masks.data")
            if hasattr(results[0].masks, 'xy'):
                xy = results[0].masks.xy[i]
                area_xy = cv2.contourArea(xy)
                print(f"  Mask {i}: {area_xy} pixels using cv2.contourArea(xy)")

if __name__ == '__main__':
    test_models()
