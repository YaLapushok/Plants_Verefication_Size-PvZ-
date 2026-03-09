import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

from core.models import unet_arugula_model, DEVICE, UNET_TRANSFORM
from core.constants import BASE_SCALE_FACTOR

def test():
    paths = [
        r"d:\papka\2\IT\Python\AI_plants\dataset\arugula\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\check-twise\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\YOLO_classification\val\arugula\arugula_20260219162241005.jpg",
    ]
    img_path = next((p for p in paths if os.path.exists(p)), None)
    
    image_pil = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image_pil.size
    img_np = np.array(image_pil)
    
    aug = UNET_TRANSFORM(image=img_np)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = unet_arugula_model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    pred_orig = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    print("UNET Raw Data:")
    for i in range(1, 4):
        mask = (pred_orig == i)
        px = mask.sum()
        area = px * (BASE_SCALE_FACTOR**2)
        print(f" ID {i}: {px} pixels -> {area:.1f} mm2")

test()
