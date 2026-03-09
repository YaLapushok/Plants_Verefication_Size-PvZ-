import os
import sys

project_root = r"d:\papka\2\IT\Python\AI_plants\Plants_Verefication_Size-PvZ-"
sys.path.append(project_root)

from core.models import arugula_seg_model, unet_arugula_model
from PIL import Image

def test():
    paths = [
        r"d:\papka\2\IT\Python\AI_plants\dataset\arugula\arugula_20260219162241005.jpg",
        r"d:\papka\2\IT\Python\AI_plants\check-twise\arugula_20260219162241005.jpg"
    ]
    img_path = next((p for p in paths if os.path.exists(p)), None)
    img_pil = Image.open(img_path).convert('RGB')
    
    # Check YOLO Outputs
    res = arugula_seg_model(img_pil, verbose=False)[0]
    names = res.names
    classes = res.boxes.cls.cpu().numpy()
    
    boxes = res.boxes.xywh.cpu().numpy()
    
    print("YOLO Detected:")
    for i, c in enumerate(classes):
        w, h = boxes[i][2], boxes[i][3]
        area = w * h
        print(f" Box {i}: cls={int(c)} -> name={names[int(c)]}, W={w:.1f}, H={h:.1f}, approx_area_px={area:.1f}")

test()
