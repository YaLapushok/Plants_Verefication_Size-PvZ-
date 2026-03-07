import math
import numpy as np
import cv2
from core.constants import BASE_SCALE_FACTOR

def calculate_yolo_metrics(masks, classes, names, boxes, classes_to_show, yolo_ru_map):
    class_metrics = {}
    valid_indices = []
    
    for i in range(len(masks)):
        cls_name_ru = yolo_ru_map.get(names[int(classes[i])].lower(), names[int(classes[i])].lower())
        if cls_name_ru.lower() in [c.lower() for c in classes_to_show]:
            valid_indices.append(i)
            mask = masks[i]
            _, _, w, h = boxes[i]

            valid_pixels = np.count_nonzero(mask)
            area_mm2 = valid_pixels * (BASE_SCALE_FACTOR ** 2)
            length_mm = math.sqrt(w ** 2 + h ** 2) * BASE_SCALE_FACTOR

            if cls_name_ru not in class_metrics:
                class_metrics[cls_name_ru] = {'count': 0, 'area': 0.0, 'length': 0.0}
            class_metrics[cls_name_ru]['count'] += 1
            class_metrics[cls_name_ru]['area'] += area_mm2
            class_metrics[cls_name_ru]['length'] += length_mm
            
    return class_metrics, valid_indices

def calculate_unet_metrics(pred_orig, unet_class_names, selected_classes):
    class_metrics = {}
    available_classes = []

    for class_id in range(1, 4):  # skip background
        cls_name = unet_class_names[class_id]
        mask = (pred_orig == class_id)
        
        # --- ROOT CLEANING ---
        # If it's the root class (usually id 1), apply erosion to remove "free space" or noise
        if class_id == 1:  # корень
            mask_uint8 = mask.astype(np.uint8) * 255
            # 3x3 kernel for subtle cleaning
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
            mask = mask_uint8 > 0
        # ---------------------

        pixel_count = int(mask.sum())

        if pixel_count == 0:
            continue

        if selected_classes is not None:
            if cls_name.lower() not in [c.lower() for c in selected_classes]:
                continue

        available_classes.append(cls_name)
        area_mm2 = pixel_count * (BASE_SCALE_FACTOR ** 2)

        # Count components
        mask_uint8 = mask.astype(np.uint8) * 255
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        object_count = num_labels - 1
        
        coords = np.argwhere(mask)
        if len(coords) > 0:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h_px = float(r1 - r0)
            w_px = float(c1 - c0)
            length_mm = math.sqrt(w_px ** 2 + h_px ** 2) * BASE_SCALE_FACTOR
        else:
            length_mm = 0.0

        class_metrics[cls_name] = {
            'count': object_count,
            'area': area_mm2,
            'length': length_mm
        }
    return class_metrics, available_classes
