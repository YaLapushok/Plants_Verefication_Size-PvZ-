import io
import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw
from core.models import (
    classify_image, 
    arugula_seg_model, 
    wheat_seg_model, 
    unet_arugula_model, 
    unet_wheat_model, 
    UNET_AVAILABLE, 
    DEVICE, 
    UNET_TRANSFORM
)
from core.utils import plot_yolo_custom, build_charts, get_font
from core.metrics import calculate_yolo_metrics, calculate_unet_metrics
from core.constants import UNET_CLASS_COLORS, UNET_CLASS_NAMES, YOLO_RU_MAP, COLOR_MAP

def _process_yolo(image_pil, is_arugula: bool, show_boxes: bool, selected_classes: list | None) -> dict:
    seg_model = arugula_seg_model if is_arugula else wheat_seg_model
    if seg_model is None:
        return {'error': 'Segmentation model not loaded.'}

    seg_results = seg_model(image_pil, verbose=False)
    if not seg_results or len(seg_results) == 0 or seg_results[0].masks is None:
        return {'error': 'No segmentation results.'}

    masks = seg_results[0].masks.data.cpu().numpy() if seg_results[0].masks is not None else []
    masks_xy = seg_results[0].masks.xy if seg_results[0].masks is not None else []
    classes = seg_results[0].boxes.cls.cpu().numpy()
    names = seg_results[0].names
    boxes = seg_results[0].boxes.xywh.cpu().numpy()

    unique_classes = {YOLO_RU_MAP.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
    available_classes = list(unique_classes)
    
    classes_to_show = unique_classes if selected_classes is None else set(selected_classes)
    class_metrics, valid_indices = calculate_yolo_metrics(masks_xy, classes, names, boxes, classes_to_show, YOLO_RU_MAP)

    total_length = sum(d['length'] for d in class_metrics.values())
    total_area = sum(d['area'] for d in class_metrics.values())

    import copy
    fr = copy.deepcopy(seg_results[0])
    if len(valid_indices) != len(masks):
        fr.masks.data = fr.masks.data[valid_indices] if len(valid_indices) > 0 else []
        fr.boxes.data = fr.boxes.data[valid_indices] if len(valid_indices) > 0 else []
        if hasattr(fr.masks, 'xy'):
            fr.masks.xy = [fr.masks.xy[idx] for idx in valid_indices] if valid_indices else []
        if hasattr(fr.masks, 'xyn'):
            fr.masks.xyn = [fr.masks.xyn[idx] for idx in valid_indices] if valid_indices else []
    
    annotated_img = plot_yolo_custom(fr, show_boxes=show_boxes)
    
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)).save(buf, format='JPEG', quality=95)
    
    area_bytes, bar_bytes = build_charts(class_metrics)

    returned_class_colors = {ru_name: COLOR_MAP.get(ru_name.lower(), '#808080') for ru_name in available_classes}

    return {
        'class_metrics': class_metrics,
        'class_colors': returned_class_colors,
        'available_classes': available_classes,
        'annotated_image_bytes': buf.getvalue(),
        'chart_bytes': area_bytes,
        'bar_chart_bytes': bar_bytes,
        'total_area': total_area,
        'total_length': total_length,
        'error': None,
    }

def _process_unet(image_pil, is_arugula: bool, selected_classes: list | None = None, show_boxes: bool = True) -> dict:
    if not UNET_AVAILABLE:
        return {'error': 'U-Net dependencies not available.'}

    import torch
    unet_model = unet_arugula_model if is_arugula else unet_wheat_model
    if unet_model is None:
        return {'error': f'U-Net model not loaded.'}

    orig_w, orig_h = image_pil.size
    img_np = np.array(image_pil)

    aug = UNET_TRANSFORM(image=img_np)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = unet_model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_orig = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    overlay_np = img_np.copy()
    for class_id, color in UNET_CLASS_COLORS.items():
        if class_id == 0: continue
        cls_name = UNET_CLASS_NAMES[class_id]
        if selected_classes is not None and cls_name.lower() not in [c.lower() for c in selected_classes]: continue
        
        mask = (pred_orig == class_id)
        if not mask.any(): continue
        
        overlay_np[mask] = (overlay_np[mask] * 0.4 + np.array(color, dtype=np.float32) * 0.6).astype(np.uint8)

    final_pil = Image.fromarray(overlay_np)
    if show_boxes:
        draw = ImageDraw.Draw(final_pil)
        font = get_font(16)
        for class_id, color in UNET_CLASS_COLORS.items():
            if class_id == 0: continue
            cls_name = UNET_CLASS_NAMES[class_id]
            if selected_classes is not None and cls_name.lower() not in [c.lower() for c in selected_classes]: continue
            mask = (pred_orig == class_id)
            if not mask.any(): continue
            
            contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= 5 and h >= 5:
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                    label = cls_name.capitalize()
                    bbox = draw.textbbox((x, y), label, font=font)
                    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    draw.rectangle([x, y-th-4, x+tw+4, y], fill=color)
                    draw.text((x+2, y-th-2), label, fill=(255,255,255), font=font)

    buf = io.BytesIO()
    final_pil.save(buf, format='JPEG', quality=95)
    
    class_metrics, available_classes = calculate_unet_metrics(pred_orig, UNET_CLASS_NAMES, selected_classes)
    total_length = sum(d['length'] for d in class_metrics.values())
    total_area = sum(d['area'] for d in class_metrics.values())

    area_bytes, bar_bytes = build_charts(class_metrics)
    returned_class_colors = {name: COLOR_MAP.get(name.lower(), '#808080') for name in available_classes}

    return {
        'class_metrics': class_metrics,
        'class_colors': returned_class_colors,
        'available_classes': available_classes,
        'annotated_image_bytes': buf.getvalue(),
        'chart_bytes': area_bytes,
        'bar_chart_bytes': bar_bytes,
        'total_area': total_area,
        'total_length': total_length,
        'error': None,
    }

def process_image(image_bytes: bytes, show_boxes: bool = True, selected_classes: list = None, seg_model: str = 'yolo') -> dict:
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        is_arugula = classify_image(image_pil)
        plant_name_ru = "Рукола" if is_arugula else "Пшеница"

        if seg_model == 'unet':
            result = _process_unet(image_pil, is_arugula, selected_classes, show_boxes)
        else:
            result = _process_yolo(image_pil, is_arugula, show_boxes, selected_classes)

        if result.get('error'):
            return result

        result['class_name'] = plant_name_ru
        result['original_image_bytes'] = image_bytes
        return result

    except Exception as e:
        logging.error(f"Error in process_image: {e}", exc_info=True)
        return {'error': str(e)}
