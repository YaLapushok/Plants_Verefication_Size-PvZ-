import io
import os
import cv2
import numpy as np
import logging
import math
from PIL import Image
from ultralytics import YOLO
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# Define base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ─────────────────────────── YOLO MODELS ────────────────────────────
try:
    logging.info("Loading YOLO models...")
    classificator_model = YOLO(os.path.join(MODELS_DIR, 'classificator.pt'))
    arugula_seg_model   = YOLO(os.path.join(MODELS_DIR, 'arugula.pt'))
    wheat_seg_model     = YOLO(os.path.join(MODELS_DIR, 'wheat.pt'))
    logging.info("YOLO models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO models: {e}")
    classificator_model = None
    arugula_seg_model   = None
    wheat_seg_model     = None

# ─────────────────────────── U-NET MODELS ─────────────────────────────
try:
    import torch
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device for U-Net: {DEVICE}")

    # U-Net model architecture used during training
    def _build_unet():
        return smp.Unet(
            encoder_name='resnet50',
            encoder_weights=None,      # weights loaded from .pth
            in_channels=3,
            classes=4,
            activation=None,
        )

    _UNET_ARUGULA_PATH = os.path.join(MODELS_DIR, 'U-Net', 'rugola_v3_best.pth')
    _UNET_WHEAT_PATH   = os.path.join(MODELS_DIR, 'U-Net', 'пшеница_4класса.pth')

    def _load_unet(path):
        if not os.path.exists(path):
            logging.warning(f"U-Net weights not found: {path}")
            return None
        model = _build_unet()
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        model.to(DEVICE)
        logging.info(f"U-Net loaded: {path}")
        return model

    unet_arugula_model = _load_unet(_UNET_ARUGULA_PATH)
    unet_wheat_model   = _load_unet(_UNET_WHEAT_PATH)

    # Validation transform (same as training)
    _UNET_TRANSFORM = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    UNET_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to load U-Net models: {e}")
    unet_arugula_model = None
    unet_wheat_model   = None
    UNET_AVAILABLE     = False


CALIB_WIDTH      = 2048
BASE_SCALE_FACTOR = 0.10556744  # mm/pixel

# Class colour map for U-Net overlay
# 0=background, 1=root, 2=stem, 3=leaf
UNET_CLASS_COLORS = {
    0: (0,   0,   0),      # black  – background (transparent in overlay)
    1: (255, 140, 0),      # orange – root (корень)
    2: (50,  205, 50),     # green  – stem (стебель)
    3: (0,   120, 255),    # blue   – leaf (листок)
}
# Arugula U-Net classes (same indices as wheat for the unified architecture)
UNET_CLASS_NAMES = {
    0: 'background',
    1: 'корень',
    2: 'стебель',
    3: 'листок',
}

# Mapping English YOLO names to Russian
YOLO_RU_MAP = {
    'koren':  'корень',
    'stebel': 'стебель',
    'listok': 'листок',
    'kolos':  'колос',
    'root':   'корень',
    'stem':   'стебель',
    'leaf':   'листок'
}

# ─────────────────────────── HELPERS ──────────────────────────────────

def _classify_image(image_pil) -> bool:
    """Return True if the image is arugula, False for wheat."""
    if classificator_model is None:
        return False
    cls_results = classificator_model(image_pil, verbose=False)
    best_class_id   = cls_results[0].probs.top1
    best_class_name = cls_results[0].names[best_class_id].lower()

    if 'arugula' in best_class_name or 'рукола' in best_class_name or 'rucola' in best_class_name:
        return True
    if best_class_id == 1 and 'wheat' not in best_class_name and 'пшеница' not in best_class_name:
        return True
    return False


def _build_charts(class_metrics: dict) -> tuple[bytes | None, bytes | None]:
    """Build pie chart (area) and bar chart (length). Returns (pie_bytes, bar_bytes)."""
    if not class_metrics:
        return None, None

    labels = [n.capitalize() for n in class_metrics.keys()]
    areas  = [d['area']   for d in class_metrics.values()]
    lengths= [d['length'] for d in class_metrics.values()]
    colors = ['#ff9f40', '#4bc0c0', '#9966ff', '#ff6384', '#36a2eb', '#ffcd56'][:len(labels)]

    area_bytes = None
    if any(a > 0 for a in areas):
        fig, ax = plt.subplots(figsize=(5, max(2.5, len(labels) * 0.8)))
        y_pos = range(len(labels))
        bars = ax.barh(list(y_pos), areas, color=colors, height=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color='#f0fdf4', fontsize=9)
        ax.set_xlabel('Площадь (мм²)', color='#f0fdf4', fontsize=9)
        ax.set_title('Площадь по частям (мм²)', color='#f0fdf4', fontsize=11)
        ax.tick_params(axis='x', colors='#f0fdf4')
        ax.spines['bottom'].set_color('#2d5a3d')
        ax.spines['left'].set_color('#2d5a3d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, areas):
            ax.text(bar.get_width() + max(areas) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', color='#f0fdf4', fontsize=8)
        fig.patch.set_facecolor('#0a1f14')
        ax.set_facecolor('#0a1f14')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', facecolor='#0a1f14')
        buf.seek(0)
        area_bytes = buf.getvalue()
        plt.close(fig)

    bar_bytes = None
    if any(l > 0 for l in lengths):
        fig, ax = plt.subplots(figsize=(5, max(2.5, len(labels) * 0.8)))
        y_pos = range(len(labels))
        bars = ax.barh(list(y_pos), lengths, color=colors, height=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color='#f0fdf4', fontsize=9)
        ax.set_xlabel('Длина (мм)', color='#f0fdf4', fontsize=9)
        ax.set_title('Длина по частям (мм)', color='#f0fdf4', fontsize=11)
        ax.tick_params(axis='x', colors='#f0fdf4')
        ax.spines['bottom'].set_color('#2d5a3d')
        ax.spines['left'].set_color('#2d5a3d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, lengths):
            ax.text(bar.get_width() + max(lengths) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', color='#f0fdf4', fontsize=8)
        fig.patch.set_facecolor('#0a1f14')
        ax.set_facecolor('#0a1f14')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', facecolor='#0a1f14')
        buf.seek(0)
        bar_bytes = buf.getvalue()
        plt.close(fig)

    return area_bytes, bar_bytes


# ─────────────────────────── YOLO PATH ────────────────────────────────

def _process_yolo(image_pil, is_arugula: bool, show_boxes: bool,
                  selected_classes: list | None) -> dict:
    seg_model = arugula_seg_model if is_arugula else wheat_seg_model
    if seg_model is None:
        return {'error': 'Segmentation model not loaded.'}

    seg_results = seg_model(image_pil, verbose=False)

    class_metrics = {}
    available_classes = []
    valid_indices = []
    masks = classes = boxes = names = None

    if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
        masks   = seg_results[0].masks.data.cpu().numpy()
        classes = seg_results[0].boxes.cls.cpu().numpy()
        names   = seg_results[0].names
        boxes   = seg_results[0].boxes.xywh.cpu().numpy()

        unique_classes = {YOLO_RU_MAP.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
        available_classes = list(unique_classes)
        classes_to_show = unique_classes if not selected_classes else set(selected_classes)

        for i in range(len(masks)):
            cls_name_ru = YOLO_RU_MAP.get(names[int(classes[i])].lower(), names[int(classes[i])].lower())
            if cls_name_ru in classes_to_show:
                valid_indices.append(i)

        for i in valid_indices:
            cls_name_ru = YOLO_RU_MAP.get(names[int(classes[i])].lower(), names[int(classes[i])].lower())
            mask = masks[i]
            _, _, w, h = boxes[i]

            valid_pixels = np.count_nonzero(mask)
            area_mm2    = valid_pixels * (BASE_SCALE_FACTOR ** 2)
            length_mm   = math.sqrt(w ** 2 + h ** 2) * BASE_SCALE_FACTOR

            if cls_name_ru not in class_metrics:
                class_metrics[cls_name_ru] = {'count': 0, 'area': 0.0, 'length': 0.0}
            class_metrics[cls_name_ru]['count']  += 1
            class_metrics[cls_name_ru]['area']   += area_mm2
            class_metrics[cls_name_ru]['length'] += length_mm

    total_length = sum(d['length'] for d in class_metrics.values())
    total_area   = sum(d['area']   for d in class_metrics.values())

    # Annotated image
    if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
        if valid_indices and len(valid_indices) < len(masks):
            import copy
            fr = copy.deepcopy(seg_results)
            fr[0].masks.data = fr[0].masks.data[valid_indices]
            fr[0].boxes.data = fr[0].boxes.data[valid_indices]
            annotated_img = fr[0].plot(boxes=show_boxes)
        else:
            annotated_img = seg_results[0].plot(boxes=show_boxes)
    else:
        annotated_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    annotated_pil  = Image.fromarray(annotated_img_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format='JPEG', quality=95)
    annotated_bytes = buf.getvalue()

    area_bytes, bar_bytes = _build_charts(class_metrics)

    return {
        'class_metrics':         class_metrics,
        'available_classes':     available_classes,
        'annotated_image_bytes': annotated_bytes,
        'chart_bytes':           area_bytes,
        'bar_chart_bytes':       bar_bytes,
        'total_area':            total_area,
        'total_length':          total_length,
        'all_masks':             masks,
        'all_classes':           classes,
        'all_boxes':             boxes,
        'names':                 names,
        'class_metrics_all':     class_metrics,
        'error':                 None,
    }


# ─────────────────────────── U-NET PATH ───────────────────────────────

def _process_unet(image_pil, is_arugula: bool, selected_classes: list | None = None, show_boxes: bool = True) -> dict:
    if not UNET_AVAILABLE:
        return {'error': 'U-Net dependencies not available.'}

    import torch

    unet_model = unet_arugula_model if is_arugula else unet_wheat_model
    if unet_model is None:
        return {'error': f'U-Net model not loaded for {"arugula" if is_arugula else "wheat"}.'}

    orig_w, orig_h = image_pil.size
    img_np = np.array(image_pil)  # RGB uint8

    # Transform
    aug = _UNET_TRANSFORM(image=img_np)
    tensor = aug['image'].unsqueeze(0).to(DEVICE)  # (1,3,512,512)

    with torch.no_grad():
        logits = unet_model(tensor)           # (1,4,512,512)
        pred   = torch.argmax(logits, dim=1)  # (1,512,512)
        pred   = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # (512,512)

    # Resize prediction back to original resolution
    pred_orig = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Build coloured overlay
    overlay = img_np.copy()
    for class_id, color in UNET_CLASS_COLORS.items():
        if class_id == 0:
            continue  # skip background
            
        cls_name = UNET_CLASS_NAMES[class_id]
        if selected_classes and cls_name not in selected_classes:
            continue
            
        mask = (pred_orig == class_id)
        if not mask.any():
            continue
            
        # Draw overlay
        overlay[mask] = (
            overlay[mask] * 0.4 + np.array(color, dtype=np.float32) * 0.6
        ).astype(np.uint8)
        
        # Draw bounding boxes if requested
        if show_boxes:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= 5 and h >= 5:  # Skip tiny noise
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)

    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format='JPEG', quality=95)
    annotated_bytes = buf.getvalue()

    # ── Metrics from pixel mask ──────────────────────────────────────
    class_metrics = {}
    available_classes = []

    for class_id in range(1, 4):   # skip background
        cls_name = UNET_CLASS_NAMES[class_id]
        mask = (pred_orig == class_id)
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue
            
        # Optional class filtering logic for U-Net
        if selected_classes and cls_name not in selected_classes:
            continue

        available_classes.append(cls_name)
        area_mm2 = pixel_count * (BASE_SCALE_FACTOR ** 2)

        # Bounding box for length
        coords = np.argwhere(mask)  # (N, 2) row, col
        if len(coords) > 0:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h_px = float(r1 - r0)
            w_px = float(c1 - c0)
            length_mm = math.sqrt(w_px ** 2 + h_px ** 2) * BASE_SCALE_FACTOR
        else:
            length_mm = 0.0

        class_metrics[cls_name] = {
            'count':  1,          # U-Net: treat each class as 1 unit per image
            'area':   area_mm2,
            'length': length_mm,
        }

    total_length = sum(d['length'] for d in class_metrics.values())
    total_area   = sum(d['area']   for d in class_metrics.values())

    area_bytes, bar_bytes = _build_charts(class_metrics)

    return {
        'class_metrics':         class_metrics,
        'available_classes':     available_classes,
        'annotated_image_bytes': annotated_bytes,
        'chart_bytes':           area_bytes,
        'bar_chart_bytes':       bar_bytes,
        'total_area':            total_area,
        'total_length':          total_length,
        'all_masks':             None,
        'all_classes':           None,
        'all_boxes':             None,
        'names':                 None,
        'class_metrics_all':     class_metrics,
        'error':                 None,
    }


# ─────────────────────────── PUBLIC API ───────────────────────────────

def process_image(
        image_bytes: bytes,
        show_boxes: bool = True,
        selected_classes: list = None,
        seg_model: str = 'yolo',
) -> dict:
    """
    Process an image with classification and segmentation.

    Parameters
    ----------
    image_bytes     : raw image bytes
    show_boxes      : show bounding boxes (YOLO only)
    selected_classes: filter to specific part classes
    seg_model       : 'yolo' | 'unet'
    """
    try:
        if classificator_model is None:
            return {'error': 'Classification model not loaded. Check logs.'}

        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        is_arugula = _classify_image(image_pil)

        plant_name_ru = "Рукола" if is_arugula else "Пшеница"

        if seg_model == 'unet':
            result = _process_unet(image_pil, is_arugula, selected_classes, show_boxes)
        else:
            result = _process_yolo(image_pil, is_arugula, show_boxes, selected_classes)

        if result.get('error'):
            return result

        result['class_name'] = plant_name_ru
        return result

    except Exception as e:
        logging.error(f"Error in process_image: {e}", exc_info=True)
        return {'error': str(e)}
