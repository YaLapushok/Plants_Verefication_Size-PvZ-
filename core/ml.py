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
    arugula_seg_model = YOLO(os.path.join(MODELS_DIR, 'arugula.pt'))
    wheat_seg_model = YOLO(os.path.join(MODELS_DIR, 'wheat.pt'))
    logging.info("YOLO models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO models: {e}")
    classificator_model = None
    arugula_seg_model = None
    wheat_seg_model = None

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
            encoder_weights=None,  # weights loaded from .pth
            in_channels=3,
            classes=4,
            activation=None,
        )


    _UNET_ARUGULA_PATH = os.path.join(MODELS_DIR, 'U-Net', 'rugola_v3_best.pth')
    _UNET_WHEAT_PATH = os.path.join(MODELS_DIR, 'U-Net', 'wheat_4classes.pth')


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
    unet_wheat_model = _load_unet(_UNET_WHEAT_PATH)

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
    unet_wheat_model = None
    UNET_AVAILABLE = False

CALIB_WIDTH = 2048
BASE_SCALE_FACTOR = 0.10556744  # mm/pixel

# Class colour map for U-Net overlay
# 0=background, 1=root, 2=stem, 3=leaf
UNET_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 140, 0),  # корень
    2: (0, 120, 255),  # листок (синий) ← поменяли
    3: (50, 205, 50),  # стебель (зелёный) ← поменяли
}
# Arugula U-Net classes (same indices as wheat for the unified architecture)
UNET_CLASS_NAMES = {
    0: 'background',
    1: 'корень',
    2: 'листок',    # ← поменяли
    3: 'стебель',   # ← поменяли
}

# Mapping English YOLO names to Russian
YOLO_RU_MAP = {
    'koren': 'корень',
    'stebel': 'стебель',
    'listok': 'листок',
    'kolos': 'колос',
    'root': 'корень',
    'stem': 'стебель',
    'leaf': 'листок'
}


def _plot_yolo_custom(result, show_boxes: bool = True) -> np.ndarray:
    """
    Custom YOLO plotting with Russian labels using PIL for Cyrillic support.
    """
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np

    img = result.orig_img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Попытка загрузить шрифт с поддержкой кириллицы
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "arial.ttf",
        "DejaVuSans.ttf",
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    names_dict = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    class_name_map = {
        'koren': 'Корень',
        'stebel': 'Стебель',
        'listok': 'Листок',
        'kolos': 'Колос',
        'root': 'Корень',
        'stem': 'Стебель',
        'leaf': 'Листок',
    }

    class_color_map = {
        'koren': (0, 140, 255),
        'stebel': (50, 205, 50),
        'listok': (255, 120, 0),
        'kolos': (134, 99, 255),
        'root': (0, 140, 255),
        'stem': (50, 205, 50),
        'leaf': (255, 120, 0),
    }

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        class_id = int(cls[i])
        confidence = conf[i]

        try:
            if isinstance(names_dict, dict):
                orig_name = names_dict.get(class_id, str(class_id))
            elif isinstance(names_dict, list):
                orig_name = names_dict[class_id] if class_id < len(names_dict) else str(class_id)
            else:
                orig_name = str(class_id)
        except:
            orig_name = str(class_id)

        orig_name_lower = str(orig_name).lower()
        ru_name = class_name_map.get(orig_name_lower, orig_name.capitalize())
        color = class_color_map.get(orig_name_lower, (128, 128, 128))
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB

        # Draw bounding box
        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        # Draw label
        label = f"{ru_name} {confidence:.2f}"

        # Get text size
        if font:
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(label) * 10
            text_height = 20

        # Draw background rectangle
        draw.rectangle(
            [x1, y1 - text_height - 8, x1 + text_width, y1],
            fill=color_rgb
        )

        # Draw text
        draw.text((x1, y1 - text_height - 4), label, fill=(255, 255, 255), font=font)

    # Convert back to OpenCV format
    img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_result

# ─────────────────────────── HELPERS ──────────────────────────────────

def _classify_image(image_pil) -> bool:
    """Return True if the image is arugula, False for wheat."""
    if classificator_model is None:
        return False
    cls_results = classificator_model(image_pil, verbose=False)
    best_class_id = cls_results[0].probs.top1
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
    areas = [d['area'] for d in class_metrics.values()]
    lengths = [d['length'] for d in class_metrics.values()]
    # Синхронизируем цвета с UNET_CLASS_COLORS
    color_map = {
        'корень': '#ff8c00',  # оранжевый
        'стебель': '#32cd32',  # зелёный
        'листок': '#0078ff',  # синий
        'колос': '#ff6384',  # розовый
    }
    colors = [color_map.get(label.lower(), '#9966ff') for label in labels]

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
        masks = seg_results[0].masks.data.cpu().numpy()
        classes = seg_results[0].boxes.cls.cpu().numpy()
        names = seg_results[0].names
        boxes = seg_results[0].boxes.xywh.cpu().numpy()

        unique_classes = {YOLO_RU_MAP.get(names[int(c)].lower(), names[int(c)].lower()) for c in classes}
        available_classes = list(unique_classes)
        # Если selected_classes == None — показать всё, если список (даже пустой) — фильтровать
        if selected_classes is None:
            classes_to_show = unique_classes
        else:
            classes_to_show = set(selected_classes)

        for i in range(len(masks)):
            cls_name_ru = YOLO_RU_MAP.get(names[int(classes[i])].lower(), names[int(classes[i])].lower())
            if cls_name_ru.lower() in [c.lower() for c in classes_to_show]:
                valid_indices.append(i)

        for i in valid_indices:
            cls_name_ru = YOLO_RU_MAP.get(names[int(classes[i])].lower(), names[int(classes[i])].lower())
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

    total_length = sum(d['length'] for d in class_metrics.values())
    total_area = sum(d['area'] for d in class_metrics.values())

    if valid_indices and len(valid_indices) < len(masks):
        import copy
        fr = copy.deepcopy(seg_results)
        fr[0].masks.data = fr[0].masks.data[valid_indices]
        fr[0].boxes.data = fr[0].boxes.data[valid_indices]
        # Исправление: вызываем отрисовку для отфильтрованных данных
        annotated_img = _plot_yolo_custom(fr[0], show_boxes=show_boxes)
    else:
        annotated_img = _plot_yolo_custom(seg_results[0], show_boxes=show_boxes)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_img_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format='JPEG', quality=95)
    annotated_bytes = buf.getvalue()

    area_bytes, bar_bytes = _build_charts(class_metrics)


    return {
        'class_metrics': class_metrics,
        'available_classes': available_classes,
        'annotated_image_bytes': annotated_bytes,
        'chart_bytes': area_bytes,
        'bar_chart_bytes': bar_bytes,
        'total_area': total_area,
        'total_length': total_length,
        'all_masks': masks,
        'all_classes': classes,
        'all_boxes': boxes,
        'names': names,
        'class_metrics_all': class_metrics,
        'error': None,
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
        logits = unet_model(tensor)  # (1,4,512,512)
        pred = torch.argmax(logits, dim=1)  # (1,512,512)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # (512,512)

    # Resize prediction back to original resolution
    pred_orig = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


    # Build coloured overlay
    overlay = img_np.copy()
    for class_id, color in UNET_CLASS_COLORS.items():
        if class_id == 0:
            continue  # skip background
        cls_name = UNET_CLASS_NAMES[class_id]
        if selected_classes is not None:
            selected_lower = [c.lower() for c in selected_classes]
            if cls_name.lower() not in selected_lower:
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
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format='JPEG', quality=95)
    annotated_bytes = buf.getvalue()

    # ── Metrics from pixel mask ─────────────────────────────────
    class_metrics = {}
    available_classes = []

    for class_id in range(1, 4):  # skip background
        cls_name = UNET_CLASS_NAMES[class_id]
        mask = (pred_orig == class_id)
        pixel_count = int(mask.sum())

        if pixel_count == 0:
            continue

        if selected_classes is not None:
            selected_lower = [c.lower() for c in selected_classes]
            if cls_name.lower() not in selected_lower:
                continue

        available_classes.append(cls_name)
        area_mm2 = pixel_count * (BASE_SCALE_FACTOR ** 2)

        # Подсчет количества отдельных объектов (связных компонент)
        mask_uint8 = mask.astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        # num_labels включает фон (0), поэтому реальных объектов: num_labels - 1
        object_count = num_labels - 1

        # Bounding box для длины (считаем по всем объектам класса)
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
            'count': object_count,  # Теперь считаем реальные объекты!
            'area': area_mm2,
            'length': length_mm,
        }

    total_length = sum(d['length'] for d in class_metrics.values())
    total_area = sum(d['area'] for d in class_metrics.values())

    area_bytes, bar_bytes = _build_charts(class_metrics)

    # Добавляем информацию о цветах для легенды U-Net
    class_colors = {}
    for class_id, color in UNET_CLASS_COLORS.items():
        if class_id == 0: continue  # пропускаем фон
        cls_name = UNET_CLASS_NAMES[class_id]
        if cls_name in class_metrics:  # добавляем только найденные классы
            class_colors[cls_name] = f"rgb{color}"


    return {
        'class_metrics': class_metrics,
        'class_colors': class_colors,
        'available_classes': available_classes,
        'annotated_image_bytes': annotated_bytes,
        'chart_bytes': area_bytes,
        'bar_chart_bytes': bar_bytes,
        'total_area': total_area,
        'total_length': total_length,
        'all_masks': None,
        'all_classes': None,
        'all_boxes': None,
        'names': None,
        'class_metrics_all': class_metrics,
        'error': None,
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
