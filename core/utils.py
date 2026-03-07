import io
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from core.constants import YOLO_RU_MAP, COLOR_MAP

def get_font(size=16):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    return ImageFont.load_default()

def plot_yolo_custom(result, show_boxes: bool = True) -> np.ndarray:
    img = result.orig_img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(16)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    names_dict = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    class_color_map_bgr = {
        'koren': (255, 140, 0),
        'stebel': (50, 205, 50),
        'listok': (0, 120, 255),
        'kolos': (255, 99, 132),
        'root': (255, 140, 0),
        'stem': (50, 205, 50),
        'leaf': (0, 120, 255),
    }

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        class_id = int(cls[i])
        confidence = conf[i]

        orig_name = names_dict.get(class_id, str(class_id))
        orig_name_lower = str(orig_name).lower()
        ru_name = YOLO_RU_MAP.get(orig_name_lower, orig_name.capitalize())
        color_rgb = class_color_map_bgr.get(orig_name_lower, (128, 128, 128))

        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        label = f"{ru_name} {confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_height - 8, x1 + text_width + 4, y1], fill=color_rgb)
        draw.text((x1 + 2, y1 - text_height - 6), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def build_charts(class_metrics: dict) -> tuple[bytes | None, bytes | None]:
    if not class_metrics:
        return None, None

    labels = [n.capitalize() for n in class_metrics.keys()]
    areas = [d['area'] for d in class_metrics.values()]
    lengths = [d['length'] for d in class_metrics.values()]
    colors = [COLOR_MAP.get(label.lower(), '#9966ff') for label in labels]

    def create_barh_chart(values, title, xlabel):
        fig, ax = plt.subplots(figsize=(5, max(2.5, len(labels) * 0.8)))
        y_pos = range(len(labels))
        bars = ax.barh(list(y_pos), values, color=colors, height=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color='#f0fdf4', fontsize=9)
        ax.set_xlabel(xlabel, color='#f0fdf4', fontsize=9)
        ax.set_title(title, color='#f0fdf4', fontsize=11)
        ax.tick_params(axis='x', colors='#f0fdf4')
        for spine in ax.spines.values():
            spine.set_color('#2d5a3d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', color='#f0fdf4', fontsize=8)
        
        fig.patch.set_facecolor('#0a1f14')
        ax.set_facecolor('#0a1f14')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', facecolor='#0a1f14')
        plt.close(fig)
        return buf.getvalue()

    area_bytes = create_barh_chart(areas, 'Площадь по частям (мм²)', 'Площадь (мм²)') if any(a > 0 for a in areas) else None
    bar_bytes = create_barh_chart(lengths, 'Длина по частям (мм)', 'Длина (мм)') if any(l > 0 for l in lengths) else None

    return area_bytes, bar_bytes
