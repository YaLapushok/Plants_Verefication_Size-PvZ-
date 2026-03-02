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

# Load models at module level to avoid reloading
try:
    logging.info("Loading YOLO models in core...")
    classificator_model = YOLO(os.path.join(MODELS_DIR, 'classificator.pt'))
    arugula_seg_model = YOLO(os.path.join(MODELS_DIR, 'arugula.pt'))
    wheat_seg_model = YOLO(os.path.join(MODELS_DIR, 'wheat.pt'))
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models. Ensure they exist in 'models/'. Error: {e}")
    classificator_model = None
    arugula_seg_model = None
    wheat_seg_model = None

CALIB_WIDTH = 2048
BASE_SCALE_FACTOR = 0.10556744  # mm/pixel


def process_image(
        image_bytes: bytes,
        show_boxes: bool = True,
        selected_classes: list = None
) -> dict:
    """
    Process image with classification and segmentation.
    """
    try:
        if classificator_model is None:
            return {'error': 'Models not loaded. Check logs.'}

        # Load image
        image_buffer = io.BytesIO(image_bytes)
        image = Image.open(image_buffer).convert('RGB')

        # 1. CLASSIFICATION
        cls_results = classificator_model(image, verbose=False)
        best_class_id = cls_results[0].probs.top1
        best_class_name = cls_results[0].names[best_class_id].lower()

        # Determine plant type
        is_arugula = False
        if 'arugula' in best_class_name or 'рукола' in best_class_name or 'rucola' in best_class_name:
            is_arugula = True
        elif best_class_id == 0 and 'wheat' not in best_class_name and 'пшеница' not in best_class_name:
            is_arugula = False
        elif best_class_id == 1 and 'wheat' not in best_class_name and 'пшеница' not in best_class_name:
            is_arugula = True

        # 2. SEGMENTATION
        if is_arugula:
            plant_name_ru = "Рукола 🌱"
            seg_model = arugula_seg_model
        else:
            plant_name_ru = "Пшеница 🌾"
            seg_model = wheat_seg_model

        seg_results = seg_model(image, verbose=False) if seg_model else None

        # 3. METRICS CALCULATION WITH FILTERING
        metrics_text = ""
        total_length = 0.0
        total_area = 0.0
        class_metrics = {}
        available_classes = []
        valid_indices = []

        if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
            masks = seg_results[0].masks.data.cpu().numpy()
            classes = seg_results[0].boxes.cls.cpu().numpy()
            names = seg_results[0].names
            boxes = seg_results[0].boxes.xywh.cpu().numpy()

            # Collect all unique classes
            unique_classes = set()
            for i in range(len(masks)):
                cls_id = int(classes[i])
                cls_name = names[cls_id].lower()
                unique_classes.add(cls_name)
            available_classes = list(unique_classes)

            # Determine which classes to show
            if selected_classes is None or len(selected_classes) == 0:
                classes_to_show = unique_classes
            else:
                classes_to_show = set(selected_classes)

            # Filter objects by selected classes
            for i in range(len(masks)):
                cls_id = int(classes[i])
                cls_name = names[cls_id].lower()
                if cls_name in classes_to_show:
                    valid_indices.append(i)

            # Calculate metrics for filtered objects
            for i in valid_indices:
                cls_id = int(classes[i])
                cls_name = names[cls_id].lower()
                mask = masks[i]

                _, _, w, h = boxes[i]

                # Area
                valid_pixels = np.count_nonzero(mask)
                area_mm2 = valid_pixels * (BASE_SCALE_FACTOR ** 2)

                # Length
                length_pixels = math.sqrt(w ** 2 + h ** 2)
                length_mm = length_pixels * BASE_SCALE_FACTOR

                if cls_name not in class_metrics:
                    class_metrics[cls_name] = {'count': 0, 'area': 0.0, 'length': 0.0}

                class_metrics[cls_name]['count'] += 1
                class_metrics[cls_name]['area'] += area_mm2
                class_metrics[cls_name]['length'] += length_mm

            total_length = sum(data['length'] for data in class_metrics.values())
            total_area = sum(data['area'] for data in class_metrics.values())

            if class_metrics:
                metrics_text = f"\n\n🌿 **Общие показатели растения:**\n" \
                               f"  Общая длина: {total_length:.2f} мм\n" \
                               f"  Общая площадь: {total_area:.2f} мм²\n" \
                               f"\n📊 **Детализация по частям:**\n"
                for c_name, data in class_metrics.items():
                    metrics_text += f"- **{c_name.capitalize()}** (шт: {data['count']}):\n"
                    metrics_text += f"  Площадь: {data['area']:.2f} мм²\n"
                    metrics_text += f"  Длина: {data['length']:.2f} мм\n"
            else:
                metrics_text = "\n\n⚠️ Выбранные классы не обнаружены на изображении."
        else:
            metrics_text = "\n\n⚠️ Не удалось распознать объекты для сегментации."

        # 4. ANNOTATED IMAGE WITH FILTERING
        if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
            if valid_indices and len(valid_indices) < len(masks):
                # Create filtered results for plotting
                import copy
                filtered_seg_results = copy.deepcopy(seg_results)

                if filtered_seg_results[0].masks is not None:
                    filtered_seg_results[0].masks.data = filtered_seg_results[0].masks.data[valid_indices]
                if filtered_seg_results[0].boxes is not None:
                    # Filter boxes data
                    filtered_seg_results[0].boxes.data = filtered_seg_results[0].boxes.data[valid_indices]

                annotated_img = filtered_seg_results[0].plot(boxes=show_boxes)
            else:
                annotated_img = seg_results[0].plot(boxes=show_boxes)
        else:
            annotated_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to JPEG
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img_rgb)
        output_buffer = io.BytesIO()
        annotated_pil.save(output_buffer, format="JPEG", quality=95)
        annotated_bytes = output_buffer.getvalue()

        # 5. PIE CHART
        chart_bytes = None
        if total_area > 0 and len(class_metrics) > 0:
            labels = [c_name.capitalize() for c_name in class_metrics.keys()]
            sizes = [data['area'] for data in class_metrics.values()]

            fig, ax = plt.subplots(figsize=(5, 5))
            colors = ['#03a062', '#00e676', '#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                   colors=colors[:len(labels)], textprops={'color': '#f0fdf4', 'fontsize': 9})
            ax.set_title('Соотношение площадей (мм²)', color='#f0fdf4', fontsize=11)
            ax.patch.set_facecolor('#0a1f14')

            chart_buffer = io.BytesIO()
            plt.savefig(chart_buffer, format='jpeg', bbox_inches='tight', facecolor='#0a1f14')
            chart_buffer.seek(0)
            chart_bytes = chart_buffer.getvalue()
            plt.close(fig)


        return {
            'class_name': plant_name_ru,
            'metrics_text': metrics_text,
            'class_metrics': class_metrics,
            'annotated_image_bytes': annotated_bytes,
            'chart_bytes': chart_bytes,
            'total_area': total_area,
            'total_length': total_length,
            'available_classes': available_classes,
            'error': None,
            # === ДЛЯ ДИНАМИЧЕСКОЙ ФИЛЬТРАЦИИ ===
            'all_masks': masks if 'masks' in locals() else None,
            'all_classes': classes if 'classes' in locals() else None,
            'all_boxes': boxes if 'boxes' in locals() else None,
            'names': names if 'names' in locals() else None,
            'class_metrics_all': class_metrics
        }


    except Exception as e:
        logging.error(f"Error in ML core process_image: {e}", exc_info=True)
        return {'error': str(e)}
