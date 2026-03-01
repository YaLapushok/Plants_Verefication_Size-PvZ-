import io
import os
import cv2
import numpy as np
import logging
import math
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cdist
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
    # Paths are absolute to the models directory
    classificator_model = YOLO(os.path.join(MODELS_DIR, 'classificator.pt'))
    arugula_seg_model = YOLO(os.path.join(MODELS_DIR, 'arugula.pt'))
    wheat_seg_model = YOLO(os.path.join(MODELS_DIR, 'wheat.pt'))
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models. Ensure they exist in 'models/'. Error: {e}")

CALIB_WIDTH = 2048
BASE_SCALE_FACTOR = 0.10556744  # mm/pixel, derived exactly for 2048px width images

def process_image(image_bytes: bytes, show_boxes: bool = True) -> dict:
    """
    Takes an image in bytes, runs classification and segmentation.
    Returns a dict with:
    - 'class_name': str (ru name)
    - 'metrics_text': str
    - 'class_metrics': dict
    - 'annotated_image_bytes': bytes
    - 'chart_bytes': bytes (or None)
    - 'total_area': float
    - 'total_length': float
    - 'error': str (if any)
    """
    try:
        image_buffer = io.BytesIO(image_bytes)
        image = Image.open(image_buffer)
        
        # 1. Classification
        cls_results = classificator_model(image, verbose=False)
        best_class_id = cls_results[0].probs.top1
        best_class_name = cls_results[0].names[best_class_id].lower()
        
        # Determine specific model
        is_arugula = False
        if 'arugula' in best_class_name or 'рукола' in best_class_name or 'rucola' in best_class_name:
            is_arugula = True
        elif best_class_id == 0 and 'wheat' not in best_class_name and 'пшеница' not in best_class_name:
            is_arugula = False
        elif best_class_id == 1 and 'wheat' not in best_class_name and 'пшеница' not in best_class_name:
            is_arugula = True

        # 2. Segmentation
        if is_arugula:
            plant_name_ru = "Рукола 🌱"
            seg_results = arugula_seg_model(image, verbose=False)
            
            # Костыль: меняем классы 'stem' и 'leaf' местами только для руколы
            # Так как модель ошибочно помечает стебель классом листка и наоборот
            if seg_results and len(seg_results) > 0 and getattr(seg_results[0], 'names', None):
                names_dict = seg_results[0].names
                leaf_id, stem_id = None, None
                for k, v in names_dict.items():
                    v_lower = str(v).lower()
                    if 'leaf' in v_lower or 'лист' in v_lower:
                        leaf_id = k
                    elif 'stem' in v_lower or 'стебель' in v_lower:
                        stem_id = k
                
                if leaf_id is not None and stem_id is not None:
                    # Подменяем словарь названий классов (для отображения)
                    new_names = names_dict.copy()
                    new_names[leaf_id], new_names[stem_id] = names_dict[stem_id], names_dict[leaf_id]
                    seg_results[0].names = new_names
                    
                    # Подменяем сами предсказания (классы в тензорах)
                    if seg_results[0].boxes is not None:
                        cls_tensor = seg_results[0].boxes.cls
                        
                        # Создаем копию для безопасной замены
                        new_cls = cls_tensor.clone()
                        
                        # Заменяем leaf_id на stem_id
                        new_cls[cls_tensor == leaf_id] = stem_id
                        # Заменяем stem_id на leaf_id
                        new_cls[cls_tensor == stem_id] = leaf_id
                        
                        seg_results[0].boxes.cls = new_cls
                        
                        # Важно для plot(): YOLO кэширует оригинальные названия при отрисовке, 
                        # иногда обращаясь к исходным данным модели.
                        # Надежнее всего поменять значения ключей в словаре names:
                        seg_results[0].names[leaf_id] = names_dict[stem_id]
                        seg_results[0].names[stem_id] = names_dict[leaf_id]
        else:
            plant_name_ru = "Пшеница 🌾"
            seg_results = wheat_seg_model(image, verbose=False)

        # 3. Metrics
        metrics_text = ""
        total_length = 0.0
        total_area = 0.0
        class_metrics = {}
        
        if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
            masks = seg_results[0].masks.data.cpu().numpy()
            classes = seg_results[0].boxes.cls.cpu().numpy()
            names = seg_results[0].names
            boxes = seg_results[0].boxes.xywh.cpu().numpy() # [x_center, y_center, width, height]
            
            for i in range(len(masks)):
                cls_id = int(classes[i])
                cls_name = names[cls_id].lower()
                mask = masks[i]
                
                # Bounding box dimensions
                _, _, w, h = boxes[i]
                
                # Area calculation
                valid_pixels = np.count_nonzero(mask)
                area_mm2 = valid_pixels * (BASE_SCALE_FACTOR ** 2)
                
                # Length calculation based on the diagonal of the bounding box
                # The user verified that this geometric extent accurately reflects the object lengths
                length_pixels = math.sqrt(w**2 + h**2)
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
            metrics_text = "\n\n⚠️ Не удалось распознать объекты для сегментации."

        # 4. Extract Annotated Image
        annotated_img = seg_results[0].plot(boxes=show_boxes)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img_rgb)
        
        output_buffer = io.BytesIO()
        annotated_pil.save(output_buffer, format="JPEG")
        annotated_bytes = output_buffer.getvalue()

        # 5. Chart
        chart_bytes = None
        if total_area > 0 and len(class_metrics) > 0:
            labels = [c_name.capitalize() for c_name in class_metrics.keys()]
            sizes = [data['area'] for data in class_metrics.values()]
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                   colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'][:len(labels)])
            ax.set_title('Соотношение площадей структур (мм²)')
            
            chart_buffer = io.BytesIO()
            plt.savefig(chart_buffer, format='jpeg', bbox_inches='tight')
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
            'error': None
        }

    except Exception as e:
        logging.error(f"Error in ML core process_image: {e}", exc_info=True)
        return {'error': str(e)}
