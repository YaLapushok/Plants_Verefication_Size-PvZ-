import os
import logging
from ultralytics import YOLO

# Define base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ─────────────────────────── YOLO MODELS ────────────────────────────
classificator_model = None
arugula_seg_model = None
wheat_seg_model = None

try:
    logging.info("Loading YOLO models...")
    classificator_model = YOLO(os.path.join(MODELS_DIR, 'classificator.pt'))
    arugula_seg_model = YOLO(os.path.join(MODELS_DIR, 'arugula.pt'))
    wheat_seg_model = YOLO(os.path.join(MODELS_DIR, 'wheat.pt'))
    logging.info("YOLO models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO models: {e}")

# ─────────────────────────── U-NET MODELS ─────────────────────────────
unet_arugula_model = None
unet_wheat_model = None
UNET_AVAILABLE = False
DEVICE = 'cpu'

try:
    import torch
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device for U-Net: {DEVICE}")

    def _build_unet():
        return smp.Unet(
            encoder_name='resnet50',
            encoder_weights=None,
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
    UNET_TRANSFORM = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    UNET_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to load U-Net models: {e}")

def classify_image(image_pil) -> bool:
    """Return True if the image is arugula, False for wheat."""
    if classificator_model is None:
        return False
    cls_results = classificator_model(image_pil, verbose=False)
    best_class_id = cls_results[0].probs.top1
    best_class_name = cls_results[0].names[best_class_id].lower()

    # Heuristics for classification
    if any(k in best_class_name for k in ['arugula', 'рукола', 'rucola']):
        return True
    if best_class_id == 1 and not any(k in best_class_name for k in ['wheat', 'пшеница']):
        return True
    return False
