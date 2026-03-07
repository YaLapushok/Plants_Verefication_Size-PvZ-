# ─────────────────────────── CONSTANTS ────────────────────────────────
BASE_SCALE_FACTOR = 0.10556744  # mm/pixel (confirmed by calibration dataset)

# Class colour map for U-Net overlay
# 0=background, 1=root, 2=stem, 3=leaf
UNET_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 140, 0),  # корень (оранжевый)
    2: (0, 120, 255),  # листок (синий)
    3: (50, 205, 50),  # стебель (зелёный)
}

UNET_CLASS_NAMES = {
    0: 'фон',
    1: 'корень',
    2: 'листок',
    3: 'стебель',
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

# Colors for UI/Charts
COLOR_MAP = {
    'корень': '#ff8c00',
    'стебель': '#32cd32',
    'листок': '#0078ff',
    'колос': '#ff6384',
}
