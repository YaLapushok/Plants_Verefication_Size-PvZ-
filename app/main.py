import sys
import os
import base64
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import cv2
import numpy as np
import math

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ml import process_image, get_ru_name, CLASS_NAME_RU

app = FastAPI(title="Plant Verification App")

STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# === ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДЛЯ ФИЛЬТРАЦИИ ===
last_result_cache = {
    'original_image_bytes': None,
    'all_masks': None,
    'all_classes': None,
    'all_boxes': None,
    'names': None,
    'class_metrics_all': None
}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
        request: Request,
        file: UploadFile = File(...),
        show_boxes: str = Form(None),
        selected_classes: list = Form(default_factory=list)
):
    global last_result_cache

    image_bytes = await file.read()
    b_show_boxes = (show_boxes == "on")

    if not selected_classes:
        selected_classes = None

    result = process_image(
        image_bytes,
        show_boxes=b_show_boxes,
        selected_classes=selected_classes
    )

    if result.get('error'):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Ошибка обработки: {result['error']}"
        })

    # Сохраняем ВСЕ данные для последующей фильтрации
    last_result_cache = {
        'original_image_bytes': image_bytes,
        'all_masks': result.get('all_masks'),
        'all_classes': result.get('all_classes'),
        'all_boxes': result.get('all_boxes'),
        'names': result.get('names'),
        'class_metrics_all': result.get('class_metrics_all', {}),
        'available_classes': result.get('available_classes', [])
    }

    annotated_b64 = base64.b64encode(result['annotated_image_bytes']).decode('utf-8')

    chart_b64 = None
    if result.get('chart_bytes'):
        chart_b64 = base64.b64encode(result['chart_bytes']).decode('utf-8')

    # Подготавливаем русские названия для шаблона
    class_names_ru = {k: get_ru_name(k) for k in result.get('class_metrics', {}).keys()}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_name": result['class_name'],
        "total_area": f"{result['total_area']:.2f}",
        "total_length": f"{result['total_length']:.2f}",
        "class_metrics": result['class_metrics'],
        "class_names_ru": class_names_ru,
        "annotated_b64": annotated_b64,
        "chart_b64": chart_b64,
        "success": True,
        "selected_classes": selected_classes,
        "available_classes": result.get('available_classes', [])
    })


@app.post("/api/filter", response_class=JSONResponse)
async def api_filter(request: Request):
    """API для динамической фильтрации без перезагрузки"""
    global last_result_cache

    try:
        data = await request.json()
        selected_classes = data.get('selected_classes', [])
        show_boxes = data.get('show_boxes', True)

        if not last_result_cache.get('original_image_bytes'):
            return JSONResponse({
                'error': 'No image in cache. Please upload an image first.'
            }, status_code=400)

        # Переобрабатываем изображение с новыми фильтрами
        from core.ml import process_image

        result = process_image(
            last_result_cache['original_image_bytes'],
            show_boxes=show_boxes,
            selected_classes=selected_classes if selected_classes else None
        )

        if result.get('error'):
            return JSONResponse({'error': result['error']}, status_code=500)

        # Кодируем в base64
        annotated_b64 = base64.b64encode(result['annotated_image_bytes']).decode('utf-8')

        return JSONResponse({
            'success': True,
            'annotated_b64': annotated_b64,
            'total_area': f"{result['total_area']:.2f}",
            'total_length': f"{result['total_length']:.2f}",
            'class_metrics': result['class_metrics']
        })

    except Exception as e:
        import logging
        logging.error(f"Filter error: {e}", exc_info=True)
        return JSONResponse({
            'error': str(e)
        }, status_code=500)
