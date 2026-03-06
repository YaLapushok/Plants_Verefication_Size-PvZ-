import sys
import os
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import logging
import zipfile
import csv
import io
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ml import process_image
from app.database import init_db, save_analysis, list_analyses, get_analysis, delete_analysis, delete_all_analyses


@asynccontextmanager
async def lifespan(app):
    init_db()
    logging.info("Database initialised.")
    yield


app = FastAPI(title="Plant Verification App", lifespan=lifespan)

STATIC_DIR    = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────── MAIN PAGE ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze", response_class=JSONResponse)
async def api_analyze(
        request: Request,
        file: UploadFile = File(...),
        show_boxes: str = Form("on"),
        seg_model: str  = Form('unet'),
        selected_classes: list = Form(default_factory=list)
):
    image_bytes  = await file.read()
    b_show_boxes = (show_boxes == "on" or show_boxes == "true" or show_boxes is True)
    if not selected_classes:
        selected_classes = None

    result = process_image(
        image_bytes,
        show_boxes=b_show_boxes,
        selected_classes=selected_classes,
        seg_model=seg_model,
    )

    if result.get('error'):
        return JSONResponse({'error': result['error']}, status_code=500)

    # ── Persist to database ──────────────────────────────────────────
    analysis_id = None
    try:
        analysis_id = save_analysis(
            filename        = file.filename or 'upload',
            plant_class     = result['class_name'],
            seg_model       = seg_model,
            total_length    = result['total_length'],
            total_area      = result['total_area'],
            class_metrics   = result['class_metrics'],
            original_bytes  = image_bytes,
            segmented_bytes = result['annotated_image_bytes'],
        )
    except Exception as db_err:
        logging.warning(f"DB save failed: {db_err}")

    annotated_b64 = base64.b64encode(result['annotated_image_bytes']).decode()
    chart_b64     = base64.b64encode(result['chart_bytes']).decode()     if result.get('chart_bytes')     else None
    bar_chart_b64 = base64.b64encode(result['bar_chart_bytes']).decode() if result.get('bar_chart_bytes') else None

    return JSONResponse({
        "success":          True,
        "analysis_id":      analysis_id,
        "class_name":       result['class_name'],
        "total_area":       f"{result['total_area']:.2f}",
        "total_length":     f"{result['total_length']:.2f}",
        "class_metrics":    result['class_metrics'],
        "annotated_b64":    annotated_b64,
        "chart_b64":        chart_b64,
        "bar_chart_b64":    bar_chart_b64,
        "seg_model":        seg_model,
        "available_classes":result.get('available_classes', []),
        "original_filename":file.filename,
    })


# ─────────────────────────── DYNAMIC FILTER ───────────────────────────

@app.post("/api/filter", response_class=JSONResponse)
async def api_filter(request: Request):
    try:
        data             = await request.json()
        analysis_id      = data.get('analysis_id')
        selected_classes = data.get('selected_classes', [])
        show_boxes       = data.get('show_boxes', True)

        if not analysis_id:
            return JSONResponse({'error': 'Missing analysis_id'}, status_code=400)

        row = get_analysis(analysis_id)
        if not row:
            return JSONResponse({'error': 'Analysis not found in DB'}, status_code=404)

        result = process_image(
            row['original_bytes'],
            show_boxes       = show_boxes,
            selected_classes = selected_classes or None,
            seg_model        = row['seg_model'],
        )

        if result.get('error'):
            return JSONResponse({'error': result['error']}, status_code=500)

        return JSONResponse({
            'success':       True,
            'annotated_b64': base64.b64encode(result['annotated_image_bytes']).decode(),
            'total_area':    f"{result['total_area']:.2f}",
            'total_length':  f"{result['total_length']:.2f}",
            'class_metrics': result['class_metrics'],
        })
    except Exception as e:
        logging.error(f"Filter error: {e}", exc_info=True)
        return JSONResponse({'error': str(e)}, status_code=500)

# ─────────────────────────── EXPORT ZIP ───────────────────────────────

@app.post("/api/export")
async def api_export(request: Request):
    try:
        data = await request.json()
        analysis_ids = data.get('analysis_ids', [])
        if not analysis_ids:
            return JSONResponse({'error': 'No ids provided'}, status_code=400)

        rows = [get_analysis(i) for i in analysis_ids if get_analysis(i) is not None]
        if not rows:
            return JSONResponse({'error': 'None of the requested analyses were found in DB'}, status_code=404)

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add images
            for row in rows:
                base_name = os.path.splitext(row['filename'])[0]
                if row.get('original_bytes'):
                    zip_file.writestr(f"images/original/{base_name}_{row['id']}.jpg", row['original_bytes'])
                if row.get('segmented_bytes'):
                    zip_file.writestr(f"images/segmented/{base_name}_{row['id']}_{row['seg_model']}.jpg", row['segmented_bytes'])
            
            # Generate CSV report
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['ID', 'Date', 'Filename', 'Plant', 'Model', 'Total Length (mm)', 'Total Area (mm2)', 'Class Metrics (JSON)'])
            for row in rows:
                writer.writerow([
                    row['id'], 
                    row['timestamp'], 
                    row['filename'], 
                    row['plant_class'], 
                    row['seg_model'],
                    f"{row['total_length']:.2f}",
                    f"{row['total_area']:.2f}",
                    json.dumps(row.get('class_metrics', {}), ensure_ascii=False)
                ])
            
            zip_file.writestr("report.csv", csv_buffer.getvalue().encode('utf-8'))
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=analysis_export.zip"}
        )
    except Exception as e:
        logging.error(f"Export error: {e}", exc_info=True)
        return JSONResponse({'error': str(e)}, status_code=500)


# ─────────────────────────── DATABASE GALLERY ─────────────────────────

@app.get("/db", response_class=HTMLResponse)
async def db_gallery(request: Request, plant: str = ""):
    rows = list_analyses()
    if plant:
        rows = [r for r in rows if plant.lower() in r['plant_class'].lower()]
    plant_types = sorted({r['plant_class'] for r in list_analyses()})
    return templates.TemplateResponse("db.html", {
        "request":     request,
        "rows":        rows,
        "plant_types": plant_types,
        "active_plant":plant,
    })


@app.get("/db/image/{analysis_id}/{kind}")
async def db_image(analysis_id: int, kind: str):
    """Serve stored original or segmented image bytes."""
    row = get_analysis(analysis_id)
    if not row:
        return Response(status_code=404)
    key = 'original_bytes' if kind == 'original' else 'segmented_bytes'
    data = row.get(key)
    if not data:
        return Response(status_code=404)
    return Response(content=bytes(data), media_type="image/jpeg")


@app.post("/db/delete/{analysis_id}")
async def db_delete(analysis_id: int):
    delete_analysis(analysis_id)
    return RedirectResponse(url="/db", status_code=303)


@app.post("/db/clear")
async def db_clear():
    delete_all_analyses()
    return RedirectResponse(url="/db", status_code=303)
