import sys
import os
import base64
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add the root directory to sys.path so we can import 'core'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ml import process_image

app = FastAPI(title="Plant Verification App")

# We will mount a static directory for CSS and any static images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...), show_boxes: str = Form(None)):
    # Read image contents
    image_bytes = await file.read()
    
    # Process it via our shared ML Core
    b_show_boxes = (show_boxes == "on")
    result = process_image(image_bytes, show_boxes=b_show_boxes)
    
    if result.get('error'):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_message": f"Ошибка обработки: {result['error']}"
        })
        
    # We will pass base64 encoded strings to embed images directly into HTML without saving to disk
    annotated_b64 = base64.b64encode(result['annotated_image_bytes']).decode('utf-8')
    
    chart_b64 = None
    if result.get('chart_bytes'):
        chart_b64 = base64.b64encode(result['chart_bytes']).decode('utf-8')
        
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_name": result['class_name'],
        "total_area": f"{result['total_area']:.2f}",
        "total_length": f"{result['total_length']:.2f}",
        "class_metrics": result['class_metrics'],
        "annotated_b64": annotated_b64,
        "chart_b64": chart_b64,
        "success": True
    })
