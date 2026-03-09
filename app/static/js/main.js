/**
 * GLOBAL STATE
 */
const analysisDataStore = {};
const processedAnalysisIds = [];
let currentAnalysisId = null;
const activeLayers = {
    'корень': true,
    'листок': true,
    'стебель': true,
    'колос': true
};

// ==================== EDITOR ENGINE ====================
const editor = {
    id: null,
    zoom: 1,
    minZoom: 0.1,
    isDrawing: false,
    isPanning: false,
    lastX: 0, lastY: 0,
    tool: 'brush',
    color: '#000000',
    history: [],

    // Elements
    container: null, wrapper: null,
    bg: null, ai: null, mask: null,
    bgCtx: null, aiCtx: null, maskCtx: null,

    init() {
        this.container = document.getElementById('canvas-container');
        this.wrapper = document.getElementById('canvas-wrapper');
        this.bg = document.getElementById('bg-canvas');
        this.ai = document.getElementById('ai-canvas');
        this.mask = document.getElementById('mask-canvas');

        if (!this.bg) return; // Not on editor page

        this.bgCtx = this.bg.getContext('2d');
        this.aiCtx = this.ai.getContext('2d');
        this.maskCtx = this.mask.getContext('2d', { willReadFrequently: true });

        this.setupEvents();
    },

    setupEvents() {
        // Drawing events on Mask
        this.mask.onmousedown = (e) => this.handleMouseDown(e);
        this.mask.onmousemove = (e) => this.handleMouseMove(e);
        this.mask.onmouseup = () => this.stopAction();
        this.mask.onmouseleave = () => this.stopAction();
        this.mask.oncontextmenu = (e) => e.preventDefault(); // Disable menu for right-click pan

        // Zoom
        this.container.onwheel = (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                this.handleZoom(e);
            }
        };

        // Undo Shortcut (Ctrl+Z)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                const modal = document.getElementById('detailModal');
                if (modal && modal.classList.contains('open')) {
                    this.undo();
                }
            }
        });
    },

    handleZoom(e) {
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const nextZoom = this.zoom * delta;

        // Clamp zoom between minZoom and 10x
        this.zoom = Math.min(Math.max(nextZoom, this.minZoom), 10);

        const containerRect = this.container.getBoundingClientRect();

        const newWidth = this.bg.width * this.zoom;
        const newHeight = this.bg.height * this.zoom;
        this.wrapper.style.width = newWidth + 'px';
        this.wrapper.style.height = newHeight + 'px';
        this.wrapper.style.transform = `scale(${this.zoom})`;

        this.container.scrollLeft = worldX * this.zoom - offsetX;
        this.container.scrollTop = worldY * this.zoom - offsetY;
    },

    handleMouseDown(e) {
        if (e.button === 2 || e.button === 1 || (e.button === 0 && e.altKey)) {
            this.isPanning = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            this.wrapper.style.cursor = 'grabbing';
        } else if (e.button === 0) {
            this.isDrawing = true;
            this.applyBrushSettings();
            this.maskCtx.beginPath();
            const pos = this.getCanvasPos(e);
            this.maskCtx.moveTo(pos.x, pos.y);
            this.maskCtx.lineTo(pos.x, pos.y);
            this.maskCtx.stroke();
            this.maskCtx.beginPath(); // Start new path for smooth continuous lines
            this.maskCtx.moveTo(pos.x, pos.y);
        }
    },

    handleMouseMove(e) {
        if (this.isPanning) {
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            this.container.scrollLeft -= dx;
            this.container.scrollTop -= dy;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        } else if (this.isDrawing) {
            const pos = this.getCanvasPos(e);
            this.maskCtx.lineTo(pos.x, pos.y);
            this.maskCtx.stroke();
        }
    },

    stopAction() {
        if (this.isDrawing) {
            this.maskCtx.closePath();
            this.saveHistory();
        }
        this.isDrawing = false;
        this.isPanning = false;
        this.wrapper.style.cursor = 'crosshair';
    },

    getCanvasPos(e) {
        const rect = this.mask.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / this.zoom,
            y: (e.clientY - rect.top) / this.zoom
        };
    },

    applyBrushSettings() {
        const sizeInput = document.getElementById('brush-size');
        const size = sizeInput ? parseInt(sizeInput.value) : 20;
        this.maskCtx.lineWidth = size;
        this.maskCtx.lineCap = 'round';
        this.maskCtx.lineJoin = 'round';

        if (this.tool === 'eraser') {
            this.maskCtx.globalCompositeOperation = 'destination-out';
            this.maskCtx.strokeStyle = 'rgba(0,0,0,1)';
        } else {
            this.maskCtx.globalCompositeOperation = 'source-over';
            this.maskCtx.strokeStyle = '#000000'; // Always black brush
        }
    },
    saveHistory() {
        this.history.push(this.mask.toDataURL());
        if (this.history.length > 20) this.history.shift();
    },

    undo() {
        if (this.history.length <= 1) return;
        this.history.pop();
        const img = new Image();
        img.onload = () => {
            this.maskCtx.clearRect(0, 0, this.mask.width, this.mask.height);
            this.maskCtx.drawImage(img, 0, 0);
        };
        img.src = this.history[this.history.length - 1];
    }
};

// Initialize editor on load
document.addEventListener('DOMContentLoaded', () => {
    editor.init();
    setupDropZone();
});


/**
 * MODAL UI HELPERS
 */
function openEditor(id) {
    const data = analysisDataStore[id];
    if (!data) return;

    currentAnalysisId = id;
    const modal = document.getElementById('detailModal');
    document.getElementById('editor-filename').innerText = data.original_filename || 'Analysis';
    const badge = document.getElementById('editor-model-badge');
    badge.className = `model-badge model-badge--${data.seg_model}`;
    badge.innerText = data.seg_model.toUpperCase();

    loadDataIntoEditor(data);
    renderClassPalette(data);
    renderEditorMetrics(data);

    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
}

function loadDataIntoEditor(data) {
    const img = new Image();
    img.onload = () => {
        const w = img.width;
        const h = img.height;

        // Calculate minZoom to fit image in container
        const contW = editor.container.clientWidth || 800;
        const contH = editor.container.clientHeight || 600;

        // Slightly padding so it doesn't touch the absolute edges
        const fitScaleX = (contW - 40) / w;
        const fitScaleY = (contH - 40) / h;

        // Minimum zoom is enough to see the whole image
        editor.minZoom = Math.min(fitScaleX, fitScaleY, 1);
        editor.zoom = editor.minZoom;

        editor.wrapper.style.transform = `scale(${editor.zoom})`;

        [editor.bg, editor.mask, editor.ai].forEach(c => {
            c.width = w;
            c.height = h;
            c.style.width = w + 'px';
            c.style.height = h + 'px';
        });

        editor.wrapper.style.width = (w * editor.zoom) + 'px';
        editor.wrapper.style.height = (h * editor.zoom) + 'px';

        // Draw the original image on the background layer
        editor.bgCtx.drawImage(img, 0, 0);
        editor.maskCtx.clearRect(0, 0, w, h);
        editor.history = [];
        editor.saveHistory();

        // Draw the AI overlay directly from known annotated data
        // No redundant API call needed on load
        const aiImg = new Image();
        aiImg.onload = () => {
            editor.aiCtx.clearRect(0, 0, editor.ai.width, editor.ai.height);
            editor.aiCtx.drawImage(aiImg, 0, 0);
        }
        aiImg.src = `data:image/jpeg;base64,${data.annotated_b64}`;

        setTimeout(() => {
            const scaledW = w * editor.zoom;
            const scaledH = h * editor.zoom;
            editor.container.scrollLeft = Math.max(0, (scaledW - editor.container.clientWidth) / 2);
            editor.container.scrollTop = Math.max(0, (scaledH - editor.container.clientHeight) / 2);
        }, 50);

    };
    img.src = `data:image/jpeg;base64,${data.original_b64 || data.annotated_b64}`;
}

function closeEditor() {
    document.getElementById('detailModal').classList.remove('open');
    document.body.style.overflow = 'auto';
}

function setTool(t) {
    editor.tool = t;
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`tool-${t}`);
    if (btn) btn.classList.add('active');
}

function renderClassPalette(data) {
    // Left intentionally blank as per UX redesign requirements.
    // The visual class picker is removed. The brush defaults to black for erasing network predictions
    renderVisibilityToggles(data);
}

function renderVisibilityToggles(data) {
    const container = document.getElementById('editor-visibility-toggles');
    if (!container) return;
    container.innerHTML = '';

    data.available_classes.forEach(cls => {
        const clsL = cls.toLowerCase();
        if (activeLayers[clsL] === undefined) activeLayers[clsL] = true;
    });

    data.available_classes.forEach(cls => {
        const clsL = cls.toLowerCase();
        const color = data.class_colors ? data.class_colors[cls] : '#808080';
        const btn = document.createElement('button');
        btn.className = `tool-btn layer-toggle ${activeLayers[clsL] ? 'active' : ''}`;
        btn.style.borderLeft = `4px solid ${color}`;
        btn.style.opacity = activeLayers[clsL] ? '1' : '0.4';
        btn.innerText = cls.charAt(0).toUpperCase() + cls.slice(1);
        btn.onclick = () => toggleClassLayer(clsL, btn);
        container.appendChild(btn);
    });
}

function pickClassColor(c, btn) {
    editor.color = c;
    document.querySelectorAll('.class-pick-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    setTool('brush');
}

function updateBrushSize(v) {
    const valDisp = document.getElementById('brush-size-val');
    if (valDisp) valDisp.innerText = v;
}

async function updateEditorView() {
    if (!currentAnalysisId) return;
    const showBoxesCheck = document.getElementById('editor-show-boxes');
    const showBoxes = showBoxesCheck ? showBoxesCheck.checked : true;
    const selectedClasses = Object.keys(activeLayers).filter(k => activeLayers[k]);

    const response = await fetch('/api/filter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            analysis_id: currentAnalysisId,
            selected_classes: selectedClasses,
            show_boxes: showBoxes
        })
    });

    const resData = await response.json();
    if (resData.success) {
        const img = new Image();
        img.onload = () => {
            editor.aiCtx.clearRect(0, 0, editor.ai.width, editor.ai.height);
            editor.aiCtx.drawImage(img, 0, 0);
        };
        img.src = `data:image/jpeg;base64,${resData.annotated_b64}`;
        renderEditorMetrics(resData);
    }
}

async function toggleClassLayer(name, btn) {
    activeLayers[name] = !activeLayers[name];
    btn.classList.toggle('active');
    btn.style.opacity = activeLayers[name] ? '1' : '0.4';
    await updateEditorView();
}

async function saveMaskChanges() {
    if (!currentAnalysisId) return;
    const btn = document.querySelector('.editor-header .btn-primary');
    const oldText = btn.innerText;
    btn.innerText = 'Обработка AI...';
    btn.disabled = true;

    try {
        const composite = document.createElement('canvas');
        composite.width = editor.bg.width;
        composite.height = editor.bg.height;
        const ctx = composite.getContext('2d');
        ctx.drawImage(editor.bg, 0, 0);
        ctx.drawImage(editor.mask, 0, 0);

        const imageB64 = composite.toDataURL('image/jpeg', 0.95).split(',')[1];
        const data = analysisDataStore[currentAnalysisId];

        const response = await fetch('/api/recalculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                analysis_id: currentAnalysisId,
                image_b64: imageB64,
                seg_model: data.seg_model,
                show_boxes: true
            })
        });

        const res = await response.json();
        if (res.success) {
            analysisDataStore[currentAnalysisId] = {
                ...data,
                total_area: res.total_area,
                total_length: res.total_length,
                class_metrics: res.class_metrics,
                annotated_b64: res.annotated_b64,
                original_b64: res.original_b64
            };

            const card = document.querySelector(`.results-container[data-analysis-id="${currentAnalysisId}"]`);
            if (card) {
                card.querySelector('.main-img').src = `data:image/jpeg;base64,${res.annotated_b64}`;
                card.querySelector('.area-disp').innerText = `${res.total_area} мм²`;
                card.querySelector('.length-disp').innerText = `${res.total_length} мм`;
            }

            loadDataIntoEditor(analysisDataStore[currentAnalysisId]);
            renderEditorMetrics(analysisDataStore[currentAnalysisId]);
            alert('Нейросеть обновила данные!');
        }
    } catch (e) {
        console.error(e);
        alert('Ошибка: ' + e);
    } finally {
        btn.innerText = oldText;
        btn.disabled = false;
    }
}

function nextImage() {
    if (!currentAnalysisId || processedAnalysisIds.length <= 1) return;
    const idx = processedAnalysisIds.indexOf(currentAnalysisId);
    const nextIdx = (idx + 1) % processedAnalysisIds.length;
    openEditor(processedAnalysisIds[nextIdx]);
}

function prevImage() {
    if (!currentAnalysisId || processedAnalysisIds.length <= 1) return;
    const idx = processedAnalysisIds.indexOf(currentAnalysisId);
    const prevIdx = (idx - 1 + processedAnalysisIds.length) % processedAnalysisIds.length;
    openEditor(processedAnalysisIds[prevIdx]);
}


/**
 * GALLERY & CARD RENDERING
 */
function renderResult(data) {
    const template = document.getElementById('result-template');
    if (!template) return;

    const clone = template.content.cloneNode(true);
    const container = clone.querySelector('.results-container');
    container.dataset.analysisId = data.analysis_id;

    clone.querySelector('.class-name-disp').innerHTML =
        `<span style="word-break: break-all;">${data.original_filename}</span> ` +
        `<span class="model-badge model-badge--${data.seg_model}">${data.seg_model.toUpperCase()}</span>`;

    clone.querySelector('.length-disp').innerText = `${data.total_length} мм`;
    clone.querySelector('.area-disp').innerText = `${data.total_area} мм²`;

    const mainImg = clone.querySelector('.main-img');
    mainImg.src = `data:image/jpeg;base64,${data.annotated_b64}`;
    mainImg.onclick = () => openEditor(data.analysis_id);

    const gallery = document.getElementById("results-gallery");
    if (gallery) gallery.prepend(clone);
}

function renderEditorMetrics(data) {
    const grid = document.getElementById('editor-metrics');
    if (!grid) return;
    grid.innerHTML = '';

    // Fixed order to prevent UI hopping
    const order = ['корень', 'стебель', 'листок', 'колос'];
    const sortedKeys = Object.keys(data.class_metrics).sort((a, b) => {
        const ia = order.indexOf(a.toLowerCase());
        const ib = order.indexOf(b.toLowerCase());
        return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });

    for (const name of sortedKeys) {
        const m = data.class_metrics[name];
        const item = document.createElement('div');
        item.className = 'metric-item compact';
        item.style.padding = '0.75rem';
        item.style.background = 'rgba(255,255,255,0.03)';
        item.style.borderRadius = '8px';

        item.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <strong style="font-size:0.95rem; color:var(--text-primary); text-transform:capitalize;">${name}</strong>
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary); display:grid; grid-template-columns: 1fr 1fr; gap:0.5rem;">
                <span>📐 ${parseFloat(m.area).toFixed(1)} мм²</span>
                <span>📏 ${parseFloat(m.length).toFixed(1)} мм</span>
            </div>
        `;
        grid.appendChild(item);
    }
}

function openEditorFromCard(btn) {
    const id = btn.closest('.results-container').dataset.analysisId;
    openEditor(id);
}


/**
 * MAIN APP LOGIC (UPLOAD & EXPORT)
 */
function setupDropZone() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    if (dropZone && fileInput) {
        dropZone.onclick = (e) => {
            if (e.target.tagName !== 'INPUT') fileInput.click();
        };

        fileInput.onchange = () => { if (fileInput.files.length) processFiles(fileInput.files); };

        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add("drop-zone--over"); };
        dropZone.ondragleave = () => dropZone.classList.remove("drop-zone--over");
        dropZone.ondrop = (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
            dropZone.classList.remove("drop-zone--over");
        };
    }
}

async function processFiles(files) {
    if (!files || files.length === 0) return;

    const empty = document.getElementById("empty-state");
    const overlay = document.getElementById("loading-overlay");
    const lText = document.getElementById("loading-text");
    const batchActions = document.getElementById("batch-actions");
    const fileInput = document.getElementById("file-input");

    if (empty) empty.style.display = 'none';
    if (overlay) {
        overlay.style.display = 'flex';
        setTimeout(() => overlay.classList.add('active'), 10);
    }

    try {
        const modelInput = document.querySelector('input[name="seg_model"]:checked');
        const segModel = modelInput ? modelInput.value : 'unet';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (lText) lText.innerText = `Анализ: ${file.name} (${i + 1}/${files.length})...`;

            const fd = new FormData();
            fd.append('file', file);
            fd.append('seg_model', segModel);
            fd.append('show_boxes', 'on');

            try {
                const res = await fetch('/api/analyze', { method: 'POST', body: fd });
                const data = await res.json();

                if (data.success) {
                    analysisDataStore[data.analysis_id] = data;
                    processedAnalysisIds.push(data.analysis_id);
                    renderResult(data);
                } else {
                    console.error("Analysis failed for", file.name, data.error);
                    alert(`Ошибка при анализе ${file.name}: ${data.error || 'Неизвестная ошибка'}`);
                }
            } catch (e) {
                console.error("Fetch error for", file.name, e);
                alert(`Ошибка сети при загрузке ${file.name}`);
            }
        }

        if (batchActions && processedAnalysisIds.length > 0) {
            batchActions.style.display = 'flex';
        }
    } catch (err) {
        console.error("Critical error in processFiles:", err);
        alert("Произошла критическая ошибка при обработке файлов: " + err.message);
    } finally {
        if (overlay) {
            overlay.classList.remove('active');
            setTimeout(() => {
                overlay.style.display = 'none';
                if (lText) lText.innerText = "Анализ...";
            }, 400);
        }
        if (fileInput) fileInput.value = '';
    }
}

async function exportAll() {
    if (processedAnalysisIds.length === 0) return;
    const btn = document.querySelector('#batch-actions .btn-primary');
    const oldHTML = btn.innerHTML;
    btn.innerText = 'Сборка архива...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysis_ids: processedAnalysisIds })
        });
        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'plant_report.zip';
            a.click();
        }
    } catch (e) {
        console.error(e);
    } finally {
        btn.innerHTML = oldHTML;
        btn.disabled = false;
    }
}

function closeModal(event) {
    if (event.target.id === 'imgModal') {
        document.getElementById('imgModal').classList.remove('open');
    }
}
