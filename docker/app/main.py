"""FastAPI inference endpoint for aerial image classification."""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

from model import load_model, predict

app = FastAPI(
    title='Aerial Image Classifier',
    description='CSC3109 Group 30 — classifies aerial images into 4 categories',
    version='1.0.0',
)

CHECKPOINT = Path(os.getenv('MODEL_PATH', '/app/models/efficientnet_b0_best.pth'))
CONFIDENCE_THRESHOLD = 0.50  # below this → flag as unknown/uncertain
_model = None

CLASS_ICONS = {
    'oil_well': '🛢️',
    'solar_panel': '☀️',
    'transformer_station': '⚡',
    'wastewater_treatment_plant': '💧',
}

CLASS_COLORS = {
    'oil_well': '#e67e22',
    'solar_panel': '#f1c40f',
    'transformer_station': '#3498db',
    'wastewater_treatment_plant': '#2ecc71',
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aerial Image Classifier — CSC3109 Group 30</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f1923;
    color: #e8eaf0;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a2a3a 0%, #0d1f2d 100%);
    border-bottom: 2px solid #2196f3;
    padding: 24px 40px;
  }
  header h1 { font-size: 1.7rem; color: #fff; font-weight: 700; }
  header p  { color: #90a4ae; font-size: 0.9rem; margin-top: 4px; }

  .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }

  .card {
    background: #1a2535;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #243447;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .card h2 { font-size: 1.1rem; color: #90caf9; margin-bottom: 18px; font-weight: 600; }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #2196f3;
    border-radius: 10px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: #0f1923;
  }
  .upload-zone:hover, .upload-zone.dragover { border-color: #64b5f6; background: #162231; }
  .upload-zone .icon { font-size: 2.5rem; margin-bottom: 10px; }
  .upload-zone p { color: #78909c; font-size: 0.9rem; }
  .upload-zone strong { color: #90caf9; }
  #fileInput { display: none; }

  #preview-wrap { margin-top: 16px; display: none; text-align: center; }
  #preview-wrap img { max-height: 220px; border-radius: 8px; border: 2px solid #2196f3; }
  #filename-label { margin-top: 8px; color: #78909c; font-size: 0.85rem; }

  /* Button */
  #predictBtn {
    display: block; width: 100%; margin-top: 20px;
    padding: 14px; font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #1565c0, #0288d1);
    color: #fff; border: none; border-radius: 8px;
    cursor: pointer; transition: opacity 0.2s;
  }
  #predictBtn:disabled { opacity: 0.4; cursor: not-allowed; }
  #predictBtn:hover:not(:disabled) { opacity: 0.9; }

  /* Spinner */
  .spinner {
    display: none; width: 28px; height: 28px; margin: 20px auto;
    border: 3px solid #243447; border-top-color: #2196f3;
    border-radius: 50%; animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result */
  #result-card { display: none; }

  .result-top {
    display: flex; align-items: center; gap: 16px; margin-bottom: 24px;
  }
  .result-icon { font-size: 3rem; }
  .result-class { font-size: 1.6rem; font-weight: 700; text-transform: replace; }
  .result-confidence { font-size: 0.9rem; color: #90a4ae; margin-top: 4px; }

  .warning-box {
    background: #3e2723; border: 1px solid #e64a19;
    border-radius: 8px; padding: 12px 16px;
    color: #ffccbc; font-size: 0.9rem; margin-bottom: 20px;
  }
  .warning-box strong { color: #ff7043; }

  /* Probability bars */
  .prob-row { margin-bottom: 14px; }
  .prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.85rem; color: #b0bec5; margin-bottom: 5px;
  }
  .prob-label span:last-child { color: #e8eaf0; font-weight: 600; }
  .bar-bg {
    background: #0f1923; border-radius: 4px; height: 10px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
  }

  /* JSON output */
  details { margin-top: 20px; }
  summary {
    cursor: pointer; color: #64b5f6; font-size: 0.9rem;
    user-select: none; padding: 6px 0;
  }
  summary:hover { color: #90caf9; }
  pre {
    background: #0f1923; border: 1px solid #243447;
    border-radius: 8px; padding: 16px;
    font-size: 0.8rem; color: #a5d6a7;
    overflow-x: auto; margin-top: 10px;
    white-space: pre-wrap; word-break: break-word;
  }

  /* Footer */
  footer {
    text-align: center; padding: 30px;
    color: #455a64; font-size: 0.8rem;
  }
</style>
</head>
<body>

<header>
  <h1>Aerial Image Classifier</h1>
  <p>CSC3109 Machine Learning &mdash; Group 30 &nbsp;|&nbsp; EfficientNet-B0 &nbsp;|&nbsp; 4 Categories</p>
</header>

<div class="container">

  <!-- Upload card -->
  <div class="card">
    <h2>Upload Aerial Image</h2>
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">🛰️</div>
      <p><strong>Click to choose an image</strong> or drag and drop here</p>
      <p>Supported: JPG, PNG, BMP &nbsp;&bull;&nbsp; Any size</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div id="preview-wrap">
      <img id="preview-img" src="" alt="preview">
      <div id="filename-label"></div>
    </div>

    <button id="predictBtn" disabled>Classify Image</button>
    <div class="spinner" id="spinner"></div>
  </div>

  <!-- Result card -->
  <div class="card" id="result-card">
    <h2>Classification Result</h2>

    <div class="warning-box" id="warning-box" style="display:none">
      <strong>⚠️ Low Confidence Warning</strong><br>
      The model's confidence is below 50%. This image may not belong to any of the
      4 trained categories (oil well, solar panel, transformer station, wastewater plant).
      The prediction below may be unreliable.
    </div>

    <div class="result-top">
      <div class="result-icon" id="res-icon"></div>
      <div>
        <div class="result-class" id="res-class"></div>
        <div class="result-confidence" id="res-conf"></div>
      </div>
    </div>

    <div id="prob-bars"></div>

    <details>
      <summary>Show raw JSON response</summary>
      <pre id="json-out"></pre>
    </details>
  </div>

</div>

<footer>CSC3109 AY2025/26 &mdash; Group 30 &nbsp;|&nbsp; Powered by FastAPI + PyTorch</footer>

<script>
const ICONS = {
  oil_well: '🛢️',
  solar_panel: '☀️',
  transformer_station: '⚡',
  wastewater_treatment_plant: '💧'
};
const COLORS = {
  oil_well: '#e67e22',
  solar_panel: '#f1c40f',
  transformer_station: '#3498db',
  wastewater_treatment_plant: '#2ecc71'
};
const LABELS = {
  oil_well: 'Oil Well',
  solar_panel: 'Solar Panel',
  transformer_station: 'Transformer Station',
  wastewater_treatment_plant: 'Wastewater Treatment Plant'
};

const fileInput   = document.getElementById('fileInput');
const uploadZone  = document.getElementById('uploadZone');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const filenameLabel = document.getElementById('filename-label');
const predictBtn  = document.getElementById('predictBtn');
const spinner     = document.getElementById('spinner');
const resultCard  = document.getElementById('result-card');

// Drag-and-drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; handleFile(); }
});

fileInput.addEventListener('change', handleFile);

function handleFile() {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  filenameLabel.textContent = file.name;
  previewWrap.style.display = 'block';
  predictBtn.disabled = false;
  resultCard.style.display = 'none';
}

predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  predictBtn.disabled = true;
  spinner.style.display = 'block';
  resultCard.style.display = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      alert('Error: ' + (data.detail || 'Unknown error'));
      return;
    }

    renderResult(data);
  } catch (e) {
    alert('Network error: ' + e.message);
  } finally {
    predictBtn.disabled = false;
    spinner.style.display = 'none';
  }
});

function renderResult(data) {
  const cls   = data.predicted_class;
  const conf  = data.confidence;
  const probs = data.probabilities;

  document.getElementById('res-icon').textContent  = ICONS[cls] || '❓';
  document.getElementById('res-class').textContent = LABELS[cls] || cls;
  document.getElementById('res-class').style.color = COLORS[cls] || '#fff';
  document.getElementById('res-conf').textContent  =
    `Confidence: ${(conf * 100).toFixed(2)}%`;

  document.getElementById('warning-box').style.display = conf < 0.50 ? 'block' : 'none';

  // Probability bars — sorted descending
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const barsHtml = sorted.map(([name, prob]) => `
    <div class="prob-row">
      <div class="prob-label">
        <span>${ICONS[name] || ''} ${LABELS[name] || name}</span>
        <span>${(prob * 100).toFixed(2)}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill"
             style="width:${(prob*100).toFixed(2)}%; background:${COLORS[name] || '#607d8b'}">
        </div>
      </div>
    </div>`).join('');
  document.getElementById('prob-bars').innerHTML = barsHtml;

  document.getElementById('json-out').textContent = JSON.stringify(data, null, 2);

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>
</body>
</html>
"""


@app.on_event('startup')
def startup():
    global _model
    if not CHECKPOINT.exists():
        raise RuntimeError(f'Checkpoint not found: {CHECKPOINT}')
    _model = load_model(CHECKPOINT)
    print(f'Model loaded from {CHECKPOINT}')


@app.get('/', response_class=HTMLResponse)
def root():
    return HTML_PAGE


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': _model is not None}


@app.get('/classes')
def classes():
    return {
        'classes': [
            'oil_well',
            'solar_panel',
            'transformer_station',
            'wastewater_treatment_plant',
        ]
    }


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image (JPEG, PNG, etc.)')

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail='Could not decode image file')

    result = predict(_model, image)

    if result['confidence'] < CONFIDENCE_THRESHOLD:
        result['warning'] = (
            'Low confidence — image may not belong to any of the 4 trained categories. '
            f'Max confidence was {result["confidence"]:.2%}, threshold is {CONFIDENCE_THRESHOLD:.0%}.'
        )

    return JSONResponse(content={
        'filename': file.filename,
        **result,
    })
