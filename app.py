"""
BananaLens — Streamlit Edition (single file, no Flask)
========================================================
Semua isi index.html, style.css, dan main.js DITEMPEL APA ADANYA
(verbatim, tidak diubah sedikitpun) sebagai string Python di bawah.

Karena Streamlit tidak punya routing custom seperti Flask
(@app.route), fetch('/predict') di main.js dijembatani dengan cara
menyuntikkan handler Tornado (server internal yang memang dipakai
Streamlit di baliknya) ke server Streamlit yang sedang jalan.
Dengan begitu /predict, /static/css/style.css, /static/js/main.js,
dan /static/uploads/* tetap bisa diakses persis seperti versi Flask,
tanpa perlu Flask sama sekali, dan tanpa mengubah HTML/CSS/JS aslinya.

Catatan: teknik injeksi route ini mengandalkan struktur internal
Streamlit (streamlit.web.server.server.Server) yang bisa berubah
antar versi. Sudah dibuat dengan fallback, tapi kalau suatu saat
gagal (misal setelah upgrade Streamlit), pesan error akan muncul
di halaman.

Jalankan dengan:
    streamlit run app.py

Install dulu:
    pip install streamlit tensorflow pillow numpy jinja2 werkzeug tornado
"""

import os
import json
import numpy as np
import tornado.web
from jinja2 import Template
from werkzeug.utils import secure_filename
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_USE_LEGACY_KERAS'] = '0'

UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# === Label mapping (folder name -> label tampilan) ===
LABEL_MAP = {
    'matang':          'Matang',
    'mentah':          'Mentah',
    'setengah-matang': 'Setengah Matang',
}

# === Data nutrisi per kelas folder (per 100g) ===
NUTRISI = {
    'matang': {
        'Energi':            '110 kkal / 100g',
        'Karbohidrat Total': '28,0 g / 100g',
        'Pati':              '~1,0 g / 100g',
        'Resistant Starch':  '<1,0 g / 100g',
        'Total Gula':        '~15,0 g / 100g',
        'Serat Diet':        '4,5 g / 100g',
        'Indeks Glikemik':   '51',
    },
    'mentah': {
        'Energi':            '89 kkal / 100g',
        'Karbohidrat Total': '22,8 g / 100g',
        'Pati':              '21,0 g / 100g',
        'Resistant Starch':  '~15,0 g / 100g',
        'Total Gula':        '<5,0 g / 100g',
        'Serat Diet':        '18,0 g / 100g',
        'Indeks Glikemik':   '42',
    },
    'setengah-matang': {
        'Energi':            '99 kkal / 100g',
        'Karbohidrat Total': '25,2 g / 100g',
        'Pati':              '~10,0 g / 100g',
        'Resistant Starch':  '~5,0 g / 100g',
        'Total Gula':        '~10,0 g / 100g',
        'Serat Diet':        '~9,0 g / 100g',
        'Indeks Glikemik':   '46',
    },
}

model = None
class_indices = None


def load_banana_model():
    global model, class_indices
    model_path = os.path.join(MODEL_FOLDER, 'pisang.keras')
    if not os.path.exists(model_path):
        print(f"⚠️  Model tidak ditemukan di {model_path}")
        return False
    try:
        model = load_model(model_path, compile=False)
        class_indices = {'matang': 0, 'mentah': 1, 'setengah-matang': 2}
        print("✅ Model berhasil dimuat!")
        return True
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath):
    img = Image.open(filepath).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_idx = int(np.argmax(preds))
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(preds[predicted_idx]) * 100
    all_probs = {idx_to_class[i]: float(preds[i]) * 100 for i in range(len(preds))}
    return predicted_class, confidence, all_probs


# ============================================================
# ASLI: index.html — VERBATIM, TIDAK DIUBAH SEDIKITPUN
# ============================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BananaLens — Klasifikasi Kematangan Pisang</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css" />
  <link rel="stylesheet" href="/static/css/style.css" />
</head>
<body>
<div class="app">

  <div class="header">
    <div class="logo-row">
      <div class="logo-icon-wrap">🍌</div>
      <div class="logo-name">Banana<span>Lens</span></div>
    </div>
    <div class="header-desc">Klasifikasi Kematangan Pisang Untuk Menentukan Kandungan Nutrisi — MobileNetV2 CNN</div>
    {% if not model_loaded %}
    <div class="model-warning">⚠️ Model belum dimuat — letakkan <strong>banana.h5</strong> di folder <code>model/</code></div>
    {% endif %}
  </div>

  <!-- TAB -->
  <div class="tab-row">
    <button class="tab-btn active" id="tabUpload"><i class="ti ti-upload"></i> Upload</button>
    <button class="tab-btn" id="tabPhoto"><i class="ti ti-camera"></i> Kamera Foto</button>
    <!-- <button class="tab-btn" id="tabRealtime"><i class="ti ti-scan"></i> Real-time</button> -->
  </div>

  <!-- ===== PANEL UPLOAD ===== -->
  <div class="panel" id="panelUpload">
    <div class="step-label">Upload Gambar Pisang</div>
    <div class="upload-area" id="dropZone">
      <div class="upload-big-icon">📁</div>
      <h3>Seret &amp; Lepas Gambar Pisang</h3>
      <p>atau klik tombol di bawah untuk memilih file</p>
      <button class="btn-primary" onclick="document.getElementById('fileInput').click()">
        <i class="ti ti-upload"></i> Pilih Gambar
      </button>
      <p class="upload-fmt">JPG · PNG · WEBP · Maks 16MB</p>
      <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.webp" style="display:none" />
    </div>
    <div class="preview-card" id="previewCard">
      <div class="preview-inner">
        <img id="previewImg" src="" alt="Preview" />
        <div class="preview-info">
          <h4 id="previewName"></h4>
          <p id="previewSize"></p>
          <div class="preview-actions">
            <button class="btn-primary" id="btnAnalyze"><i class="ti ti-sparkles"></i> Analisis</button>
            <button class="btn-ghost" id="btnCancel">Ganti</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== PANEL KAMERA FOTO ===== -->
  <div class="panel" id="panelPhoto" style="display:none">
    <div class="step-label">Kamera — Ambil Foto</div>
    <div class="cam-card">
      <div class="cam-placeholder" id="photoPlaceholder">
        <div class="spinner-ring"></div>
        <p>Membuka kamera…</p>
      </div>
      <video id="photoVideo" autoplay playsinline style="display:none;width:100%;border-radius:10px;max-height:340px;object-fit:cover;background:#000"></video>
      <canvas id="photoCanvas" style="display:none"></canvas>
      <div class="cam-controls" id="photoControls" style="display:none">
        <button class="btn-capture" id="btnCapture"><i class="ti ti-camera"></i> Ambil Foto</button>
        <button class="btn-ghost" id="btnStopPhoto"><i class="ti ti-player-stop"></i> Stop Kamera</button>
      </div>
    </div>
    <div class="preview-card" id="photoPreviewCard" style="display:none">
      <div class="preview-inner">
        <img id="photoPreviewImg" src="" alt="Hasil foto" />
        <div class="preview-info">
          <h4>Foto dari Kamera</h4>
          <p id="photoPreviewTime"></p>
          <div class="preview-actions">
            <button class="btn-primary" id="btnPhotoAnalyze"><i class="ti ti-sparkles"></i> Analisis</button>
            <button class="btn-ghost" id="btnPhotoRetake"><i class="ti ti-refresh"></i> Foto Ulang</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== PANEL REAL-TIME =====
  <div class="panel" id="panelRealtime" style="display:none">
    <div class="step-label">Kamera — Scan Real-time</div>

    <div class="interval-row">
      <span class="interval-label"><i class="ti ti-clock"></i> Interval scan:</span>
      <div class="interval-btns">
        <button class="interval-btn" data-ms="1000">1 dtk</button>
        <button class="interval-btn active" data-ms="2000">2 dtk</button>
        <button class="interval-btn" data-ms="3000">3 dtk</button>
      </div>
    </div>

    <div class="rt-layout">
      <div class="rt-left">
        <div class="cam-card rt-cam-card">
          <div class="cam-placeholder" id="rtPlaceholder">
            <div class="spinner-ring"></div>
            <p>Membuka kamera…</p>
          </div>
          <div class="rt-video-wrap" id="rtVideoWrap" style="display:none">
            <video id="rtVideo" autoplay playsinline style="width:100%;border-radius:10px;object-fit:cover;background:#000;display:block"></video>
            <canvas id="rtCanvas" style="display:none"></canvas>
            <div class="scan-overlay">
              <div class="scan-line"></div>
              <div class="scan-corner tl"></div>
              <div class="scan-corner tr"></div>
              <div class="scan-corner bl"></div>
              <div class="scan-corner br"></div>
            </div>
            <div class="rt-status-badge" id="rtStatusBadge">
              <span class="pulse-dot scanning"></span> Memulai scan…
            </div>
          </div>
          <div class="cam-controls" id="rtControls" style="display:none">
            <button class="btn-rt-pause" id="btnRtPause"><i class="ti ti-player-pause"></i> Pause</button>
            <button class="btn-rt-resume" id="btnRtResume" style="display:none"><i class="ti ti-player-play"></i> Resume</button>
            <button class="btn-ghost" id="btnStopRt"><i class="ti ti-player-stop"></i> Stop</button>
          </div>
        </div>
      </div>

      <div class="rt-right" id="rtRight">
        <div class="rt-result-empty" id="rtResultEmpty">
          <i class="ti ti-eye-off" style="font-size:32px;color:var(--muted);margin-bottom:8px"></i>
          <p>Arahkan pisang ke kamera</p>
        </div>
        <div class="rt-result-card" id="rtResultCard" style="display:none">
          <div class="rt-thumb-row">
            <img id="rtThumb" src="" alt="frame" />
            <div class="rt-kelas-info">
              <div class="rt-kelas" id="rtKelas">—</div>
              <span class="badge" id="rtBadge">—</span>
              <div class="rt-conf-row">
                <div class="conf-bar-outer"><div class="conf-bar-inner" id="rtConfBar"></div></div>
                <span class="conf-pct" id="rtConfPct">—</span>
              </div>
            </div>
          </div>
          <div class="rt-prob-wrap">
            <div class="prob-section-title">Probabilitas</div>
            <div class="prob-grid" id="rtProbGrid"></div>
          </div>
          <div class="rt-nutrisi-wrap">
            <div class="rt-nutrisi-title" id="rtNutrisiTitle"></div>
            <div class="nutrisi-grid" id="rtNutrisiGrid"></div>
          </div>
          <div class="rt-timestamp">🕐 Update: <span id="rtTimestamp">—</span></div>
        </div>
      </div>
    </div>
  </div>

  ===== RESULT (Upload & Foto) ===== -->
  <div class="result-wrap" id="resultWrap">
    <div class="step-label" style="margin-top:8px">Hasil Klasifikasi</div>
    <div class="loading-card" id="loadingCard">
      <div class="spinner-ring"></div>
      <p>Menganalisis gambar pisang…</p>
    </div>
    <div class="result-card-main" id="resultMain">
      <div class="result-top">
        <div class="result-top-inner">
          <div class="result-img-wrap">
            <img id="resultImg" src="" alt="Hasil" />
          </div>
          <div class="result-info">
            <div class="result-kelas" id="resultKelas">—</div>
            <div class="result-badge-row"><span class="badge" id="resultBadge">—</span></div>
            <div class="conf-label">Kepercayaan Model</div>
            <div class="conf-row">
              <div class="conf-bar-outer"><div class="conf-bar-inner" id="confBar"></div></div>
              <div class="conf-pct" id="confPct">—</div>
            </div>
          </div>
        </div>
        <div class="prob-section">
          <div class="prob-section-title">Distribusi Probabilitas</div>
          <div class="prob-grid" id="probGrid"></div>
        </div>
      </div>
      <div class="info-card" style="margin-bottom:14px">
        <div class="info-card-title"><i class="ti ti-leaf"></i> Kandungan Nutrisi <span id="nutrisiSubtitle"></span></div>
        <div class="nutrisi-grid" id="nutrisiGrid"></div>
      </div>
      <div class="bottom-action">
        <button class="btn-ghost" id="btnReset">← Analisis Lagi</button>
      </div>
    </div>
    <div class="error-card" id="errorCard">
      <div class="error-icon">⚠️</div>
      <p id="errorMsg"></p>
      <button class="btn-ghost" id="btnResetErr">← Coba Lagi</button>
    </div>
  </div>

</div>
<script src="/static/js/main.js"></script>
</body>
</html>
"""

# ============================================================
# ASLI: style.css — VERBATIM, TIDAK DIUBAH SEDIKITPUN
# ============================================================
STYLE_CSS = r"""*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --yellow:#F5A623; --yellow-light:#FFF8E7; --yellow-border:#FFD77A; --yellow-deep:#C47C00;
  --bg:#FAFAF8; --surface:#FFFFFF; --border:rgba(0,0,0,0.08);
  --text:#1A1208; --muted:#6B7280;
  --ff:'Sora',system-ui,sans-serif; --ffi:'Inter',system-ui,sans-serif;
  --r-lg:16px; --r-md:10px; --r-sm:8px;
}
body { font-family:var(--ffi); background:var(--bg); color:var(--text); min-height:100vh; }
.app { max-width:960px; margin:0 auto; padding:28px 20px 56px; }

/* HEADER */
.header { text-align:center; margin-bottom:24px; }
.logo-row { display:inline-flex; align-items:center; gap:10px; margin-bottom:6px; }
.logo-icon-wrap { width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,#FFD600,#F5A623); display:flex; align-items:center; justify-content:center; font-size:20px; }
.logo-name { font-family:var(--ff); font-size:22px; font-weight:700; letter-spacing:-0.5px; }
.logo-name span { color:var(--yellow-deep); }
.header-desc { font-size:13px; color:var(--muted); }
.model-warning { display:inline-block; margin-top:10px; background:#FFF8E7; border:0.5px solid var(--yellow-border); border-radius:var(--r-sm); padding:7px 14px; font-size:12px; color:#7A5800; }
.model-warning code { background:rgba(0,0,0,0.07); border-radius:4px; padding:1px 5px; }

/* TABS */
.tab-row { display:flex; gap:6px; margin-bottom:20px; background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:4px; }
.tab-btn { flex:1; display:flex; align-items:center; justify-content:center; gap:6px; padding:9px 14px; border:none; border-radius:12px; font-family:var(--ff); font-size:13px; font-weight:500; cursor:pointer; color:var(--muted); background:transparent; transition:all 0.2s; }
.tab-btn i { font-size:16px; }
.tab-btn.active { background:#1A1208; color:#fff; }
.tab-btn:not(.active):hover { background:var(--bg); color:var(--text); }

/* STEP LABEL */
.step-label { font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); border-left:2px solid var(--yellow); padding-left:8px; margin-bottom:12px; }

/* UPLOAD */
.upload-area { padding:36px 24px; border:1.5px dashed #D1C9B0; border-radius:var(--r-lg); text-align:center; cursor:pointer; background:var(--yellow-light); transition:border-color 0.2s,background 0.2s; margin-bottom:14px; }
.upload-area:hover,.upload-area.dragover { border-color:var(--yellow); background:#FFF3D0; }
.upload-big-icon { font-size:32px; margin-bottom:8px; }
.upload-area h3 { font-family:var(--ff); font-size:15px; font-weight:600; margin-bottom:4px; }
.upload-area p { font-size:13px; color:var(--muted); margin-bottom:14px; }
.upload-fmt { font-size:11px; color:#A89070; letter-spacing:0.05em; margin-top:10px; margin-bottom:0 !important; }

/* CAMERA SOURCE */
.cam-source-row { display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
.cam-src-btn { flex:1; min-width:100px; display:flex; align-items:center; justify-content:center; gap:6px; padding:8px 12px; border:0.5px solid var(--border); border-radius:var(--r-md); font-family:var(--ffi); font-size:12px; font-weight:500; cursor:pointer; background:var(--surface); color:var(--muted); transition:all 0.2s; }
.cam-src-btn i { font-size:15px; }
.cam-src-btn.active { background:#1A1208; color:#fff; border-color:#1A1208; }
.cam-src-btn:not(.active):hover { border-color:var(--yellow); color:var(--text); }

/* INTERVAL */
.interval-row { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.interval-label { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:5px; }
.interval-btns { display:flex; gap:6px; }
.interval-btn { padding:5px 12px; border:0.5px solid var(--border); border-radius:100px; font-size:12px; cursor:pointer; background:var(--surface); color:var(--muted); transition:all 0.15s; }
.interval-btn.active { background:var(--yellow); color:#1A1208; border-color:var(--yellow); font-weight:600; }

/* CAM CARD */
.cam-card { background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:14px; margin-bottom:10px; overflow:hidden; }
.rt-cam-card { margin-bottom:0; }
.cam-placeholder { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:160px; color:var(--muted); }
.cam-placeholder p { font-size:13px; }
.cam-controls { display:flex; gap:8px; margin-top:12px; padding-top:12px; border-top:0.5px solid var(--border); flex-wrap:wrap; }

/* RT VIDEO WRAP */
.rt-video-wrap { position:relative; }

/* SCAN OVERLAY */
.scan-overlay { position:absolute; inset:0; pointer-events:none; border-radius:10px; overflow:hidden; }
.scan-line { position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#F5A623,transparent); animation:scanMove 2s linear infinite; top:0; }
@keyframes scanMove { 0%{top:0;opacity:1} 90%{top:100%;opacity:1} 100%{top:100%;opacity:0} }
.scan-corner { position:absolute; width:18px; height:18px; border-color:#F5A623; border-style:solid; }
.scan-corner.tl { top:8px; left:8px; border-width:2px 0 0 2px; border-radius:3px 0 0 0; }
.scan-corner.tr { top:8px; right:8px; border-width:2px 2px 0 0; border-radius:0 3px 0 0; }
.scan-corner.bl { bottom:8px; left:8px; border-width:0 0 2px 2px; border-radius:0 0 0 3px; }
.scan-corner.br { bottom:8px; right:8px; border-width:0 2px 2px 0; border-radius:0 0 3px 0; }

/* STATUS BADGE */
.rt-status-badge { position:absolute; bottom:10px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,0.65); color:#fff; font-size:11px; border-radius:100px; padding:4px 12px; display:flex; align-items:center; gap:6px; white-space:nowrap; backdrop-filter:blur(4px); }
.pulse-dot { width:7px; height:7px; border-radius:50%; background:#4CAF50; flex-shrink:0; animation:pulseDot 1.2s ease-in-out infinite; }
.pulse-dot.scanning { background:#F5A623; }
.pulse-dot.paused { background:#aaa; animation:none; }
@keyframes pulseDot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }

/* RT LAYOUT */
.rt-layout { display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }
@media(max-width:600px) { .rt-layout { grid-template-columns:1fr; } }

/* RT RESULT */
.rt-right { background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:14px; min-height:200px; }
.rt-result-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:180px; color:var(--muted); }
.rt-result-empty p { font-size:13px; }
.rt-result-card { display:none; }
.rt-thumb-row { display:flex; gap:10px; align-items:center; margin-bottom:12px; padding-bottom:12px; border-bottom:0.5px solid var(--border); }
.rt-thumb-row img { width:64px; height:64px; border-radius:8px; object-fit:cover; border:0.5px solid var(--border); flex-shrink:0; }
.rt-kelas-info { flex:1; min-width:0; }
.rt-kelas { font-family:var(--ff); font-size:18px; font-weight:700; margin-bottom:4px; }
.rt-conf-row { display:flex; align-items:center; gap:8px; margin-top:6px; }
.rt-prob-wrap { margin-bottom:12px; padding-bottom:12px; border-bottom:0.5px solid var(--border); }
.rt-nutrisi-title { font-family:var(--ff); font-size:12px; font-weight:600; margin-bottom:8px; color:var(--muted); }
.rt-timestamp { font-size:10px; color:var(--muted); margin-top:10px; padding-top:8px; border-top:0.5px solid var(--border); }

/* PAUSE/RESUME BUTTONS */
.btn-rt-pause { display:inline-flex; align-items:center; gap:6px; background:#FFF3D0; color:#8B5800; border:0.5px solid var(--yellow-border); border-radius:100px; padding:8px 16px; font-family:var(--ffi); font-size:12px; font-weight:500; cursor:pointer; transition:all 0.15s; }
.btn-rt-pause:hover { background:var(--yellow-light); }
.btn-rt-resume { display:inline-flex; align-items:center; gap:6px; background:#E8F5E9; color:#27500A; border:0.5px solid #9FE1CB; border-radius:100px; padding:8px 16px; font-family:var(--ffi); font-size:12px; font-weight:500; cursor:pointer; transition:all 0.15s; }
.btn-rt-resume:hover { background:#C8E6C9; }

/* BUTTONS */
.btn-primary { background:#1A1208; color:#fff; border:none; border-radius:100px; padding:10px 20px; font-family:var(--ff); font-size:13px; font-weight:600; cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:opacity 0.15s,transform 0.15s; }
.btn-primary:hover { opacity:0.82; transform:translateY(-1px); }
.btn-primary i { font-size:15px; }
.btn-capture { flex:1; display:flex; align-items:center; justify-content:center; gap:8px; background:linear-gradient(135deg,#FFD600,#F5A623); color:#1A1208; border:none; border-radius:100px; padding:11px 20px; font-family:var(--ff); font-size:13px; font-weight:600; cursor:pointer; transition:opacity 0.15s,transform 0.15s; }
.btn-capture:hover { opacity:0.88; transform:translateY(-1px); }
.btn-ghost { background:transparent; color:var(--muted); border:0.5px solid var(--border); border-radius:100px; padding:8px 18px; font-family:var(--ffi); font-size:12px; cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:border-color 0.15s,color 0.15s; }
.btn-ghost:hover { border-color:var(--yellow); color:var(--text); }

/* PREVIEW CARD */
.preview-card { display:none; background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:14px; margin-bottom:14px; }
.preview-inner { display:flex; gap:14px; align-items:center; }
.preview-inner img,.preview-img-wrap img { width:80px; height:80px; border-radius:var(--r-md); object-fit:cover; border:0.5px solid var(--border); display:block; flex-shrink:0; }
.preview-info { flex:1; min-width:0; }
.preview-info h4 { font-family:var(--ff); font-size:13px; font-weight:600; margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.preview-info p { font-size:12px; color:var(--muted); margin-bottom:10px; }
.preview-actions { display:flex; gap:8px; flex-wrap:wrap; }

/* RESULT */
.result-wrap { display:none; }
.loading-card { display:none; background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:32px; text-align:center; }
.spinner-ring { width:36px; height:36px; border-radius:50%; border:3px solid #EDE2C8; border-top-color:var(--yellow); animation:spin 0.75s linear infinite; margin:0 auto 10px; }
@keyframes spin { to{transform:rotate(360deg)} }
.loading-card p { font-size:14px; color:var(--muted); }
.result-card-main { display:none; }
.result-top { background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:18px; margin-bottom:10px; }
.result-top-inner { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:14px; padding-bottom:14px; border-bottom:0.5px solid var(--border); }
.result-img-wrap img { width:78px; height:78px; border-radius:var(--r-md); object-fit:cover; border:0.5px solid var(--border); display:block; }
.result-info { flex:1; min-width:140px; }
.result-kelas { font-family:var(--ff); font-size:20px; font-weight:700; margin-bottom:5px; }
.result-badge-row { margin-bottom:8px; }
.badge { display:inline-flex; align-items:center; gap:4px; font-size:11px; font-weight:600; border-radius:20px; padding:3px 10px; letter-spacing:0.03em; }
.badge-matang   { background:#FFF8E7; color:#8B5800;  border:0.5px solid #FFD77A; }
.badge-mentah   { background:#EAF3DE; color:#27500A;  border:0.5px solid #9FE1CB; }
.badge-setengah { background:#F9FBE7; color:#3B6D11;  border:0.5px solid #C0DD97; }
.conf-label { font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; }
.conf-row { display:flex; align-items:center; gap:8px; }
.conf-bar-outer { flex:1; height:5px; background:#EDE2C8; border-radius:100px; overflow:hidden; }
.conf-bar-inner { height:100%; background:var(--yellow); border-radius:100px; transition:width 1s cubic-bezier(.4,0,.2,1); width:0%; }
.conf-pct { font-family:var(--ff); font-size:13px; font-weight:700; color:var(--yellow-deep); min-width:40px; text-align:right; }
.prob-section-title { font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.prob-grid { display:flex; flex-direction:column; gap:7px; }
.prob-item-row { display:flex; align-items:center; gap:8px; }
.prob-name { font-size:12px; color:var(--muted); min-width:110px; }
.prob-bar-outer { flex:1; height:4px; background:#EDE2C8; border-radius:100px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:100px; transition:width 1s cubic-bezier(.4,0,.2,1); width:0%; }
.fill-matang{background:#F5A623} .fill-mentah{background:#5CB85C} .fill-setengah{background:#A8C34F}
.prob-val { font-size:12px; font-weight:500; min-width:38px; text-align:right; }
.info-card { background:var(--surface); border:0.5px solid var(--border); border-radius:var(--r-lg); padding:16px; }
.info-card-title { font-family:var(--ff); font-size:13px; font-weight:600; margin-bottom:12px; display:flex; align-items:center; gap:6px; }
.info-card-title i { color:var(--yellow); font-size:15px; }
.info-card-title span { font-size:11px; font-weight:400; color:var(--muted); margin-left:4px; }
.nutrisi-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
@media(max-width:480px){.nutrisi-grid{grid-template-columns:1fr}}
.nutrisi-item { background:var(--bg); border:0.5px solid var(--border); border-radius:var(--r-sm); padding:7px 10px; transition:border-color 0.15s; }
.nutrisi-item:hover{border-color:var(--yellow-border)}
.nkey { font-size:10px; text-transform:uppercase; letter-spacing:0.05em; color:var(--muted); margin-bottom:2px; }
.nval { font-size:12px; font-weight:500; }
.bottom-action { margin-top:12px; }
.error-card { display:none; background:#FFF5F5; border:0.5px solid #F7C1C1; border-radius:var(--r-lg); padding:28px; text-align:center; }
.error-icon { font-size:1.8rem; margin-bottom:8px; }
#errorMsg { font-size:13px; color:#A32D2D; margin-bottom:12px; }
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeUp 0.3s ease both}
"""

# ============================================================
# ASLI: main.js — VERBATIM, TIDAK DIUBAH SEDIKITPUN
# ============================================================
MAIN_JS = r"""// ============================================================
// SHARED RESULT ELEMENTS
// ============================================================
const resultWrap  = document.getElementById('resultWrap');
const loadingCard = document.getElementById('loadingCard');
const resultMain  = document.getElementById('resultMain');
const resultImg   = document.getElementById('resultImg');
const resultKelas = document.getElementById('resultKelas');
const resultBadge = document.getElementById('resultBadge');
const confBar     = document.getElementById('confBar');
const confPct     = document.getElementById('confPct');
const probGrid    = document.getElementById('probGrid');
const nutrisiGrid = document.getElementById('nutrisiGrid');
const nutrisiSub  = document.getElementById('nutrisiSubtitle');
const errorCard   = document.getElementById('errorCard');
const errorMsg    = document.getElementById('errorMsg');

// ============================================================
// TAB SWITCHER — hanya upload & photo
// ============================================================
const tabs   = { upload: document.getElementById('tabUpload'), photo: document.getElementById('tabPhoto') };
const panels = { upload: document.getElementById('panelUpload'), photo: document.getElementById('panelPhoto') };

function switchTab(name) {
  Object.keys(tabs).forEach(k => {
    tabs[k].classList.toggle('active', k === name);
    panels[k].style.display = k === name ? 'block' : 'none';
  });
  hideResult();
  if (name !== 'photo') stopPhotoCamera();
  if (name === 'photo') startPhotoCamera();
}

document.getElementById('tabUpload').addEventListener('click', () => switchTab('upload'));
document.getElementById('tabPhoto').addEventListener('click',  () => switchTab('photo'));

// ============================================================
// UTILITIES
// ============================================================
function hideResult() {
  resultWrap.style.display  = 'none';
  resultMain.style.display  = 'none';
  errorCard.style.display   = 'none';
  loadingCard.style.display = 'none';
  confBar.style.width = '0%';
}

const FILL      = { 'Matang':'fill-matang','Mentah':'fill-mentah','Setengah Matang':'fill-setengah' };
const EMOJI     = { matang:'🍌', mentah:'🟢', 'setengah-matang':'🌿' };
const BADGE_CLS = { matang:'badge-matang', mentah:'badge-mentah', 'setengah-matang':'badge-setengah' };

function buildProbBars(container, all_probs) {
  container.innerHTML = '';
  Object.entries(all_probs).forEach(([name, pct]) => {
    const row = document.createElement('div');
    row.className = 'prob-item-row';
    row.innerHTML = `<span class="prob-name">${name}</span>
      <div class="prob-bar-outer"><div class="prob-bar-fill ${FILL[name]||'fill-matang'}" data-w="${pct.toFixed(1)}"></div></div>
      <span class="prob-val">${pct.toFixed(1)}%</span>`;
    container.appendChild(row);
  });
  setTimeout(() => container.querySelectorAll('.prob-bar-fill').forEach(b => b.style.width = b.dataset.w+'%'), 80);
}

function buildNutrisi(container, nutrisi) {
  container.innerHTML = '';
  Object.entries(nutrisi).forEach(([k,v], i) => {
    const el = document.createElement('div');
    el.className = 'nutrisi-item fade-in';
    el.style.animationDelay = `${i*30}ms`;
    el.innerHTML = `<div class="nkey">${k}</div><div class="nval">${v}</div>`;
    container.appendChild(el);
  });
}

async function sendToPredict(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res  = await fetch('/predict', { method:'POST', body:fd });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Terjadi kesalahan');
  return data;
}

// ============================================================
// TAB 1 — UPLOAD
// ============================================================
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewCard = document.getElementById('previewCard');
const previewImg  = document.getElementById('previewImg');
const previewName = document.getElementById('previewName');
const previewSize = document.getElementById('previewSize');
let uploadFile = null;

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if(fileInput.files[0]) handleUpload(fileInput.files[0]); });

function handleUpload(f) {
  const ok = ['image/jpeg','image/png','image/webp'];
  if (!ok.includes(f.type)) { alert('Format tidak didukung.'); return; }
  if (f.size > 16*1024*1024) { alert('File terlalu besar (maks 16MB).'); return; }
  uploadFile = f;
  const r = new FileReader();
  r.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = f.name.length > 34 ? f.name.slice(0,32)+'…' : f.name;
    previewSize.textContent = (f.size/1024).toFixed(1)+' KB';
    previewCard.style.display = 'block';
    hideResult();
  };
  r.readAsDataURL(f);
}

document.getElementById('btnCancel').addEventListener('click', () => {
  uploadFile = null; previewCard.style.display='none'; fileInput.value=''; hideResult();
});

document.getElementById('btnAnalyze').addEventListener('click', async () => {
  if (!uploadFile) return;
  resultImg.src = previewImg.src;
  showLoading();
  try { renderResult(await sendToPredict(uploadFile)); }
  catch(e) { showError(e.message); }
});

// ============================================================
// TAB 2 — KAMERA FOTO (auto open)
// ============================================================
const photoVideo       = document.getElementById('photoVideo');
const photoCanvas      = document.getElementById('photoCanvas');
const photoPlaceholder = document.getElementById('photoPlaceholder');
const photoControls    = document.getElementById('photoControls');
const photoPreviewCard = document.getElementById('photoPreviewCard');
const photoPreviewImg  = document.getElementById('photoPreviewImg');
const photoPreviewTime = document.getElementById('photoPreviewTime');
let photoStream = null;
let photoBlob   = null;

async function startPhotoCamera() {
  photoPlaceholder.style.display = 'flex';
  photoPlaceholder.innerHTML = `<div class="spinner-ring"></div><p>Membuka kamera…</p>`;
  photoVideo.style.display    = 'none';
  photoControls.style.display = 'none';
  try {
    const constraints = [
      { video:{ facingMode:'user', width:{ideal:1280}, height:{ideal:720} }, audio:false },
      { video:{ width:{ideal:1280}, height:{ideal:720} }, audio:false },
      { video:true, audio:false }
    ];
    let stream = null;
    for (const c of constraints) {
      try { stream = await navigator.mediaDevices.getUserMedia(c); break; } catch(e) { continue; }
    }
    if (!stream) throw new Error('Tidak dapat mengakses kamera');
    photoStream = stream;
    photoVideo.srcObject = stream;
    photoPlaceholder.style.display = 'none';
    photoVideo.style.display       = 'block';
    photoControls.style.display    = 'flex';
  } catch(e) {
    photoPlaceholder.innerHTML = `<i class="ti ti-camera-off" style="font-size:36px;color:#e57373;margin-bottom:8px"></i>
      <p style="color:#c62828;font-size:12px">Gagal: ${e.message}</p>
      <p style="color:var(--muted);font-size:11px;margin-top:4px">Pastikan izin kamera diberikan di browser</p>`;
  }
}

function stopPhotoCamera() {
  if (photoStream) { photoStream.getTracks().forEach(t=>t.stop()); photoStream=null; }
  photoVideo.srcObject = null;
  photoVideo.style.display       = 'none';
  photoControls.style.display    = 'none';
  photoPreviewCard.style.display = 'none';
  photoBlob = null;
}

document.getElementById('btnCapture').addEventListener('click', () => {
  if (!photoStream) return;
  const w = photoVideo.videoWidth||640, h = photoVideo.videoHeight||480;
  photoCanvas.width=w; photoCanvas.height=h;
  photoCanvas.getContext('2d').drawImage(photoVideo,0,0,w,h);
  const dataURL = photoCanvas.toDataURL('image/jpeg',0.92);
  photoPreviewImg.src = dataURL;
  photoPreviewTime.textContent = new Date().toLocaleTimeString('id-ID');
  photoCanvas.toBlob(b => { photoBlob=b; }, 'image/jpeg', 0.92);
  photoPreviewCard.style.display = 'block';
  hideResult();
});

document.getElementById('btnPhotoRetake').addEventListener('click', () => {
  photoPreviewCard.style.display='none'; photoBlob=null; hideResult();
});

document.getElementById('btnPhotoAnalyze').addEventListener('click', async () => {
  if (!photoBlob) return;
  resultImg.src = photoPreviewImg.src;
  showLoading();
  try { renderResult(await sendToPredict(new File([photoBlob],'foto.jpg',{type:'image/jpeg'}))); }
  catch(e) { showError(e.message); }
});

document.getElementById('btnStopPhoto').addEventListener('click', () => {
  stopPhotoCamera();
  setTimeout(() => startPhotoCamera(), 300);
});

// ============================================================
// SHARED RENDER & RESET
// ============================================================
function showLoading() {
  resultWrap.style.display  = 'block';
  loadingCard.style.display = 'block';
  resultMain.style.display  = 'none';
  errorCard.style.display   = 'none';
}

function renderResult(data) {
  const { predicted_class, label, confidence, all_probs, nutrisi } = data;
  resultKelas.textContent = label;
  confPct.textContent = confidence.toFixed(1)+'%';
  setTimeout(() => { confBar.style.width = confidence+'%'; }, 80);
  resultBadge.className = 'badge '+(BADGE_CLS[predicted_class]||'badge-matang');
  resultBadge.textContent = (EMOJI[predicted_class]||'') + ' ' + label;
  buildProbBars(probGrid, all_probs);
  nutrisiSub.textContent = `— Pisang ${label}`;
  buildNutrisi(nutrisiGrid, nutrisi);
  loadingCard.style.display = 'none';
  resultMain.style.display  = 'block';
  resultMain.classList.add('fade-in');
}

function showError(msg) {
  errorMsg.textContent = msg;
  loadingCard.style.display = 'none';
  errorCard.style.display   = 'block';
}

[document.getElementById('btnReset'), document.getElementById('btnResetErr')].forEach(btn => {
  if(btn) btn.addEventListener('click', () => {
    hideResult();
    uploadFile = null;
    previewCard.style.display      = 'none';
    fileInput.value                = '';
    photoPreviewCard.style.display = 'none';
    photoBlob = null;
  });
});
"""


# ============================================================
# BACKEND: handler Tornado pengganti route Flask
# (bukan Flask — memanfaatkan Tornado yang memang dipakai
#  Streamlit di baliknya, disuntikkan ke server yang aktif)
# ============================================================
class PredictHandler(tornado.web.RequestHandler):
    def check_xsrf_cookie(self):
        pass  # endpoint API, tidak pakai proteksi XSRF form Streamlit

    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    def post(self):
        if model is None:
            self.set_status(500)
            self.write(json.dumps({'error': 'Model belum dimuat. Pastikan file pisang.keras ada di folder model/'}))
            return

        files = self.request.files.get('file')
        if not files:
            self.set_status(400)
            self.write(json.dumps({'error': 'Tidak ada file yang diunggah'}))
            return

        fileinfo = files[0]
        filename = fileinfo.get('filename', '')
        if not filename:
            self.set_status(400)
            self.write(json.dumps({'error': 'Pilih file terlebih dahulu'}))
            return

        if not allowed_file(filename):
            self.set_status(400)
            self.write(json.dumps({'error': 'Format file tidak didukung. Gunakan JPG, PNG, atau WEBP'}))
            return

        safe_name = secure_filename(filename)
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        with open(save_path, 'wb') as f:
            f.write(fileinfo['body'])

        try:
            predicted_class, confidence, all_probs = predict_image(save_path)
            label = LABEL_MAP.get(predicted_class, predicted_class)
            nutrisi = NUTRISI.get(predicted_class, {})
            self.write(json.dumps({
                'success': True,
                'predicted_class': predicted_class,
                'label': label,
                'confidence': round(confidence, 2),
                'all_probs': {LABEL_MAP.get(k, k): round(v, 2) for k, v in all_probs.items()},
                'nutrisi': nutrisi,
                'image_url': f'/static/uploads/{safe_name}',
            }))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({'error': f'Gagal memproses gambar: {str(e)}'}))


class StyleHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "text/css")
        self.write(STYLE_CSS)


class MainJsHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/javascript")
        self.write(MAIN_JS)


def _get_tornado_app():
    """Cari objek Tornado Application milik server Streamlit yang aktif,
    kompatibel dengan beberapa versi Streamlit."""
    Server = None
    for modpath in (
        "streamlit.web.server.server",
        "streamlit.server.server",
    ):
        try:
            mod = __import__(modpath, fromlist=["Server"])
            Server = getattr(mod, "Server", None)
            if Server:
                break
        except Exception:
            continue
    if Server is None:
        return None
    try:
        server = Server.get_current()
    except Exception:
        server = None
    if server is None:
        return None
    for attr in ("_app", "_tornado_app", "app"):
        tornado_app = getattr(server, attr, None)
        if tornado_app is not None:
            return tornado_app
    return None


def register_backend_routes():
    tornado_app = _get_tornado_app()
    if tornado_app is None:
        return False
    if getattr(tornado_app, "_banana_routes_added", False):
        return True
    tornado_app.add_handlers(r".*$", [
        (r"/predict", PredictHandler),
        (r"/static/css/style\.css", StyleHandler),
        (r"/static/js/main\.js", MainJsHandler),
        (r"/static/uploads/(.*)", tornado.web.StaticFileHandler, {"path": UPLOAD_FOLDER}),
    ])
    tornado_app._banana_routes_added = True
    return True


# ============================================================
# STREAMLIT ENTRYPOINT
# ============================================================
@st.cache_resource(show_spinner="🔄 Memuat model AI...")
def init_model():
    load_banana_model()
    return True


@st.cache_resource(show_spinner=False)
def init_routes():
    return register_backend_routes()


st.set_page_config(page_title="BananaLens", page_icon="🍌", layout="centered")

init_model()
routes_ok = init_routes()

if not routes_ok:
    st.error(
        "⚠️ Gagal menyuntikkan endpoint backend (/predict) ke server Streamlit. "
        "Ini bisa terjadi kalau struktur internal Streamlit berubah setelah update versi. "
        "Coba jalankan ulang aplikasi (streamlit run app.py), atau cek versi Streamlit yang terpasang."
    )

_rendered_html = Template(INDEX_HTML).render(model_loaded=(model is not None))
components.html(_rendered_html, height=1800, scrolling=True)