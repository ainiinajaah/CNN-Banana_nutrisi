import os, io, base64, datetime
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="BananaLens", page_icon="🍌", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"/>
<style>
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stHeader"]{display:none!important}
[data-testid="stAppViewContainer"]{background:#FAFAF8}
[data-testid="stMain"] > div:first-child{padding-top:0!important}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --yellow:#F5A623;--yellow-light:#FFF8E7;--yellow-border:#FFD77A;--yellow-deep:#C47C00;
  --bg:#FAFAF8;--surface:#FFFFFF;--border:rgba(0,0,0,0.08);
  --text:#1A1208;--muted:#6B7280;
  --ff:'Sora',system-ui,sans-serif;--ffi:'Inter',system-ui,sans-serif;
  --r-lg:16px;--r-md:10px;--r-sm:8px;
}
body{font-family:var(--ffi);background:var(--bg);color:var(--text)}
.app{max-width:960px;margin:0 auto;padding:28px 20px 56px}
.header{text-align:center;margin-bottom:24px}
.logo-row{display:inline-flex;align-items:center;gap:10px;margin-bottom:6px}
.logo-icon-wrap{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#FFD600,#F5A623);display:flex;align-items:center;justify-content:center;font-size:20px}
.logo-name{font-family:var(--ff);font-size:22px;font-weight:700;letter-spacing:-0.5px}
.logo-name span{color:var(--yellow-deep)}
.header-desc{font-size:13px;color:var(--muted)}
.model-warning{display:inline-block;margin-top:10px;background:#FFF8E7;border:0.5px solid var(--yellow-border);border-radius:var(--r-sm);padding:7px 14px;font-size:12px;color:#7A5800}
.model-warning code{background:rgba(0,0,0,0.07);border-radius:4px;padding:1px 5px}
.step-label{font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);border-left:2px solid var(--yellow);padding-left:8px;margin-bottom:12px;margin-top:8px}
.upload-area{padding:36px 24px;border:1.5px dashed #D1C9B0;border-radius:var(--r-lg);text-align:center;cursor:pointer;background:var(--yellow-light);margin-bottom:14px}
.upload-big-icon{font-size:32px;margin-bottom:8px}
.upload-area h3{font-family:var(--ff);font-size:15px;font-weight:600;margin-bottom:4px}
.upload-area p{font-size:13px;color:var(--muted)}
.upload-fmt{font-size:11px;color:#A89070;letter-spacing:0.05em;margin-top:10px}
.preview-card{background:var(--surface);border:0.5px solid var(--border);border-radius:var(--r-lg);padding:14px;margin-bottom:14px}
.preview-inner{display:flex;gap:14px;align-items:center}
.preview-inner img{width:80px;height:80px;border-radius:var(--r-md);object-fit:cover;border:0.5px solid var(--border);display:block;flex-shrink:0}
.preview-info{flex:1;min-width:0}
.preview-info h4{font-family:var(--ff);font-size:13px;font-weight:600;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.preview-info p{font-size:12px;color:var(--muted)}
.result-top{background:var(--surface);border:0.5px solid var(--border);border-radius:var(--r-lg);padding:18px;margin-bottom:10px}
.result-top-inner{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:14px;padding-bottom:14px;border-bottom:0.5px solid var(--border)}
.result-img-wrap img{width:78px;height:78px;border-radius:var(--r-md);object-fit:cover;border:0.5px solid var(--border);display:block}
.result-info{flex:1;min-width:140px}
.result-kelas{font-family:var(--ff);font-size:20px;font-weight:700;margin-bottom:5px}
.result-badge-row{margin-bottom:8px}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;border-radius:20px;padding:3px 10px;letter-spacing:0.03em}
.badge-matang{background:#FFF8E7;color:#8B5800;border:0.5px solid #FFD77A}
.badge-mentah{background:#EAF3DE;color:#27500A;border:0.5px solid #9FE1CB}
.badge-setengah{background:#F9FBE7;color:#3B6D11;border:0.5px solid #C0DD97}
.conf-label{font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.conf-row{display:flex;align-items:center;gap:8px}
.conf-bar-outer{flex:1;height:5px;background:#EDE2C8;border-radius:100px;overflow:hidden}
.conf-bar-inner{height:100%;background:var(--yellow);border-radius:100px;transition:width 1s cubic-bezier(.4,0,.2,1)}
.conf-pct{font-family:var(--ff);font-size:13px;font-weight:700;color:var(--yellow-deep);min-width:40px;text-align:right}
.prob-section-title{font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.prob-grid{display:flex;flex-direction:column;gap:7px}
.prob-item-row{display:flex;align-items:center;gap:8px}
.prob-name{font-size:12px;color:var(--muted);min-width:110px}
.prob-bar-outer{flex:1;height:4px;background:#EDE2C8;border-radius:100px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:100px}
.fill-matang{background:#F5A623}.fill-mentah{background:#5CB85C}.fill-setengah{background:#A8C34F}
.prob-val{font-size:12px;font-weight:500;min-width:38px;text-align:right}
.info-card{background:var(--surface);border:0.5px solid var(--border);border-radius:var(--r-lg);padding:16px}
.info-card-title{font-family:var(--ff);font-size:13px;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.info-card-title i{color:var(--yellow);font-size:15px}
.info-card-title span{font-size:11px;font-weight:400;color:var(--muted);margin-left:4px}
.nutrisi-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
@media(max-width:480px){.nutrisi-grid{grid-template-columns:1fr}}
.nutrisi-item{background:var(--bg);border:0.5px solid var(--border);border-radius:var(--r-sm);padding:7px 10px;transition:border-color 0.15s}
.nutrisi-item:hover{border-color:var(--yellow-border)}
.nkey{font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:var(--muted);margin-bottom:2px}
.nval{font-size:12px;font-weight:500}
.error-card{background:#FFF5F5;border:0.5px solid #F7C1C1;border-radius:var(--r-lg);padding:28px;text-align:center}
.error-icon{font-size:1.8rem;margin-bottom:8px}
.error-msg{font-size:13px;color:#A32D2D;margin-bottom:12px}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeUp 0.3s ease both}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)
LABEL_MAP = {'matang':'Matang','mentah':'Mentah','setengah-matang':'Setengah Matang'}
EMOJI     = {'matang':'🍌','mentah':'🟢','setengah-matang':'🌿'}
BADGE_CLS = {'matang':'badge-matang','mentah':'badge-mentah','setengah-matang':'badge-setengah'}
FILL_CLS  = {'Matang':'fill-matang','Mentah':'fill-mentah','Setengah Matang':'fill-setengah'}

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

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model CNN…")
def load_banana_model():
    path = 'model/pisang.keras'
    if not os.path.exists(path):
        return None
    try:
        return load_model(path, compile=False)
    except Exception:
        return None

def predict_pil(model, pil_img):
    idx2cls = {0:'matang', 1:'mentah', 2:'setengah-matang'}
    img   = pil_img.convert('RGB').resize(IMG_SIZE)
    arr   = preprocess_input(np.array(img, dtype=np.float32))
    preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    pidx  = int(np.argmax(preds))
    pcls  = idx2cls[pidx]
    conf  = float(preds[pidx]) * 100
    probs = {LABEL_MAP[idx2cls[i]]: float(preds[i]) * 100 for i in range(3)}
    return pcls, conf, probs

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode()

# ── HTML builders ─────────────────────────────────────────────────────────────
def prob_bars_html(probs):
    rows = ''
    for name, pct in probs.items():
        fc = FILL_CLS.get(name, 'fill-matang')
        rows += (
            f'<div class="prob-item-row">'
            f'<span class="prob-name">{name}</span>'
            f'<div class="prob-bar-outer"><div class="prob-bar-fill {fc}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="prob-val">{pct:.1f}%</span>'
            f'</div>'
        )
    return f'<div class="prob-grid">{rows}</div>'

def nutrisi_html(pcls):
    items = ''
    for k, v in NUTRISI.get(pcls, {}).items():
        items += f'<div class="nutrisi-item"><div class="nkey">{k}</div><div class="nval">{v}</div></div>'
    return f'<div class="nutrisi-grid">{items}</div>'

def result_html(pil_img, pcls, conf, probs):
    label = LABEL_MAP[pcls]; emoji = EMOJI[pcls]; bdg = BADGE_CLS[pcls]
    b64   = pil_to_b64(pil_img)
    return (
        f'<div class="result-top fade-in">'
        f'<div class="result-top-inner">'
        f'<div class="result-img-wrap"><img src="data:image/jpeg;base64,{b64}"/></div>'
        f'<div class="result-info">'
        f'<div class="result-kelas">{label}</div>'
        f'<div class="result-badge-row"><span class="badge {bdg}">{emoji} {label}</span></div>'
        f'<div class="conf-label">Kepercayaan Model</div>'
        f'<div class="conf-row">'
        f'<div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{conf:.1f}%"></div></div>'
        f'<div class="conf-pct">{conf:.1f}%</div>'
        f'</div></div></div>'
        f'<div class="prob-section-title">Distribusi Probabilitas</div>'
        f'{prob_bars_html(probs)}'
        f'</div>'
        f'<div class="info-card fade-in" style="margin-top:10px">'
        f'<div class="info-card-title"><i class="ti ti-leaf"></i> Kandungan Nutrisi <span>— Pisang {label}</span></div>'
        f'{nutrisi_html(pcls)}'
        f'</div>'
    )

model = load_banana_model()

# ── HEADER ────────────────────────────────────────────────────────────────────
warn = (
    '' if model else
    '<div class="model-warning">⚠️ Model belum dimuat — letakkan '
    '<strong>banana.keras</strong> di folder <code>model/</code></div>'
)
st.markdown(f'''
<div class="app">
<div class="header">
  <div class="logo-row">
    <div class="logo-icon-wrap">🍌</div>
    <div class="logo-name">Banana<span>Lens</span></div>
  </div>
  <div class="header-desc">Klasifikasi Kematangan Pisang — MobileNetV2 CNN</div>
  {warn}
</div>
''', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_up, tab_photo = st.tabs(["📁  Upload", "📷  Kamera Foto"])

# ══════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════
with tab_up:
    st.markdown('<div class="step-label">Upload Gambar Pisang</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="upload-area">
      <div class="upload-big-icon">📁</div>
      <h3>Seret &amp; Lepas Gambar Pisang</h3>
      <p>atau pilih file di bawah ini</p>
      <p class="upload-fmt">JPG · PNG · WEBP · Maks 16MB</p>
    </div>''', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih gambar", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        pil     = Image.open(uploaded)
        size_kb = uploaded.size / 1024
        b64     = pil_to_b64(pil)
        name_d  = (uploaded.name[:34]+"…") if len(uploaded.name)>34 else uploaded.name
        st.markdown(f'''
        <div class="preview-card fade-in">
          <div class="preview-inner">
            <img src="data:image/jpeg;base64,{b64}"/>
            <div class="preview-info">
              <h4>{name_d}</h4>
              <p>{size_kb:.1f} KB</p>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

        if st.button("✨  Analisis", key="btn_up", type="primary"):
            if not model:
                st.markdown('<div class="error-card"><div class="error-icon">⚠️</div><p class="error-msg">Model belum dimuat.</p></div>', unsafe_allow_html=True)
            else:
                with st.spinner("Menganalisis gambar pisang…"):
                    try:
                        pcls, conf, probs = predict_pil(model, pil)
                        st.markdown(result_html(pil, pcls, conf, probs), unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-card"><div class="error-icon">⚠️</div><p class="error-msg">Gagal: {e}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 2 — KAMERA FOTO
# ══════════════════════════════════════════════════
with tab_photo:
    st.markdown('<div class="step-label">Kamera — Ambil Foto</div>', unsafe_allow_html=True)
    cam_img = st.camera_input("Arahkan kamera ke pisang lalu ambil foto",
                               label_visibility="collapsed")
    if cam_img:
        pil = Image.open(cam_img)
        ts  = datetime.datetime.now().strftime('%H:%M:%S')
        b64 = pil_to_b64(pil)
        st.markdown(f'''
        <div class="preview-card fade-in">
          <div class="preview-inner">
            <img src="data:image/jpeg;base64,{b64}"/>
            <div class="preview-info">
              <h4>Foto dari Kamera</h4>
              <p>{ts}</p>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

        if st.button("✨  Analisis Foto", key="btn_photo", type="primary"):
            if not model:
                st.markdown('<div class="error-card"><div class="error-icon">⚠️</div><p class="error-msg">Model belum dimuat.</p></div>', unsafe_allow_html=True)
            else:
                with st.spinner("Menganalisis…"):
                    try:
                        pcls, conf, probs = predict_pil(model, pil)
                        st.markdown(result_html(pil, pcls, conf, probs), unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-card"><div class="error-icon">⚠️</div><p class="error-msg">Gagal: {e}</p></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
