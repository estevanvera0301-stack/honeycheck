
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib, os, tempfile, base64
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

st.set_page_config(
    page_title="HoneyCheck — Pureza y Origen",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
#  PAPER & INK — Sistema de diseño
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #F8F7F3;
    --surface:   #FFFFFF;
    --ink:       #1A1A2E;
    --ink-mid:   #4A4A6A;
    --ink-light: #8A8AA0;
    --accent:    #185FA5;
    --accent-lt: #E6F1FB;
    --border:    #E0DFDA;
    --border-md: #C8C7C0;
    --honey:     #1D9E75;
    --jarabe:    #E24B4A;
    --mezcla:    #7F77DD;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
}

/* Ocultar elementos nativos de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAlert"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background-color: var(--surface) !important;
    border: 1.5px dashed var(--border-md) !important;
    border-radius: 4px !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
    background-color: var(--accent-lt) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 2px; }

/* ── HERO ── */
.hero {
    width: 100%;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0;
}
.hero-inner {
    max-width: 1200px;
    margin: 0 auto;
    padding: 56px 48px 52px;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 64px;
    align-items: center;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 72px;
    font-weight: 700;
    line-height: 0.9;
    color: var(--ink);
    margin: 0 0 8px 0;
    letter-spacing: -2px;
}
.hero-title em {
    font-style: italic;
    color: var(--accent);
}
.hero-rule {
    width: 48px;
    height: 2px;
    background: var(--ink);
    margin: 24px 0;
}
.hero-desc {
    font-size: 16px;
    font-weight: 300;
    line-height: 1.7;
    color: var(--ink-mid);
    max-width: 480px;
    margin: 0 0 32px 0;
}
.hero-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-tag {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-mid);
    border: 1px solid var(--border-md);
    padding: 5px 14px;
    border-radius: 2px;
    background: var(--bg);
}

/* ── LOGO SIDE ── */
.hero-logo-side {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    border-left: 1px solid var(--border);
    padding-left: 64px;
}
.hero-logo-img {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    object-fit: contain;
    background: white;
    padding: 6px;
    border: 1px solid var(--border);
}
.hero-logo-text {
    text-align: center;
}
.hero-logo-uni {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--ink-mid);
    line-height: 1.5;
}
.hero-logo-grupo {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 300;
    color: var(--ink-light);
    margin-top: 4px;
    line-height: 1.4;
}
.hero-arch {
    width: 100%;
    border-top: 1px solid var(--border);
    padding-top: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.arch-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
}
.arch-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
}
.arch-value {
    font-family: 'Playfair Display', serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink);
    text-align: right;
}
.arch-acc {
    font-size: 11px;
    font-weight: 500;
    color: var(--accent);
}

/* ── UPLOAD SECTION ── */
.upload-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 48px 56px;
}
.section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 12px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 24px 0;
}

/* ── EMPTY STATE ── */
.empty-state {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 48px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
}
.empty-card {
    background: var(--surface);
    padding: 32px 28px;
}
.empty-num {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 700;
    color: var(--border-md);
    line-height: 1;
    margin-bottom: 12px;
}
.empty-head {
    font-size: 14px;
    font-weight: 500;
    color: var(--ink);
    margin-bottom: 6px;
}
.empty-body {
    font-size: 13px;
    font-weight: 300;
    color: var(--ink-light);
    line-height: 1.6;
}

/* ── RESULTS ── */
.results-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 48px 80px;
}
.sample-meta {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 16px 0;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}
.sample-index {
    font-family: 'Playfair Display', serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink-light);
    min-width: 32px;
}
.sample-filename {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 1px;
    color: var(--ink);
    text-transform: uppercase;
}
.phase-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin: 28px 0 16px 0;
}

/* ── RESULTADO CARD ── */
.result-card {
    padding: 24px 28px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}
.result-card.honey  { border-left: 3px solid var(--honey);  }
.result-card.jarabe { border-left: 3px solid var(--jarabe); }
.result-card.mezcla { border-left: 3px solid var(--mezcla); }
.result-type {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 6px;
}
.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 600;
    color: var(--ink);
}
.result-value.honey  { color: var(--honey); }
.result-value.jarabe { color: var(--jarabe); }
.result-value.mezcla { color: var(--mezcla); }

/* ── PROB BARS ── */
.prob-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 22px;
    height: 100%;
}
.prob-header {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 16px;
}
.prob-row { margin: 12px 0; }
.prob-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.prob-cls { font-size: 14px; font-weight: 400; color: var(--ink); }
.prob-pct { font-size: 14px; font-weight: 500; }
.prob-track {
    background: var(--bg);
    border-radius: 2px;
    height: 3px;
    width: 100%;
    overflow: hidden;
}

/* ── GEO CARDS ── */
.geo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 4px;
}
.geo-card {
    padding: 28px 24px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
    text-align: center;
    transition: all 0.2s;
}
.geo-card.active {
    border-color: var(--accent);
    border-left-width: 3px;
}
.geo-card.inactive { opacity: 0.35; }
.geo-region {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 10px;
}
.geo-pct {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 8px;
}
.geo-tag {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent);
}

/* ── FOOTER ── */
.footer {
    border-top: 1px solid var(--border);
    background: var(--surface);
    padding: 24px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left {
    font-size: 11px;
    font-weight: 300;
    color: var(--ink-light);
    letter-spacing: 1px;
}
.footer-right {
    font-size: 11px;
    font-weight: 500;
    color: var(--ink-mid);
    letter-spacing: 1px;
}

/* Status bar */
.status-bar {
    background: var(--accent-lt);
    border-bottom: 1px solid #B5D4F4;
    padding: 8px 48px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    display: inline-block;
}
.status-bar.warn {
    background: #FEF3CD;
    border-color: #F5D47A;
    color: #856404;
}
.status-bar.warn .status-dot { background: #856404; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════
T_MIN, T_MAX, N_PTS = -35.0, 195.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH  = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO   = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#1D9E75", "#E24B4A", "#7F77DD"]
CSS_AUTH     = ["honey", "jarabe", "mezcla"]

# ═══════════════════════════════════════════════════════════════════
#  LOGO en base64
# ═══════════════════════════════════════════════════════════════════
def logo_b64():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "logo_uniquindio.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ═══════════════════════════════════════════════════════════════════
#  MODELOS
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelos():
    base = os.path.dirname(__file__)
    try:
        m_auth  = joblib.load(os.path.join(base, "modelo_svm_optimizado.pkl"))
        sc_auth = joblib.load(os.path.join(base, "scaler_B_final.pkl"))
        m_geo   = joblib.load(os.path.join(base, "modelo_origen_geografico.pkl"))
        sc_geo  = joblib.load(os.path.join(base, "scaler_geo_A.pkl"))
        pca_geo = joblib.load(os.path.join(base, "pca_geo.pkl"))
        return m_auth, sc_auth, m_geo, sc_geo, pca_geo
    except:
        return None, None, None, None, None

# ═══════════════════════════════════════════════════════════════════
#  DSC
# ═══════════════════════════════════════════════════════════════════
def leer_dsc(ruta):
    filas, iniciado = [], False
    with open(ruta, "r", encoding="latin-1") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("##"):
                iniciado = True; continue
            if not iniciado or linea == "": continue
            partes = linea.split("\t")
            if len(partes) < 5: continue
            try:
                filas.append((float(partes[0]), float(partes[2]), int(partes[4])))
            except ValueError:
                continue
    df = pd.DataFrame(filas, columns=["Temp","DSC","Segmento"])
    return df[df["Segmento"] == 4].reset_index(drop=True)

def interpolar(df):
    T, dsc = df["Temp"].values, df["DSC"].values
    if T.min() > T_MIN or T.max() < T_MAX:
        return None
    f = interp1d(T, dsc, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(T_GRILLA)

def extraer_features(dsc_curve, T=T_GRILLA):
    feats = {}
    feats["enthalpy_total"] = trapezoid(dsc_curve, T)
    for zona, (a, b) in [("low",(-30,30)),("mid",(30,100)),("high",(100,190))]:
        mask = (T>=a)&(T<=b)
        feats[f"enthalpy_{zona}"] = trapezoid(dsc_curve[mask], T[mask])
    feats["dsc_min"]        = np.min(dsc_curve)
    feats["dsc_min_temp"]   = T[np.argmin(dsc_curve)]
    feats["dsc_max"]        = np.max(dsc_curve)
    feats["dsc_max_temp"]   = T[np.argmax(dsc_curve)]
    feats["dsc_mean"]       = np.mean(dsc_curve)
    feats["dsc_std"]        = np.std(dsc_curve)
    feats["dsc_slope_mean"] = np.mean(np.diff(dsc_curve))
    picos, props = find_peaks(-dsc_curve, prominence=0.05, distance=50)
    feats["n_picos"] = len(picos)
    if len(picos) > 0:
        idx_p = picos[np.argmax(props["prominences"])]
        feats["pico1_temp"]  = T[idx_p]
        feats["pico1_valor"] = dsc_curve[idx_p]
        feats["pico1_prom"]  = props["prominences"][np.argmax(props["prominences"])]
    else:
        feats["pico1_temp"] = feats["pico1_valor"] = feats["pico1_prom"] = 0.0
    umbral    = 0.1 * feats["dsc_min"]
    onset_idx = np.argmax(dsc_curve < umbral)
    feats["onset_temp"] = T[onset_idx] if onset_idx > 0 else 0.0
    return feats

def graficar_termograma(dsc_curve, color_linea):
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Zonas
    ax.axvspan(-35,  30, alpha=0.04, color="#185FA5", zorder=0)
    ax.axvspan( 30, 100, alpha=0.04, color="#1D9E75", zorder=0)
    ax.axvspan(100, 195, alpha=0.04, color="#E24B4A", zorder=0)

    for x_line in [30, 100]:
        ax.axvline(x_line, color="#E0DFDA", linewidth=0.8, linestyle="--", zorder=1)

    ax.axhline(0, color="#C8C7C0", linewidth=0.8, zorder=2)
    ax.plot(T_GRILLA, dsc_curve, color=color_linea, linewidth=1.8, zorder=5)
    ax.fill_between(T_GRILLA, dsc_curve, where=dsc_curve < 0,
                    alpha=0.12, color=color_linea, zorder=3)
    ax.fill_between(T_GRILLA, dsc_curve, where=dsc_curve >= 0,
                    alpha=0.05, color=color_linea, zorder=3)

    for x_c, lbl, c_lbl in [
        (  -3, "Fusión",        "#185FA5"),
        (  65, "Transición",    "#1D9E75"),
        ( 147, "Caramelización","#E24B4A"),
    ]:
        ax.text(x_c, 1.0, lbl, ha="center", va="top", fontsize=8,
                color=c_lbl, alpha=0.6, fontfamily="sans-serif",
                transform=ax.get_xaxis_transform(), clip_on=True)

    ax.set_xlabel("Temperatura (°C)", fontsize=10, color="#8A8AA0", labelpad=8)
    ax.set_ylabel("Flujo de calor (mW/mg)", fontsize=10, color="#8A8AA0", labelpad=8)
    ax.tick_params(colors="#8A8AA0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#E0DFDA")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#E0DFDA", linewidth=0.4, linestyle="-", alpha=0.8)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(which="minor", length=2, color="#E0DFDA")
    ax.set_xlim(-35, 195)
    plt.tight_layout(pad=0.8)
    return fig

# ═══════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════
logo = logo_b64()
logo_html = f'<img src="data:image/png;base64,{logo}" class="hero-logo-img" alt="Logo Universidad del Quindío">' \
    if logo else '<div style="width:110px;height:110px;border-radius:50%;background:#F8F7F3;border:1px solid #E0DFDA;display:flex;align-items:center;justify-content:center;font-size:11px;color:#8A8AA0;text-align:center;padding:12px;">UniQ</div>'

st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-left">
      <div class="hero-eyebrow">Calorimetría diferencial de barrido · Machine Learning</div>
      <h1 class="hero-title">Honey<em>Check</em></h1>
      <div class="hero-rule"></div>
      <p class="hero-desc">
        Detección de adulteración y trazabilidad geográfica de mieles colombianas
        mediante análisis DSC y modelos de clasificación supervisada.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">Sistema jerárquico v2.0</span>
        <span class="hero-tag">NETZSCH DSC 214 Polyma</span>
        <span class="hero-tag">SVM · PCA</span>
      </div>
    </div>
    <div class="hero-logo-side">
      {logo_html}
      <div class="hero-logo-text">
        <div class="hero-logo-uni">Universidad<br>del Quindío</div>
        <div class="hero-logo-grupo">Grupo de Investigación<br>Plaguicidas y Salud</div>
      </div>
      <div class="hero-arch">
        <div class="arch-row">
          <span class="arch-label">Nivel 01 — Autenticidad</span>
          <span class="arch-acc">98.39%</span>
        </div>
        <div class="arch-row">
          <span class="arch-label">Nivel 02 — Origen</span>
          <span class="arch-acc">82.00%</span>
        </div>
        <div class="arch-row">
          <span class="arch-label">Muestras entrenamiento</span>
          <span class="arch-value" style="font-size:13px; color:#1A1A2E;">62</span>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  STATUS
# ═══════════════════════════════════════════════════════════════════
m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()

if m_auth is not None:
    st.markdown('<div class="status-bar"><span class="status-dot"></span>Sistemas calibrados — Listo para análisis</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-bar warn"><span class="status-dot"></span>Modelos no encontrados — Modo demostración visual</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  UPLOAD
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="upload-wrap">
  <div class="section-label">Paso 01</div>
  <h2 class="section-title">Carga de termogramas</h2>
</div>
""", unsafe_allow_html=True)

# Contenedor con padding lateral consistente
with st.container():
    col_pad_l, col_main, col_pad_r = st.columns([48, 1104, 48])
    with col_main:
        archivos = st.file_uploader(
            "Archivos .txt del NETZSCH DSC 214 Polyma",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

# ═══════════════════════════════════════════════════════════════════
#  EMPTY STATE
# ═══════════════════════════════════════════════════════════════════
if not archivos:
    st.markdown("""
    <div class="upload-wrap" style="padding-top:0;">
    <div class="empty-state">
      <div class="empty-card">
        <div class="empty-num">01</div>
        <div class="empty-head">Cargue el termograma</div>
        <div class="empty-body">Seleccione o arrastre los archivos .txt generados por el equipo NETZSCH DSC 214 Polyma.</div>
      </div>
      <div class="empty-card">
        <div class="empty-num">02</div>
        <div class="empty-head">Clasificación automática</div>
        <div class="empty-body">El sistema evalúa autenticidad mediante SVM Lineal entrenado con 62 muestras reales.</div>
      </div>
      <div class="empty-card">
        <div class="empty-num">03</div>
        <div class="empty-head">Trazabilidad geográfica</div>
        <div class="empty-body">Si la muestra es miel auténtica, el sistema determina el origen entre Eje Cafetero y Orinoquía.</div>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      <div class="footer-left">HoneyCheck · Sistema Jerárquico v2.0 · © 2024</div>
      <div class="footer-right">Universidad del Quindío — Grupo Plaguicidas y Salud</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  RESULTADOS
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="results-wrap">
  <div class="section-label">Paso 02</div>
  <h2 class="section-title">Informe analítico — {len(archivos)} muestra{'s' if len(archivos)!=1 else ''}</h2>
</div>
""", unsafe_allow_html=True)

for i, archivo in enumerate(archivos):
    st.markdown(f"""
    <div class="results-wrap" style="padding-top:0; padding-bottom:0;">
      <div class="sample-meta">
        <span class="sample-index">{str(i+1).zfill(2)}</span>
        <span class="sample-filename">{archivo.name}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(archivo.read())
        tmp_path = tmp.name

    try:
        df_raw     = leer_dsc(tmp_path)
        dsc_interp = interpolar(df_raw)

        if dsc_interp is None:
            st.error("El archivo no cubre el rango térmico requerido (−35 a 195 °C).")
            continue

        if m_auth is None:
            st.info("Modelos no disponibles — mostrando solo el termograma.")
            fig = graficar_termograma(dsc_interp, "#185FA5")
            with st.container():
                _, col_plot, _ = st.columns([48, 1104, 48])
                with col_plot:
                    st.pyplot(fig, use_container_width=True)
            plt.close()
            continue

        # ── NIVEL 1 ──────────────────────────────────────────────
        feats  = extraer_features(dsc_interp)
        X_feat = sc_auth.transform(np.array(list(feats.values())).reshape(1, -1))
        pred   = m_auth.predict(X_feat)[0]
        probs  = m_auth.predict_proba(X_feat)[0]

        css_cls = CSS_AUTH[pred]

        st.markdown(f"""
        <div class="results-wrap" style="padding-top:0; padding-bottom:0;">
          <div class="phase-label">Fase 01 — Evaluación de autenticidad</div>
          <div class="result-card {css_cls}">
            <div>
              <div class="result-type">Clasificación</div>
              <div class="result-value {css_cls}">{CLASES_AUTH[pred]}</div>
            </div>
            <div style="font-size:11px; font-weight:500; letter-spacing:2px; text-transform:uppercase; color:var(--ink-light);">
              Confianza {probs[pred]*100:.1f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _, col_content, _ = st.columns([48, 1104, 48])
        with col_content:
            col_g, col_m = st.columns([2.6, 1.2])
            with col_g:
                fig = graficar_termograma(dsc_interp, COLORES_AUTH[pred])
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_m:
                bars_html = '<div class="prob-wrap"><div class="prob-header">Distribución probabilística</div>'
                for cls, prob, col in zip(CLASES_AUTH, probs, COLORES_AUTH):
                    bars_html += f"""
                    <div class="prob-row">
                      <div class="prob-top">
                        <span class="prob-cls">{cls}</span>
                        <span class="prob-pct" style="color:{col};">{prob*100:.1f}%</span>
                      </div>
                      <div class="prob-track">
                        <div style="background:{col}; width:{prob*100:.1f}%; height:100%; border-radius:2px;"></div>
                      </div>
                    </div>"""
                bars_html += '</div>'
                st.markdown(bars_html, unsafe_allow_html=True)

        # ── NIVEL 2 ──────────────────────────────────────────────
        if pred == 0:
            X_norm   = sc_geo.transform(dsc_interp.reshape(1, -1))
            X_pca    = pca_geo.transform(X_norm)
            pred_geo = m_geo.predict(X_pca)[0]
            prob_geo = m_geo.predict_proba(X_pca)[0]

            st.markdown("""
            <div class="results-wrap" style="padding-top:0; padding-bottom:0;">
              <div class="phase-label">Fase 02 — Trazabilidad geográfica</div>
            </div>
            """, unsafe_allow_html=True)

            _, col_geo, _ = st.columns([48, 1104, 48])
            with col_geo:
                geo_html = '<div class="geo-grid">'
                for cls, prob in zip(CLASES_GEO, prob_geo):
                    activo = prob > 0.5
                    cls_card = "active" if activo else "inactive"
                    tag = '<div class="geo-tag">Origen determinado</div>' if activo else ""
                    geo_html += f"""
                    <div class="geo-card {cls_card}">
                      <div class="geo-region">{cls}</div>
                      <div class="geo-pct">{prob*100:.1f}%</div>
                      {tag}
                    </div>"""
                geo_html += '</div>'
                st.markdown(geo_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar: {e}")
    finally:
        os.unlink(tmp_path)

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="height:48px;"></div>
<div class="footer">
  <div class="footer-left">HoneyCheck · Sistema Jerárquico v2.0 · © 2024</div>
  <div class="footer-right">Universidad del Quindío — Grupo Plaguicidas y Salud</div>
</div>
""", unsafe_allow_html=True)
