
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib, os, tempfile
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

st.set_page_config(
    page_title="HoneyCheck — Pureza y Origen",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

    :root {
        --sidebar-bg:       #5C3A00;
        --sidebar-card:     #8C5A10;
        --sidebar-accent:   #F5A800;
        --sidebar-gold:     #FFD000;
        --sidebar-text:     #FFF3B0;
        --sidebar-muted:    #C89040;
        --sidebar-border:   rgba(245,168,0,0.22);
        --yellow-blaze:     #FFD000;
        --yellow-main:      #F5A800;
        --yellow-deep:      #C87800;
        --yellow-dark:      #7A4800;
        --yellow-darkest:   #3A2000;
        --cream-purest:     #FFFCE8;
        --cream-base:       #FFF8D8;
        --cream-mid:        #FFF0A8;
        --cream-deep:       #FFE870;
        --text-primary:     #2A1400;
        --text-body:        #4A2800;
        --text-muted:       #906030;
        --border-yellow:    rgba(245,168,0,0.35);
        --border-mid:       rgba(245,168,0,0.55);
        --border-soft:      rgba(245,168,0,0.18);
        --shadow-warm:      rgba(42,20,0,0.09);
        --shadow-deep:      rgba(42,20,0,0.16);
        --green-bg:         #EAF7EC;
        --green-border:     rgba(30,100,48,0.28);
        --red-bg:           #FAEEE8;
        --red-border:       rgba(160,48,32,0.26);
        --bee-pattern: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' stroke='%23F5A800' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round' fill='none' opacity='0.07'%3E%3Cpath d='M92,60 Q80,45 92,35 M108,60 Q120,45 108,35'/%3E%3Cpath d='M88,72 A 12 12 0 0 1 112,72'/%3E%3Ccircle cx='100' cy='95' r='16'/%3E%3Cpath d='M85,110 Q 75,140 100,165 Q 125,140 115,110 Z'/%3E%3Cpath d='M83,125 Q 100,135 117,125 M88,140 Q 100,150 112,140'/%3E%3Cpath d='M 82,90 L 25,50 L 15,65 L 60,105 L 82,100 Z'/%3E%3Cpath d='M 25,50 L 50,90 L 15,65 M 50,90 L 60,105'/%3E%3Cpath d='M 78,108 L 40,125 L 55,140 L 82,118 Z'/%3E%3Cpath d='M 40,125 L 75,114'/%3E%3Cpath d='M 118,90 L 175,50 L 185,65 L 140,105 L 118,100 Z'/%3E%3Cpath d='M 175,50 L 150,90 L 185,65 M 150,90 L 140,105'/%3E%3Cpath d='M 122,108 L 160,125 L 145,140 L 118,118 Z'/%3E%3Cpath d='M 160,125 L 125,114'/%3E%3C/svg%3E");
    }

    html, body, div, p, h1, h2, h3, h4, h5, h6, label, button, ul, li {
        font-family: 'Cormorant Garamond', serif !important;
        font-variant-numeric: lining-nums !important;
    }
    span { font-variant-numeric: lining-nums !important; }
    .logo-brand { font-family: 'Playfair Display', serif !important; }
    .stIcon, .material-symbols-rounded {
        font-family: "Material Symbols Rounded", sans-serif !important;
    }

    /* ── BARRA SUPERIOR STREAMLIT — transparente sobre el fondo crema ── */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stAppHeader,
    header.stAppHeader {
        background-color: var(--cream-base) !important;
        background-image: none !important;
        border-bottom: 1px solid var(--border-yellow) !important;
        box-shadow: 0 1px 8px rgba(42,20,0,0.06) !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg,
    [data-testid="stHeader"] [data-testid="baseButton-header"] {
        color: var(--yellow-deep) !important;
        stroke: var(--yellow-deep) !important;
    }

    /* ── FONDO PRINCIPAL ── */
    .stApp {
        background-color: var(--cream-base) !important;
        background-image:
            var(--bee-pattern),
            radial-gradient(ellipse 62% 52% at 92% 6%,  rgba(255,208,0,0.18) 0%, transparent 62%),
            radial-gradient(ellipse 44% 38% at 4%  88%,  rgba(245,168,0,0.12) 0%, transparent 55%),
            linear-gradient(158deg, var(--cream-purest) 0%, var(--cream-base) 45%, var(--cream-mid) 100%) !important;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat !important;
        background-size: 120px 120px, cover, cover, cover !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, var(--sidebar-gold), #FFE85A, var(--sidebar-gold), transparent);
    }
    [data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }

    /* ── UPLOAD ZONE — forzar fondo claro en todos los modos ── */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploader"],
    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 252, 232, 0.95) !important;
        border: 1.5px dashed var(--border-yellow) !important;
        border-radius: 6px !important;
        transition: all 0.3s ease;
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--yellow-main) !important;
        background-color: rgba(255,252,232,0.98) !important;
        box-shadow: 0 0 24px rgba(255,208,0,0.18) !important;
    }
    /* Texto y botón dentro del uploader */
    [data-testid="stFileUploadDropzone"] *,
    [data-testid="stFileUploader"] * {
        color: var(--text-body) !important;
    }
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploaderDropzoneInstructions"] button {
        background-color: var(--yellow-main) !important;
        color: var(--yellow-darkest) !important;
        border: none !important;
        border-radius: 4px !important;
    }
    [data-testid="stAlert"] { display: none !important; }

    /* ── BARRA DE ESTADO ── */
    .status-bar {
        display: flex; align-items: center; gap: 14px; padding: 12px 22px;
        background: linear-gradient(90deg, rgba(255,208,0,0.14), rgba(245,168,0,0.05), transparent);
        border-left: 3px solid var(--yellow-main); border-radius: 0 4px 4px 0;
        margin: 20px 0 8px 0;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--yellow-main);
        box-shadow: 0 0 10px rgba(255,208,0,0.70); flex-shrink: 0;
        animation: pulse-dot 2.5s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px rgba(255,208,0,0.70); }
        50%       { opacity: 0.55; box-shadow: 0 0 18px rgba(255,208,0,0.28); }
    }
    .status-text {
        font-size: 11px; letter-spacing: 2px; color: var(--yellow-dark);
        text-transform: uppercase; font-weight: 600;
    }
    .status-bar-warn { border-left-color: var(--yellow-deep); }
    .status-bar-warn .status-dot { background: var(--yellow-deep); animation: none; }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--cream-mid); }
    ::-webkit-scrollbar-thumb { background: var(--cream-deep); border-radius: 3px; }

    /* ── STAT CARDS ── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 32px 0 40px 0;
    }
    .stat-cell {
        background: #FFFFFF; padding: 26px 16px; text-align: center;
        border: 1px solid var(--border-yellow); border-radius: 6px;
        box-shadow: 0 2px 16px var(--shadow-warm);
        position: relative; overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        min-width: 0;
    }
    .stat-cell::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, var(--yellow-main), var(--yellow-blaze), var(--yellow-main));
    }
    .stat-cell:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(255,208,0,0.22);
        border-color: var(--border-mid);
    }
    .stat-value {
        display: flex; align-items: baseline; justify-content: center; gap: 2px;
        line-height: 1; min-height: 48px; flex-wrap: wrap;
    }
    .stat-num {
        font-size: clamp(22px, 4vw, 42px) !important;
        font-weight: 700;
        color: var(--yellow-deep);
        letter-spacing: -1px;
        word-break: break-all;
    }
    .stat-unit {
        font-size: clamp(18px, 3vw, 38px) !important;
        font-weight: 700;
        color: var(--yellow-deep);
    }
    .stat-label {
        font-size: clamp(8px, 1.5vw, 10px);
        letter-spacing: 1.5px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-top: 12px;
        font-weight: 600;
        line-height: 1.4;
        word-break: normal;
        overflow-wrap: break-word;
        hyphens: auto;
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        display: flex; align-items: center; gap: 16px;
        margin: 40px 0 24px 0; padding-bottom: 14px;
        border-bottom: 1.5px solid var(--border-yellow);
        flex-wrap: wrap;
    }
    .section-number {
        font-size: 12px; font-weight: bold; color: var(--yellow-deep);
        letter-spacing: 2px; background: rgba(255,208,0,0.18);
        padding: 4px 10px; border-radius: 2px; flex-shrink: 0;
    }
    .section-title {
        font-size: clamp(18px, 3vw, 24px); font-weight: 700;
        color: var(--text-primary) !important; margin: 0;
    }
    .section-line {
        flex: 1; height: 1px; min-width: 20px;
        background: linear-gradient(90deg, var(--border-yellow), transparent);
    }

    /* ── TARJETAS DE MUESTRAS ── */
    .sample-header {
        display: flex; align-items: center; gap: 14px; padding: 16px 20px;
        background: linear-gradient(90deg, rgba(255,255,255,0.97), rgba(255,252,232,0.85));
        border: 1.5px solid var(--border-yellow); border-bottom: none;
        border-radius: 8px 8px 0 0; margin-top: 28px;
        flex-wrap: wrap;
    }
    .sample-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: var(--yellow-blaze);
        box-shadow: 0 0 12px rgba(255,208,0,0.60); flex-shrink: 0;
    }
    .sample-name {
        font-size: 14px; font-weight: 600;
        color: var(--text-primary); letter-spacing: 1px;
        word-break: break-all;
    }
    .sample-body {
        border: 1.5px solid var(--border-yellow); border-top: none;
        border-radius: 0 0 8px 8px; padding: 20px;
        background: rgba(255,255,255,0.92); backdrop-filter: blur(4px);
        box-shadow: 0 4px 24px var(--shadow-warm);
    }
    .level-badge {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 11px; font-weight: 600; letter-spacing: 2px;
        color: var(--yellow-darkest); text-transform: uppercase;
        padding: 6px 16px; border: 1px solid var(--border-yellow);
        border-radius: 2px; background: rgba(255,208,0,0.14);
        margin: 16px 0 14px 0;
    }
    .level-badge::before { content: '⬡'; font-size: 12px; color: var(--yellow-main); }

    /* ── RESULTADOS ── */
    .resultado-base {
        padding: 20px 24px; border-radius: 6px; margin: 10px 0 20px 0;
        display: flex; align-items: center; justify-content: space-between; border: 1.5px solid;
        flex-wrap: wrap; gap: 12px;
    }
    .resultado-real {
        background: linear-gradient(135deg, var(--green-bg), rgba(255,255,255,0.97));
        border-color: var(--green-border);
    }
    .resultado-real .res-icon { color: #2E7D40; }
    .resultado-adulterada {
        background: linear-gradient(135deg, var(--red-bg), rgba(255,255,255,0.97));
        border-color: var(--red-border);
    }
    .resultado-adulterada .res-icon { color: #C03820; }
    .resultado-mezcla {
        background: linear-gradient(135deg, rgba(255,208,0,0.12), rgba(255,255,255,0.97));
        border-color: var(--border-yellow);
    }
    .resultado-mezcla .res-icon { color: var(--yellow-main); }
    .res-label {
        font-size: 11px; font-weight: 600; letter-spacing: 2px;
        color: var(--text-muted); text-transform: uppercase; margin-bottom: 5px;
    }
    .res-value { font-size: clamp(20px, 4vw, 28px); font-weight: 700; color: var(--text-primary); }
    .res-icon { font-size: 36px; opacity: 0.85; }

    /* ── PROBABILIDADES ── */
    .prob-container {
        background: #FFFFFF; border: 1.5px solid var(--border-yellow);
        border-radius: 6px; padding: 22px; height: 100%;
        box-shadow: 0 2px 12px var(--shadow-warm);
    }
    .prob-title {
        font-size: 11px; font-weight: 600; letter-spacing: 2px;
        color: var(--text-muted); text-transform: uppercase; margin-bottom: 18px;
    }
    .prob-row { margin: 14px 0; }
    .prob-row-top {
        display: flex; justify-content: space-between;
        align-items: baseline; margin-bottom: 7px; flex-wrap: wrap; gap: 4px;
    }
    .prob-cls { font-size: 15px; font-weight: 600; color: var(--text-primary); }
    .prob-pct { font-size: 15px; font-weight: 700; }
    .prob-track {
        background: var(--cream-mid); border-radius: 3px;
        height: 6px; width: 100%; overflow: hidden;
    }

    /* ── GEO CARDS ── */
    .geo-card {
        padding: 24px 16px; border-radius: 8px; border: 1.5px solid var(--border-soft);
        background: #FFFFFF; text-align: center; position: relative;
        overflow: hidden; transition: all 0.3s;
        box-shadow: 0 2px 12px var(--shadow-warm);
    }
    .geo-card.active {
        border-color: var(--yellow-blaze);
        background: linear-gradient(160deg, #FFFFFF, rgba(255,240,100,0.12));
        box-shadow: 0 6px 28px rgba(255,208,0,0.22);
    }
    .geo-card.active::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, var(--yellow-main), var(--yellow-blaze), var(--yellow-main));
    }
    .geo-card.inactive { opacity: 0.38; }
    .geo-region {
        font-size: 11px; font-weight: 600; letter-spacing: 2px;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;
    }
    .geo-pct { font-size: clamp(36px, 8vw, 52px) !important; font-weight: 700; line-height: 1; margin-bottom: 6px; }
    .geo-winner-tag {
        font-size: 10px; font-weight: 600; letter-spacing: 2px;
        color: var(--yellow-deep); text-transform: uppercase; margin-top: 10px;
    }

    .sample-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-yellow), transparent);
        margin: 48px 0;
    }

    /* ── FOOTER ── */
    .footer {
        margin-top: 80px; padding: 28px 0 16px 0;
        border-top: 1.5px solid var(--border-yellow);
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px;
    }
    .footer-left { font-size: 11px; font-weight: 600; color: var(--text-muted); letter-spacing: 1px; }
    .footer-right { font-size: 16px; font-weight: 600; color: var(--text-muted); }
    .footer-gold { color: var(--yellow-deep); font-weight: 700; }

    /* ══════════════════════════════════════════════════════════════
       📱 RESPONSIVE MÓVIL
    ══════════════════════════════════════════════════════════════ */
    @media screen and (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        /* Estadísticas: 2 columnas en lugar de 4 */
        .stats-row {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px !important;
            margin: 20px 0 28px 0 !important;
        }
        .stat-cell {
            padding: 18px 10px !important;
        }
        .stat-num {
            font-size: 28px !important;
        }
        .stat-unit {
            font-size: 24px !important;
        }
        .stat-label {
            font-size: 9px !important;
            letter-spacing: 1px !important;
            margin-top: 10px !important;
        }

        /* Hero */
        .logo-brand {
            font-size: 42px !important;
        }

        /* Resultados: columna en móvil */
        .resultado-base {
            flex-direction: column !important;
            text-align: center !important;
            gap: 10px !important;
            padding: 16px !important;
        }

        /* Footer */
        .footer {
            flex-direction: column !important;
            text-align: center !important;
            gap: 6px !important;
        }

        /* Upload: asegurar fondo claro también en móvil */
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 252, 232, 0.97) !important;
        }
    }

    @media screen and (max-width: 480px) {
        .stats-row {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .stat-num {
            font-size: 24px !important;
        }
        .stat-label {
            font-size: 8px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════
T_MIN, T_MAX, N_PTS = -35.0, 195.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#2E7D40", "#C03820", "#C87800"]
COLORES_GEO  = ["#C87800", "#C87800"]

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
#  PROCESAMIENTO DSC
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
    for zona, (a, b) in [("low",(-35,35)),("mid",(35,105)),("high",(105,195))]:
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
        feats["pico1_temp"]  = 0.0
        feats["pico1_valor"] = 0.0
        feats["pico1_prom"]  = 0.0
    umbral    = 0.1 * feats["dsc_min"]
    onset_idx = np.argmax(dsc_curve < umbral)
    feats["onset_temp"] = T[onset_idx] if onset_idx > 0 else 0.0
    return feats

def graficar_termograma(dsc_curve, nombre, color_linea):
    bg      = "#FFFCE8"
    grid_c  = "#FFE060"
    tick_c  = "#906030"
    label_c = "#3A2000"

    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=bg)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    ax.axvspan(-30,  30, alpha=0.07, color="#5580AA", zorder=0)
    ax.axvspan( 30, 100, alpha=0.08, color="#F5A800", zorder=0)
    ax.axvspan(100, 190, alpha=0.07, color="#C05020", zorder=0)

    for x_line in [-30, 30, 100, 190]:
        ax.axvline(x_line, color=grid_c, linewidth=0.6, linestyle=":", zorder=1)

    ax.axhline(0, color="#FFD060", linewidth=0.9, linestyle="-", zorder=2)
    ax.plot(T_GRILLA, dsc_curve, color=color_linea, linewidth=2.2, zorder=5, solid_capstyle="round")

    ax.fill_between(T_GRILLA, dsc_curve, where=dsc_curve < 0,  alpha=0.16, color=color_linea, zorder=3)
    ax.fill_between(T_GRILLA, dsc_curve, where=dsc_curve >= 0, alpha=0.07, color=color_linea, zorder=3)

    for x_c, lbl, c_lbl in [
        (  0, "FUSIÓN",        "#3A70A0"),
        ( 65, "TRANSICIÓN",    "#A07800"),
        (145, "CARAMELIZACIÓN","#A04800"),
    ]:
        ax.text(x_c, 1.0, lbl, ha="center", va="top", fontsize=7.5, color=c_lbl, alpha=0.65,
                fontfamily="monospace", fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=True)

    ax.set_xlabel("Temperatura  (°C)", fontsize=11, color=label_c, labelpad=8)
    ax.set_ylabel("Flujo de calor  (mW/mg)", fontsize=11, color=label_c, labelpad=8)
    ax.tick_params(colors=tick_c, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
        spine.set_linewidth(0.8)
    ax.grid(True, color=grid_c, linewidth=0.4, linestyle="-", alpha=0.55)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(which="minor", length=2, color=grid_c)
    ax.set_xlim(-30, 190)
    plt.tight_layout(pad=0.8)
    return fig

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 0 20px 0; text-align:center; border-bottom:1px solid rgba(255,208,0,0.20); margin-bottom:24px;">
        <div class="logo-brand" style="font-size:28px; font-weight:900; color:#FFD000; letter-spacing:-0.5px;">
            Honey<span style="font-style:italic; color:#FFE85A;">Check</span>
        </div>
        <div style="font-size:12px; font-weight:600; letter-spacing:3px; color:#C89040; text-transform:uppercase; margin-top:4px;">
            Sistema Jerárquico V2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="font-size:18px; font-weight:700; letter-spacing:2px; color:#C89040; text-transform:uppercase; margin-bottom:14px;">
            Arquitectura
        </div>
        <div style="padding:14px 16px; background:#8C5A10; border-left:3px solid #FFD000; margin-bottom:8px; border-radius:4px; box-shadow:0 4px 14px rgba(0,0,0,0.20);">
            <div style="font-size:11px; font-weight:700; color:#C89040; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 01</div>
            <div style="font-size:18px; font-weight:700; color:#FFF3B0;">Autenticidad</div>
            <div style="font-size:13px; font-weight:600; color:#FFD88A; margin-top:3px;">SVM Lineal · Acc 98.39%</div>
        </div>
        <div style="padding:14px 16px; background:#8C5A10; border-left:3px solid #F5A800; margin-bottom:8px; border-radius:4px; box-shadow:0 4px 14px rgba(0,0,0,0.20);">
            <div style="font-size:11px; font-weight:700; color:#C89040; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 02</div>
            <div style="font-size:18px; font-weight:700; color:#FFF3B0;">Origen Geográfico</div>
            <div style="font-size:13px; font-weight:600; color:#FFD88A; margin-top:3px;">SVM+PCA · Acc 82.00%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="margin-bottom:20px;">
        <div style="font-size:18px; font-weight:700; letter-spacing:2px; color:#C89040; text-transform:uppercase; margin-bottom:12px;">Clases detectables</div>
    """, unsafe_allow_html=True)

    clases_colores = [
        ("Miel auténtica",   "#D6F0DA", "#1B6B30"),
        ("Jarabe comercial",  "#FAEEE8", "#C03820"),
        ("Mezcla de azúcares","#FFF8E0", "#C87800"),
    ]
    for cls, bg_c, txt_c in clases_colores:
        st.markdown(f"""
        <div style="background:{bg_c}; border:1px solid {txt_c}44; padding:12px 16px; margin-bottom:8px; border-radius:4px; box-shadow:0 4px 10px rgba(0,0,0,0.22);">
            <div style="font-size:16px; font-weight:700; color:{txt_c}; letter-spacing:0.5px;">{cls}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div style="margin-top:20px; margin-bottom:20px;">
        <div style="font-size:18px; font-weight:700; letter-spacing:2px; color:#C89040; text-transform:uppercase; margin-bottom:12px;">Orígenes geográficos</div>
    """, unsafe_allow_html=True)
    for cls in CLASES_GEO:
        st.markdown(f"""
        <div style="background:#8C5A10; border:1px solid rgba(255,208,0,0.30); padding:12px 16px; margin-bottom:8px; border-radius:4px; box-shadow:0 4px 10px rgba(0,0,0,0.20);">
            <div style="font-size:16px; font-weight:700; color:#FFD000; letter-spacing:0.5px;">{cls}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:auto; padding-top:20px; border-top:1px solid rgba(255,208,0,0.14);">
        <div style="font-size:13px; letter-spacing:1px; color:#7A5020; text-transform:uppercase; line-height:1.6; font-weight:700;">
            Universidad del Quindío<br>Grupo de Investigación<br>Plaguicidas y Salud
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="position:relative; width:100%; min-height:420px; overflow:hidden; border-radius:12px; display:flex; align-items:center; border:1px solid rgba(255,208,0,0.30); box-shadow:0 4px 28px rgba(42,20,0,0.10); margin-bottom:2rem;">

<div style="position:absolute; inset:0; background:#FFFCE8;
  background-image:
    radial-gradient(ellipse 65% 80% at 85% 50%, rgba(255,208,0,0.28) 0%, transparent 70%),
    radial-gradient(ellipse 45% 60% at 10% 30%, rgba(255,252,232,0.95) 0%, transparent 65%),
    radial-gradient(ellipse 30% 50% at 50% 90%, rgba(245,168,0,0.10) 0%, transparent 60%);
"></div>

<div style="position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg, transparent, #FFD000, #F5A800, #FFD000, transparent); z-index:6;"></div>

<div style="position:absolute; right:60px; top:50%; transform:translateY(-50%);
  width:280px; height:280px; opacity:0.14; pointer-events:none; z-index:5;">
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"
  stroke="#F5A800" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none"
  style="width:100%; height:100%;">
<path d="M92,60 Q80,45 92,35 M108,60 Q120,45 108,35" />
<path d="M88,72 A 12 12 0 0 1 112,72" />
<circle cx="100" cy="95" r="16" />
<path d="M85,110 Q 75,140 100,165 Q 125,140 115,110 Z" />
<path d="M83,125 Q 100,135 117,125 M88,140 Q 100,150 112,140" />
<path d="M 82,90 L 25,50 L 15,65 L 60,105 L 82,100 Z" />
<path d="M 25,50 L 50,90 L 15,65 M 50,90 L 60,105" />
<path d="M 78,108 L 40,125 L 55,140 L 82,118 Z" />
<path d="M 40,125 L 75,114" />
<path d="M 118,90 L 175,50 L 185,65 L 140,105 L 118,100 Z" />
<path d="M 175,50 L 150,90 L 185,65 M 150,90 L 140,105" />
<path d="M 122,108 L 160,125 L 145,140 L 118,118 Z" />
<path d="M 160,125 L 125,114" />
</svg>
</div>

<div style="position:relative; z-index:10; padding:40px 36px; width:100%; max-width:680px;">
<div style="font-size:13px; font-weight:700; letter-spacing:2px; color:#C87800; text-transform:uppercase; margin-bottom:14px; opacity:0.9;">
Calorimetría diferencial de barrido · Machine Learning
</div>
<div class="logo-brand" style="font-size:clamp(40px,6vw,84px); font-weight:900; line-height:0.90; color:#2A1400; letter-spacing:-1.5px; margin:0 0 6px 0;">
Honey<span style="color:#F5A800; font-style:italic;">Check</span>
</div>
<div style="width:72px; height:3px; margin:18px 0; background:linear-gradient(90deg,#FFD000,#F5A800,transparent); border-radius:1px;"></div>
<p style="font-size:clamp(15px,2.5vw,20px); font-weight:600; color:#4A2800; margin:0 0 24px 0; max-width:500px; line-height:1.55;">
Detección de adulteración y trazabilidad geográfica de mieles colombianas mediante análisis DSC y modelos de clasificación supervisada.
</p>
<div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
<span style="font-size:11px; font-weight:700; letter-spacing:1px; color:#4A2800; border:1px solid rgba(255,208,0,0.55); padding:6px 12px; border-radius:2px; background:rgba(255,255,255,0.75); text-transform:uppercase;">SISTEMA JERÁRQUICO V2.0</span>
<span style="font-size:11px; font-weight:700; letter-spacing:1px; color:#4A2800; border:1px solid rgba(255,208,0,0.55); padding:6px 12px; border-radius:2px; background:rgba(255,255,255,0.75); text-transform:uppercase;">UNIVERSIDAD DEL QUINDÍO</span>
<span style="font-size:11px; font-weight:700; letter-spacing:1px; color:#4A2800; border:1px solid rgba(255,208,0,0.55); padding:6px 12px; border-radius:2px; background:rgba(255,255,255,0.75); text-transform:uppercase;">NETZSCH DSC 214 POLYMA</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  STATS ROW
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-row">
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">98.4</span><span class="stat-unit">%</span></div>
        <div class="stat-label">Precisión Autenticidad</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">82.0</span><span class="stat-unit">%</span></div>
        <div class="stat-label">Precisión Origen</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">62</span></div>
        <div class="stat-label">Muestras Entrenamiento</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">p&lt;0.001</span></div>
        <div class="stat-label">Confianza Estadística</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CARGAR MODELOS
# ═══════════════════════════════════════════════════════════════════
m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()
if m_auth is not None:
    st.markdown('''
    <div class="status-bar">
        <div class="status-dot"></div>
        <span class="status-text">Sistemas de clasificación calibrados — Entorno listo para análisis</span>
    </div>''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="status-bar status-bar-warn">
        <div class="status-dot"></div>
        <span class="status-text">Modelos no encontrados — Modo demostración visual activo</span>
    </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  UPLOAD
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-number">01</span>
    <h2 class="section-title">Carga de Termogramas</h2>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

archivos = st.file_uploader(
    "Archivos .txt del NETZSCH DSC 214 Polyma",
    type=["txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if not archivos:
    st.markdown("""
    <div style="padding:52px 20px; text-align:center; border:1.5px dashed rgba(255,208,0,0.30); border-radius:8px; background:rgba(255,255,255,0.65); margin-top:16px;">
        <div style="font-size:44px; margin-bottom:18px; opacity:0.38; color:#F5A800;">⬡</div>
        <div style="font-size:24px; font-weight:700; color:#906030; margin-bottom:10px;">Sistema en espera</div>
        <div style="font-size:12px; font-weight:600; letter-spacing:2.5px; color:#C8A050; text-transform:uppercase;">
            Seleccione o arrastre archivos .txt para iniciar el procesamiento
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <div class="footer-left">HONEYCHECK · SISTEMA JERÁRQUICO V2.0 · © 2024</div>
        <div class="footer-right">Universidad del Quindío <span class="footer-gold">— Grupo Plaguicidas y Salud</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════
#  ANÁLISIS
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="section-header">
    <span class="section-number">02</span>
    <h2 class="section-title">Informe Analítico — {len(archivos)} muestra{'s' if len(archivos)!=1 else ''}</h2>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

for i, archivo in enumerate(archivos):

    st.markdown(f"""
    <div class="sample-header">
        <div class="sample-dot"></div>
        <div class="sample-name">{archivo.name.upper()}</div>
        <div style="margin-left:auto; font-size:11px; font-weight:600; color:#C8A050; letter-spacing:2px; text-transform:uppercase;">
            MUESTRA {str(i+1).zfill(2)}
        </div>
    </div>
    <div class="sample-body">
    """, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(archivo.read())
        tmp_path = tmp.name

    try:
        df_raw     = leer_dsc(tmp_path)
        dsc_interp = interpolar(df_raw)

        if dsc_interp is None:
            st.error("El archivo no contiene el rango térmico completo requerido (−35 a 195 °C).")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        if m_auth is None:
            st.info("Modelos no disponibles — mostrando solo el termograma.")
            fig = graficar_termograma(dsc_interp, archivo.name, "#F5A800")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        feats  = extraer_features(dsc_interp)
        X_feat = sc_auth.transform(np.array(list(feats.values())).reshape(1, -1))
        pred   = m_auth.predict(X_feat)[0]
        probs  = m_auth.predict_proba(X_feat)[0]

        st.markdown('<div class="level-badge">FASE 01 — Evaluación de Autenticidad</div>', unsafe_allow_html=True)

        css_map  = ["resultado-real","resultado-adulterada","resultado-mezcla"]
        icon_map = ["✦", "⚠", "◉"]
        st.markdown(f"""
        <div class="resultado-base {css_map[pred]}">
            <div>
                <div class="res-label">Clasificación</div>
                <div class="res-value">{CLASES_AUTH[pred]}</div>
            </div>
            <div class="res-icon">{icon_map[pred]}</div>
        </div>
        """, unsafe_allow_html=True)

        col_g, col_m = st.columns([2.5, 1.2])
        with col_g:
            fig = graficar_termograma(dsc_interp, archivo.name, COLORES_AUTH[pred])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_m:
            bars_html = '<div class="prob-container"><div class="prob-title">Distribución probabilística</div>'
            for cls, prob, col in zip(CLASES_AUTH, probs, COLORES_AUTH):
                bars_html += f"""
                <div class="prob-row">
                    <div class="prob-row-top">
                        <span class="prob-cls">{cls}</span>
                        <span class="prob-pct" style="color:{col};">{prob*100:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div style="background:{col}; width:{prob*100:.1f}%; height:100%; border-radius:3px;"></div>
                    </div>
                </div>"""
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

        if pred == 0:
            st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="level-badge">FASE 02 — Trazabilidad Geográfica</div>', unsafe_allow_html=True)

            X_norm   = sc_geo.transform(dsc_interp.reshape(1, -1))
            X_pca    = pca_geo.transform(X_norm)
            pred_geo = m_geo.predict(X_pca)[0]
            prob_geo = m_geo.predict_proba(X_pca)[0]

            cols_geo = st.columns(2)
            for col_ui, cls, prob, color in zip(cols_geo, CLASES_GEO, prob_geo, COLORES_GEO):
                activo = prob > 0.5
                cls_card = "active" if activo else "inactive"
                with col_ui:
                    st.markdown(f"""
                    <div class="geo-card {cls_card}">
                        <div class="geo-region">{cls}</div>
                        <div class="geo-pct" style="color:{color};">{prob*100:.1f}%</div>
                        {"<div class='geo-winner-tag'>⬡ Origen determinado</div>" if activo else ""}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar el espectro térmico: {e}")
    finally:
        os.unlink(tmp_path)

    st.markdown("</div>", unsafe_allow_html=True)

    if i < len(archivos) - 1:
        st.markdown('<div class="sample-divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-left">HONEYCHECK · SISTEMA JERÁRQUICO V2.0 · © 2024</div>
    <div class="footer-right">Universidad del Quindío <span class="footer-gold">— Grupo Plaguicidas y Salud</span></div>
</div>
""", unsafe_allow_html=True)
