
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

# ═══════════════════════════════════════════════════════════════════
#  CSS: AMBER LIGHT — Gradiente crema-ámbar, patrón abejas, sidebar oscuro
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Cormorant+Garamond:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        /* Paleta principal — luz ámbar */
        --cream-light:   #FFFDF4;
        --cream-base:    #FFF8E1;
        --cream-mid:     #FDEFC3;
        --amber-soft:    #F5D47A;
        --amber-mid:     #E8A820;
        --amber-deep:    #C8820A;
        --amber-dark:    #8B5E00;
        --amber-darkest: #5C3D00;

        /* Textos */
        --text-primary:  #3D2200;
        --text-body:     #5C3A0A;
        --text-muted:    #9A7040;
        --text-dim:      #C4A46A;

        /* Tarjetas */
        --card-bg:       #FFFFFF;
        --card-border:   rgba(200,130,10,0.35);
        --card-shadow:   rgba(139,94,0,0.10);

        /* Sidebar — oscuro como contraste */
        --sidebar-bg:    #0F0A03;
        --sidebar-surf:  #1A1205;
        --sidebar-border:rgba(192,154,46,0.18);
        --sidebar-gold:  #F0C040;
        --sidebar-text:  #F5EDD6;
        --sidebar-muted: #7A6A3A;

        /* UI */
        --border-gold:   rgba(200,130,10,0.30);
        --green-ok:      #2E7D40;
        --red-bad:       #8B2A2A;
        --amber-mix:     #8B6020;

        /* Patrón abeja geométrica SVG embebido */
        --bee-pattern: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' stroke='%23C8820A' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round' fill='none' opacity='0.08'%3E%3Cpath d='M92,60 Q80,45 92,35 M108,60 Q120,45 108,35'/%3E%3Cpath d='M88,72 A 12 12 0 0 1 112,72'/%3E%3Ccircle cx='100' cy='95' r='16'/%3E%3Cpath d='M85,110 Q 75,140 100,165 Q 125,140 115,110 Z'/%3E%3Cpath d='M83,125 Q 100,135 117,125 M88,140 Q 100,150 112,140'/%3E%3Cpath d='M 82,90 L 25,50 L 15,65 L 60,105 L 82,100 Z'/%3E%3Cpath d='M 25,50 L 50,90 L 15,65 M 50,90 L 60,105'/%3E%3Cpath d='M 78,108 L 40,125 L 55,140 L 82,118 Z'/%3E%3Cpath d='M 40,125 L 75,114'/%3E%3Cpath d='M 118,90 L 175,50 L 185,65 L 140,105 L 118,100 Z'/%3E%3Cpath d='M 175,50 L 150,90 L 185,65 M 150,90 L 140,105'/%3E%3Cpath d='M 122,108 L 160,125 L 145,140 L 118,118 Z'/%3E%3Cpath d='M 160,125 L 125,114'/%3E%3C/svg%3E");
    }

    /* ── Base ─────────────────────────────────────────────────── */
    .stApp {
        background-color: var(--cream-base) !important;
        background-image:
            var(--bee-pattern),
            linear-gradient(
                160deg,
                var(--cream-light) 0%,
                var(--cream-base)  35%,
                var(--cream-mid)   65%,
                var(--amber-soft)  100%
            ) !important;
        background-repeat: repeat, no-repeat !important;
        background-size: 120px 120px, cover !important;
        background-attachment: fixed !important;
        color: var(--text-primary);
        font-family: 'Cormorant Garamond', serif;
    }
    
    /* CORRECCIÓN 1: Damos un poco de espacio arriba para que el texto no se corte */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* ── Sidebar oscuro — contraste elegante ───────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--sidebar-gold), transparent);
    }
    [data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--sidebar-gold) !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* ── Textos globales ───────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        color: var(--text-primary) !important;
    }
    p, span, div, label {
        font-family: 'Cormorant Garamond', serif;
        color: var(--text-body);
    }

    /* ── Upload zone ───────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.75) !important;
        border: 1.5px dashed var(--border-gold) !important;
        border-radius: 6px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--amber-deep) !important;
        background-color: rgba(255,248,225,0.9) !important;
        box-shadow: 0 0 20px rgba(200,130,10,0.12);
    }

    /* ── Ocultar alerts nativos de Streamlit ──────────────────── */
    [data-testid="stAlert"] { display: none !important; }

    /* ── Barra de estado ámbar custom ──────────────────────────── */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 22px;
        background: linear-gradient(90deg, rgba(232,168,32,0.12), rgba(245,212,122,0.06), transparent);
        border-left: 3px solid var(--amber-mid);
        border-radius: 0 4px 4px 0;
        margin: 20px 0 8px 0;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--amber-mid);
        box-shadow: 0 0 8px rgba(232,168,32,0.6);
        flex-shrink: 0;
        animation: pulse-dot 2.5s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(232,168,32,0.6); }
        50%        { opacity: 0.55; box-shadow: 0 0 14px rgba(232,168,32,0.25); }
    }
    .status-text {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2.5px;
        color: var(--amber-dark);
        text-transform: uppercase;
    }
    .status-bar-warn { border-left-color: var(--amber-dark); }
    .status-bar-warn .status-dot { background: var(--amber-dark); animation: none; }

    /* ── Scrollbar ─────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--cream-mid); }
    ::-webkit-scrollbar-thumb { background: var(--amber-soft); border-radius: 3px; }

    /* ── STAT CARDS ────────────────────────────────────────────── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 32px 0 40px 0;
    }
    .stat-cell {
        background: var(--card-bg);
        padding: 26px 20px;
        text-align: center;
        border: 1px solid var(--card-border);
        border-radius: 6px;
        box-shadow: 0 2px 16px var(--card-shadow);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-cell::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--amber-soft), var(--amber-mid), var(--amber-soft));
    }
    .stat-cell:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(139,94,0,0.14);
    }
    
    /* CORRECCIÓN 2: Ajuste de tamaño y alineación vertical de los números */
    .stat-value {
        display: flex;
        align-items: center; /* Centrado vertical en lugar de baselínea */
        justify-content: center;
        gap: 4px;
        line-height: 1;
        min-height: 55px; /* Altura fija para que todos cuadren igual */
    }
    .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 42px; /* Tamaño unificado para que p<0.001 no se desborde */
        font-weight: 700;
        font-style: normal;
        color: var(--amber-deep);
        letter-spacing: -1px;
        line-height: 1;
        font-variant-numeric: lining-nums; /* Asegura que los números se alineen bien */
    }
    .stat-unit {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        font-style: normal;
        color: var(--amber-deep);
        margin-top: 8px; /* Ajuste visual para el % */
        letter-spacing: 0;
    }
    .stat-label {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 2.5px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-top: 14px;
    }

    /* ── SECTION HEADER ────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 40px 0 24px 0;
        padding-bottom: 14px;
        border-bottom: 1.5px solid var(--amber-soft);
    }
    .section-number {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--amber-mid);
        letter-spacing: 2px;
        background: rgba(232,168,32,0.12);
        padding: 4px 10px;
        border-radius: 2px;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 22px;
        color: var(--text-primary) !important;
        margin: 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--amber-soft), transparent);
    }

    /* ── SAMPLE CARD ───────────────────────────────────────────── */
    .sample-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 24px;
        background: linear-gradient(90deg, rgba(255,248,225,0.9), rgba(255,255,255,0.7));
        border: 1.5px solid var(--card-border);
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        margin-top: 28px;
    }
    .sample-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: var(--amber-mid);
        box-shadow: 0 0 10px rgba(232,168,32,0.5);
        flex-shrink: 0;
    }
    .sample-name {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        color: var(--text-primary);
        letter-spacing: 1px;
    }
    .sample-body {
        border: 1.5px solid var(--card-border);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 28px;
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 24px var(--card-shadow);
    }

    /* ── LEVEL BADGE ───────────────────────────────────────────── */
    .level-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--amber-darkest);
        text-transform: uppercase;
        padding: 6px 16px;
        border: 1px solid var(--amber-mid);
        border-radius: 2px;
        background: rgba(245,212,122,0.20);
        margin: 16px 0 14px 0;
    }
    .level-badge::before {
        content: '⬡';
        font-size: 10px;
        color: var(--amber-mid);
    }

    /* ── RESULTADO CARDS ───────────────────────────────────────── */
    .resultado-base {
        padding: 22px 28px;
        border-radius: 6px;
        margin: 10px 0 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1.5px solid;
    }
    .resultado-real {
        background: linear-gradient(135deg, rgba(46,125,64,0.10), rgba(240,255,245,0.8));
        border-color: rgba(46,125,64,0.35);
    }
    .resultado-real .res-icon { color: #2E7D40; }
    .resultado-adulterada {
        background: linear-gradient(135deg, rgba(139,42,42,0.10), rgba(255,245,245,0.8));
        border-color: rgba(139,42,42,0.35);
    }
    .resultado-adulterada .res-icon { color: #C0392B; }
    .resultado-mezcla {
        background: linear-gradient(135deg, rgba(200,130,10,0.12), rgba(255,253,240,0.8));
        border-color: rgba(200,130,10,0.35);
    }
    .resultado-mezcla .res-icon { color: var(--amber-deep); }
    .res-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .res-value {
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
    }
    .res-icon { font-size: 36px; opacity: 0.85; }

    /* ── PROBABILITY BARS ──────────────────────────────────────── */
    .prob-container {
        background: rgba(255,255,255,0.85);
        border: 1.5px solid var(--card-border);
        border-radius: 6px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 2px 12px var(--card-shadow);
    }
    .prob-title {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .prob-row { margin: 14px 0; }
    .prob-row-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 7px;
    }
    .prob-cls {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'Cormorant Garamond', serif;
    }
    .prob-pct {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        font-weight: 500;
    }
    .prob-track {
        background: var(--cream-mid);
        border-radius: 3px;
        height: 5px;
        width: 100%;
        overflow: hidden;
    }

    /* ── GEO CARDS ─────────────────────────────────────────────── */
    .geo-card {
        padding: 28px;
        border-radius: 8px;
        border: 1.5px solid var(--card-border);
        background: rgba(255,255,255,0.85);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
        box-shadow: 0 2px 12px var(--card-shadow);
    }
    .geo-card.active {
        border-color: var(--amber-mid);
        background: linear-gradient(160deg, rgba(255,255,255,0.95), rgba(253,239,195,0.6));
        box-shadow: 0 6px 28px rgba(200,130,10,0.15);
    }
    .geo-card.active::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--amber-soft), var(--amber-mid), var(--amber-soft));
    }
    .geo-card.inactive { opacity: 0.40; }
    .geo-region {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 10px;
    }
    .geo-pct {
        font-family: 'Playfair Display', serif;
        font-size: 52px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 6px;
    }
    .geo-winner-tag {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        color: var(--amber-deep);
        text-transform: uppercase;
        margin-top: 10px;
    }

    /* ── DIVIDER ───────────────────────────────────────────────── */
    .sample-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--amber-soft), transparent);
        margin: 48px 0;
    }

    /* ── FOOTER ────────────────────────────────────────────────── */
    .footer {
        margin-top: 80px;
        padding: 28px 0 16px 0;
        border-top: 1.5px solid var(--amber-soft);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .footer-left {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        color: var(--text-muted);
        letter-spacing: 1.5px;
    }
    .footer-right {
        font-family: 'Cormorant Garamond', serif;
        font-size: 15px;
        color: var(--text-muted);
    }
    .footer-gold { color: var(--amber-deep); font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════
T_MIN, T_MAX, N_PTS = -30.0, 190.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#2E7D40", "#C0392B", "#C8820A"]
COLORES_GEO  = ["#C89A2E", "#8B5A2B"]

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
        feats["pico1_temp"]  = 0.0
        feats["pico1_valor"] = 0.0
        feats["pico1_prom"]  = 0.0
    umbral    = 0.1 * feats["dsc_min"]
    onset_idx = np.argmax(dsc_curve < umbral)
    feats["onset_temp"] = T[onset_idx] if onset_idx > 0 else 0.0
    return feats

# ═══════════════════════════════════════════════════════════════════
#  GRÁFICA TERMOGRAMA — Tema claro ámbar
# ═══════════════════════════════════════════════════════════════════
def graficar_termograma(dsc_curve, nombre, color_linea):
    bg      = "#FFFDF4"
    grid_c  = "#EDD89A"
    tick_c  = "#9A7040"
    label_c = "#7A5020"

    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=bg)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # Zonas térmicas
    ax.axvspan(-30,  30, alpha=0.07, color="#5599CC", zorder=0)
    ax.axvspan( 30, 100, alpha=0.07, color="#E8A820", zorder=0)
    ax.axvspan(100, 190, alpha=0.07, color="#CC5533", zorder=0)

    for x_line in [-30, 30, 100, 190]:
        ax.axvline(x_line, color=grid_c, linewidth=0.6, linestyle=":", zorder=1)

    ax.axhline(0, color="#C4A46A", linewidth=0.9, linestyle="-", zorder=2)

    ax.plot(T_GRILLA, dsc_curve, color=color_linea, linewidth=2.0,
            zorder=5, solid_capstyle="round")

    ax.fill_between(T_GRILLA, dsc_curve,
                    where=dsc_curve < 0,
                    alpha=0.15, color=color_linea, zorder=3)
    ax.fill_between(T_GRILLA, dsc_curve,
                    where=dsc_curve >= 0,
                    alpha=0.06, color=color_linea, zorder=3)

    for x_c, lbl, c_lbl in [
        (  0, "FUSIÓN",        "#3377AA"),
        ( 65, "TRANSICIÓN",    "#C8820A"),
        (145, "CARAMELIZACIÓN","#AA4422"),
    ]:
        ax.text(x_c, 1.0, lbl,
                ha="center", va="top",
                fontsize=7.5, color=c_lbl, alpha=0.65,
                fontfamily="monospace", fontweight="bold",
                transform=ax.get_xaxis_transform(),
                clip_on=True)

    ax.set_xlabel("Temperatura  (°C)", fontsize=10, color=label_c, labelpad=8)
    ax.set_ylabel("Flujo de calor  (mW/mg)", fontsize=10, color=label_c, labelpad=8)
    ax.tick_params(colors=tick_c, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
        spine.set_linewidth(0.8)
    ax.grid(True, color=grid_c, linewidth=0.4, linestyle="-", alpha=0.6)
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
    <div style="padding: 28px 0 20px 0; text-align:center; border-bottom: 1px solid rgba(192,154,46,0.15); margin-bottom: 24px;">
        <div style="font-family:'Playfair Display',serif; font-size:28px; font-weight:900; color:#F0C040; letter-spacing:-0.5px;">
            Honey<span style="font-style:italic;">Check</span>
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:4px; color:#5A4A2A; text-transform:uppercase; margin-top:4px;">
            Sistema Jerárquico V2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 28px;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:3px; color:#5A4A2A; text-transform:uppercase; margin-bottom:14px;">
            Arquitectura
        </div>
        <div style="padding: 14px 16px; background: rgba(192,154,46,0.06); border-left: 2px solid #C89A2E; margin-bottom: 8px; border-radius: 0 4px 4px 0;">
            <div style="font-family:'DM Mono',monospace; font-size:9px; color:#8B6914; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 01</div>
            <div style="font-size:15px; font-weight:600; color:#F5EDD6; font-family:'Cormorant Garamond',serif;">Autenticidad</div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5A4A2A; margin-top:3px;">SVM Lineal · Acc 98.39%</div>
        </div>
        <div style="padding: 14px 16px; background: rgba(192,154,46,0.04); border-left: 2px solid #8B6914; margin-bottom: 8px; border-radius: 0 4px 4px 0;">
            <div style="font-family:'DM Mono',monospace; font-size:9px; color:#5A4A2A; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 02</div>
            <div style="font-size:15px; font-weight:600; color:#F5EDD6; font-family:'Cormorant Garamond',serif;">Origen Geográfico</div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5A4A2A; margin-top:3px;">SVM+PCA · Acc 82.00%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ICONOS_AUTH = ["◆", "◆", "◆"]
    COLORES_SIDE = ["#4CAF7D", "#E05555", "#F0C040"]
    st.markdown("""<div style="margin-bottom:20px;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:3px; color:#5A4A2A; text-transform:uppercase; margin-bottom:12px;">Clases detectables</div>
    """, unsafe_allow_html=True)
    for cls, col, ico in zip(CLASES_AUTH, COLORES_SIDE, ICONOS_AUTH):
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
            <span style="color:{col}; font-size:7px;">{ico}</span>
            <span style="font-size:14px; color:#C8B89A; font-family:'Cormorant Garamond',serif;">{cls}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div style="margin-top:20px; margin-bottom:20px;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:3px; color:#5A4A2A; text-transform:uppercase; margin-bottom:12px;">Orígenes geográficos</div>
    """, unsafe_allow_html=True)
    for cls, col in zip(CLASES_GEO, COLORES_GEO):
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 7px 0 7px 12px; border-bottom: 1px solid rgba(255,255,255,0.04);">
            <span style="color:{col}; font-size:7px;">◆</span>
            <span style="font-size:13px; color:#8A7A5A; font-family:'Cormorant Garamond',serif;">{cls}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:auto; padding-top:20px; border-top:1px solid rgba(192,154,46,0.1);">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; color:#3D2E0A; text-transform:uppercase; line-height:1.8;">
            Universidad del Quindío<br>Grupo de Investigación<br>Ciencia e Ingeniería de Alimentos
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  HERO — Panal SVG claro (Abeja Geométrica y bordes redondeados)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="position:relative; width:100%; min-height:420px; overflow:hidden; border-radius:12px; display:flex; align-items:center; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 2rem;">

<!-- Fondo gradiente -->
<div style="position:absolute; inset:0; background-color:#FFFDF4; background-image: radial-gradient(ellipse 65% 80% at 85% 50%, rgba(245,212,122,0.50) 0%, transparent 70%), radial-gradient(ellipse 45% 60% at 10% 30%, rgba(255,245,210,0.70) 0%, transparent 65%), radial-gradient(ellipse 30% 50% at 50% 90%, rgba(232,168,32,0.12) 0%, transparent 60%);"></div>

<!-- Línea dorada izquierda -->
<div style="position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg, transparent, #E8A820, #C8820A, transparent); z-index:6;"></div>

<!-- Abeja Geométrica Ilustrada — derecha -->
<div style="position:absolute; right:60px; top:50%; transform:translateY(-50%); width:340px; height:340px; opacity:0.18; pointer-events:none; z-index:5;">
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" stroke="#C8820A" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none" style="width:100%; height:100%;">
<!-- Antenas -->
<path d="M92,60 Q80,45 92,35 M108,60 Q120,45 108,35" />
<!-- Cabeza -->
<path d="M88,72 A 12 12 0 0 1 112,72" />
<!-- Tórax -->
<circle cx="100" cy="95" r="16" />
<!-- Abdomen -->
<path d="M85,110 Q 75,140 100,165 Q 125,140 115,110 Z" />
<!-- Rayas abdomen -->
<path d="M83,125 Q 100,135 117,125 M88,140 Q 100,150 112,140" />
<!-- Ala Sup Izquierda -->
<path d="M 82,90 L 25,50 L 15,65 L 60,105 L 82,100 Z" />
<path d="M 25,50 L 50,90 L 15,65 M 50,90 L 60,105" />
<!-- Ala Inf Izquierda -->
<path d="M 78,108 L 40,125 L 55,140 L 82,118 Z" />
<path d="M 40,125 L 75,114" />
<!-- Ala Sup Derecha -->
<path d="M 118,90 L 175,50 L 185,65 L 140,105 L 118,100 Z" />
<path d="M 175,50 L 150,90 L 185,65 M 150,90 L 140,105" />
<!-- Ala Inf Derecha -->
<path d="M 122,108 L 160,125 L 145,140 L 118,118 Z" />
<path d="M 160,125 L 125,114" />
</svg>
</div>

<!-- Contenido del hero -->
<div style="position:relative; z-index:10; padding:52px 56px; width:100%; max-width:680px;">
<div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:4px; color:#C8820A; text-transform:uppercase; margin-bottom:18px; opacity:0.9;">
Calorimetría diferencial de barrido · Machine Learning
</div>
<div style="font-family:'Playfair Display',serif; font-size:clamp(52px,6vw,84px); font-weight:900; line-height:0.90; color:#3D2200; letter-spacing:-1.5px; margin:0 0 6px 0;">
Honey<span style="color:#C8820A; font-style:italic;">Check</span>
</div>
<div style="width:72px; height:2px; margin:22px 0; background:linear-gradient(90deg,#E8A820,#F5D47A,transparent); border-radius:1px;"></div>
<p style="font-family:'Cormorant Garamond',serif; font-size:19px; font-weight:500; color:#5C3A0A; margin:0 0 30px 0; max-width:500px; line-height:1.55;">
Detección de adulteración y trazabilidad geográfica de mieles colombianas mediante análisis DSC y modelos de clasificación supervisada.
</p>
<div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
<span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">Sistema Jerárquico V2.0</span>
<span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">Universidad del Quindío</span>
<span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">NETZSCH DSC 214 Polyma</span>
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
        <div class="stat-label">Precisión · Autenticidad</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">82.0</span><span class="stat-unit">%</span></div>
        <div class="stat-label">Precisión · Origen</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">62</span></div>
        <div class="stat-label">Muestras de entrenamiento</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value"><span class="stat-num">p&lt;0.001</span></div>
        <div class="stat-label">Confianza estadística</div>
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
    <div style="padding: 52px; text-align:center; border: 1.5px dashed rgba(200,130,10,0.30); border-radius:8px; background: rgba(255,255,255,0.55); margin-top: 16px; backdrop-filter: blur(4px);">
        <div style="font-size:44px; margin-bottom:18px; opacity:0.5;">⬡</div>
        <div style="font-family:'Playfair Display',serif; font-size:22px; color:#8B6020; margin-bottom:10px;">Sistema en espera</div>
        <div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2.5px; color:#C4A46A; text-transform:uppercase;">Seleccione o arrastre archivos .txt para iniciar el procesamiento</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <div class="footer-left">HONEYCHECK · SISTEMA JERÁRQUICO V2.0 · © 2024</div>
        <div class="footer-right">Universidad del Quindío <span class="footer-gold">— Grupo CIA</span></div>
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
        <div style="margin-left:auto; font-family:'DM Mono',monospace; font-size:10px; color:#C4A46A; letter-spacing:2px;">
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
            st.error("El archivo no contiene el rango térmico completo requerido (−30 a 190 °C).")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        if m_auth is None:
            st.info("Modelos no disponibles — mostrando solo el termograma.")
            fig = graficar_termograma(dsc_interp, archivo.name, "#C89A2E")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        # ── NIVEL 1: Autenticidad ────────────────────────────────
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

        # ── NIVEL 2: Origen geográfico ───────────────────────────
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
    <div class="footer-right">Universidad del Quindío <span class="footer-gold">— Grupo CIA</span></div>
</div>
""", unsafe_allow_html=True)
