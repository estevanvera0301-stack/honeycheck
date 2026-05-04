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
#  CSS: DARK LUXURY — Fondo negro, dorado, tipografía refinada
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Cormorant+Garamond:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --gold-bright:  #F0C040;
        --gold-mid:     #C89A2E;
        --gold-dark:    #8B6914;
        --gold-muted:   #3D2E0A;
        --obsidian:     #080808;
        --surface-1:    #101010;
        --surface-2:    #161616;
        --surface-3:    #1E1E1E;
        --surface-4:    #252525;
        --border:       rgba(192,154,46,0.18);
        --border-glow:  rgba(240,192,64,0.45);
        --text-primary: #F5EDD6;
        --text-muted:   #8A7A5A;
        --text-dim:     #4A3F2A;
        --green-ok:     #2E7D55;
        --red-bad:      #8B2A2A;
        --amber-mix:    #8B6020;
    }

    /* ── Base ─────────────────────────────────────────────────── */
    .stApp {
        background-color: var(--obsidian) !important;
        color: var(--text-primary);
        font-family: 'Cormorant Garamond', serif;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* ── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--surface-1) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold-bright), transparent);
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--gold-bright) !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* ── Textos globales ───────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        color: var(--text-primary) !important;
    }
    p, span, div, label { font-family: 'Cormorant Garamond', serif; }

    /* ── Upload zone ───────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        transition: all 0.4s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--gold-bright) !important;
        box-shadow: 0 0 30px rgba(240,192,64,0.08), inset 0 0 20px rgba(240,192,64,0.03);
    }

    /* ── Alerts ────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--gold-mid) !important;
        border-radius: 2px !important;
        color: var(--text-primary) !important;
    }

    /* ── Scrollbar ─────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--obsidian); }
    ::-webkit-scrollbar-thumb { background: var(--gold-dark); border-radius: 2px; }

    /* ══════════════════════════════════════════════════════════════
       COMPONENTES CUSTOM
    ══════════════════════════════════════════════════════════════ */

    /* ── HERO ──────────────────────────────────────────────────── */
    .hero-wrapper {
        position: relative;
        width: 100%;
        min-height: 480px;
        overflow: hidden;
        margin: -1rem -1rem 0 -1rem;
        display: flex;
        align-items: flex-end;
    }
    .hero-bg {
        position: absolute;
        inset: 0;
        background-image: url('https://images.unsplash.com/photo-1587049352851-8d4e89133924?w=1800&q=90&fit=crop');
        background-size: cover;
        background-position: center 35%;
        filter: brightness(0.45) saturate(0.7);
        transform: scale(1.02);
        transition: transform 8s ease;
    }
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(
            to bottom,
            rgba(8,8,8,0.15) 0%,
            rgba(8,8,8,0.0) 30%,
            rgba(8,8,8,0.6) 65%,
            rgba(8,8,8,0.97) 100%
        );
    }
    .hero-grain {
        position: absolute;
        inset: 0;
        opacity: 0.035;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        background-repeat: repeat;
        background-size: 128px 128px;
    }
    .hero-content {
        position: relative;
        z-index: 10;
        padding: 0 56px 52px 56px;
        width: 100%;
    }
    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 4px;
        color: var(--gold-mid);
        text-transform: uppercase;
        margin-bottom: 16px;
        opacity: 0.9;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(48px, 6vw, 80px);
        font-weight: 900;
        line-height: 0.92;
        color: #FFFFFF;
        letter-spacing: -1px;
        margin: 0 0 4px 0;
    }
    .hero-title-gold {
        color: var(--gold-bright);
        font-style: italic;
    }
    .hero-subtitle {
        font-family: 'Cormorant Garamond', serif;
        font-size: 20px;
        font-weight: 400;
        color: rgba(245,237,214,0.75);
        margin: 18px 0 28px 0;
        max-width: 520px;
        line-height: 1.5;
    }
    .hero-divider {
        width: 64px;
        height: 1px;
        background: linear-gradient(90deg, var(--gold-bright), transparent);
        margin-bottom: 20px;
    }
    .hero-badges {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        align-items: center;
    }
    .hero-badge {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        color: var(--gold-mid);
        border: 1px solid rgba(192,154,46,0.3);
        padding: 7px 16px;
        border-radius: 1px;
        text-transform: uppercase;
        background: rgba(8,8,8,0.5);
        backdrop-filter: blur(8px);
    }

    /* ── STAT CARDS ────────────────────────────────────────────── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        margin: 32px 0 40px 0;
    }
    .stat-cell {
        background: var(--surface-1);
        padding: 28px 24px;
        text-align: center;
        position: relative;
        transition: background 0.3s;
    }
    .stat-cell:hover { background: var(--surface-2); }
    .stat-cell::after {
        content: '';
        position: absolute;
        bottom: 0; left: 50%;
        transform: translateX(-50%) scaleX(0);
        width: 40px; height: 1px;
        background: var(--gold-bright);
        transition: transform 0.3s;
    }
    .stat-cell:hover::after { transform: translateX(-50%) scaleX(1); }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 36px;
        font-weight: 700;
        color: var(--gold-bright);
        line-height: 1;
    }
    .stat-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-top: 8px;
    }

    /* ── SECTION HEADER ────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 40px 0 24px 0;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .section-number {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--gold-dark);
        letter-spacing: 2px;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 22px;
        color: var(--text-primary);
        margin: 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }

    /* ── SAMPLE CARD ───────────────────────────────────────────── */
    .sample-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 24px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-bottom: none;
        border-radius: 2px 2px 0 0;
        margin-top: 28px;
    }
    .sample-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--gold-bright);
        box-shadow: 0 0 8px var(--gold-bright);
        flex-shrink: 0;
    }
    .sample-name {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        color: var(--text-primary);
        letter-spacing: 1px;
    }
    .sample-body {
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 2px 2px;
        padding: 28px;
        background: var(--surface-1);
    }

    /* ── LEVEL BADGE ───────────────────────────────────────────── */
    .level-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--gold-mid);
        text-transform: uppercase;
        padding: 6px 14px;
        border: 1px solid rgba(192,154,46,0.25);
        border-radius: 1px;
        background: rgba(192,154,46,0.05);
        margin: 16px 0 12px 0;
    }
    .level-badge::before {
        content: '';
        width: 4px; height: 4px;
        border-radius: 50%;
        background: var(--gold-bright);
    }

    /* ── RESULTADO CARDS ───────────────────────────────────────── */
    .resultado-base {
        padding: 22px 28px;
        border-radius: 2px;
        margin: 10px 0 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid;
    }
    .resultado-real {
        background: linear-gradient(135deg, rgba(46,125,85,0.15), rgba(46,125,85,0.05));
        border-color: rgba(46,125,85,0.4);
    }
    .resultado-real .res-icon { color: #4CAF7D; }
    .resultado-adulterada {
        background: linear-gradient(135deg, rgba(139,42,42,0.15), rgba(139,42,42,0.05));
        border-color: rgba(139,42,42,0.4);
    }
    .resultado-adulterada .res-icon { color: #E05555; }
    .resultado-mezcla {
        background: linear-gradient(135deg, rgba(139,96,32,0.15), rgba(139,96,32,0.05));
        border-color: rgba(139,96,32,0.4);
    }
    .resultado-mezcla .res-icon { color: var(--gold-bright); }
    .res-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .res-value {
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
    }
    .res-icon {
        font-size: 36px;
        opacity: 0.9;
    }

    /* ── PROBABILITY BARS ──────────────────────────────────────── */
    .prob-container {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 24px;
        height: 100%;
    }
    .prob-title {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .prob-row { margin: 14px 0; }
    .prob-row-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 6px;
    }
    .prob-cls {
        font-size: 14px;
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
        background: var(--surface-3);
        border-radius: 0;
        height: 4px;
        width: 100%;
        overflow: hidden;
    }

    /* ── GEO CARDS ─────────────────────────────────────────────── */
    .geo-card {
        padding: 28px;
        border-radius: 2px;
        border: 1px solid var(--border);
        background: var(--surface-2);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
    }
    .geo-card.active {
        border-color: rgba(192,154,46,0.5);
        background: var(--surface-3);
        box-shadow: 0 0 40px rgba(192,154,46,0.06), inset 0 1px 0 rgba(240,192,64,0.15);
    }
    .geo-card.active::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold-bright), transparent);
    }
    .geo-card.inactive { opacity: 0.35; }
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
        font-size: 48px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 6px;
    }
    .geo-winner-tag {
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        color: var(--gold-mid);
        text-transform: uppercase;
        margin-top: 10px;
    }

    /* ── DIVIDER ───────────────────────────────────────────────── */
    .sample-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 48px 0;
    }

    /* ── UPLOAD SECTION TITLE ──────────────────────────────────── */
    .upload-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* ── FOOTER ────────────────────────────────────────────────── */
    .footer {
        margin-top: 80px;
        padding: 32px 0 16px 0;
        border-top: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .footer-left {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--text-dim);
        letter-spacing: 1px;
    }
    .footer-right {
        font-family: 'Cormorant Garamond', serif;
        font-size: 14px;
        color: var(--text-muted);
    }
    .footer-gold {
        color: var(--gold-dark);
    }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════
T_MIN, T_MAX, N_PTS = -30.0, 190.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#4CAF7D", "#E05555", "#F0C040"]
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
#  GRÁFICA TERMOGRAMA — Estética dark luxury
# ═══════════════════════════════════════════════════════════════════
def graficar_termograma(dsc_curve, nombre, color_linea):
    bg     = "#0D0D0D"
    grid_c = "#222222"
    tick_c = "#6A5A3A"
    label_c= "#9A8A6A"

    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=bg)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # Zonas térmicas — muy sutiles
    ax.axvspan(-30,  30, alpha=0.04, color="#5588AA", zorder=0)
    ax.axvspan( 30, 100, alpha=0.04, color="#C89A2E", zorder=0)
    ax.axvspan(100, 190, alpha=0.04, color="#AA4444", zorder=0)

    # Líneas de zona
    for x_line, lbl in [(-30,"−30"), (30,"30°"), (100,"100°"), (190,"190°")]:
        ax.axvline(x_line, color=grid_c, linewidth=0.5, linestyle=":", zorder=1)

    # Línea cero
    ax.axhline(0, color="#3A3A3A", linewidth=0.8, linestyle="-", zorder=2)

    # Curva principal con glow sutil
    ax.plot(T_GRILLA, dsc_curve, color=color_linea, linewidth=1.8,
            zorder=5, solid_capstyle="round")

    # Fill bajo la curva
    ax.fill_between(T_GRILLA, dsc_curve,
                    where=dsc_curve < 0,
                    alpha=0.12, color=color_linea, zorder=3)
    ax.fill_between(T_GRILLA, dsc_curve,
                    where=dsc_curve >= 0,
                    alpha=0.05, color=color_linea, zorder=3)

    # Etiquetas de zona
    for x_c, lbl, c_lbl in [
        (  0, "FUSIÓN",       "#5599BB"),
        ( 65, "TRANSICIÓN",   "#C89A2E"),
        (145, "CARAMELIZACIÓN","#AA5544"),
    ]:
        ax.text(x_c, ax.get_ylim()[1] if ax.get_ylim()[1]!=0 else 0.1,
                lbl, ha="center", va="top",
                fontsize=7.5, color=c_lbl, alpha=0.55,
                fontfamily="monospace", fontweight="bold",
                transform=ax.get_xaxis_transform(),
                clip_on=True)

    ax.set_xlabel("Temperatura  (°C)", fontsize=10, color=label_c, labelpad=8)
    ax.set_ylabel("Flujo de calor  (mW/mg)", fontsize=10, color=label_c, labelpad=8)

    ax.tick_params(colors=tick_c, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
        spine.set_linewidth(0.6)

    ax.grid(True, color=grid_c, linewidth=0.4, linestyle="-", alpha=0.7)
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
        <div style="padding: 14px 16px; background: rgba(192,154,46,0.04); border-left: 2px solid #C89A2E; margin-bottom: 8px; border-radius: 0 2px 2px 0;">
            <div style="font-family:'DM Mono',monospace; font-size:9px; color:#8B6914; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 01</div>
            <div style="font-size:15px; font-weight:600; color:#F5EDD6; font-family:'Cormorant Garamond',serif;">Autenticidad</div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5A4A2A; margin-top:3px;">SVM Lineal · Acc 98.39%</div>
        </div>
        <div style="padding: 14px 16px; background: rgba(192,154,46,0.04); border-left: 2px solid #8B6914; margin-bottom: 8px; border-radius: 0 2px 2px 0;">
            <div style="font-family:'DM Mono',monospace; font-size:9px; color:#5A4A2A; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">NIVEL 02</div>
            <div style="font-size:15px; font-weight:600; color:#F5EDD6; font-family:'Cormorant Garamond',serif;">Origen Geográfico</div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5A4A2A; margin-top:3px;">SVM+PCA · Acc 82.00%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:3px; color:#5A4A2A; text-transform:uppercase; margin-bottom:12px;">Clases detectables</div>
    """, unsafe_allow_html=True)
    ICONOS_AUTH = ["◆", "◆", "◆"]
    for cls, col, ico in zip(CLASES_AUTH, COLORES_AUTH, ICONOS_AUTH):
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
            <span style="color:{col}; font-size:7px;">{ico}</span>
            <span style="font-size:14px; color:#C8B89A; font-family:'Cormorant Garamond',serif;">{cls}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px; margin-bottom:20px;">
        <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:3px; color:#5A4A2A; text-transform:uppercase; margin-bottom:12px;">Orígenes geográficos</div>
    """, unsafe_allow_html=True)
    for cls, col in zip(CLASES_GEO, COLORES_GEO):
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 7px 0 7px 12px; border-bottom: 1px solid rgba(255,255,255,0.03);">
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
#  HERO — Abeja fotográfica estilo National Geographic
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-bg"></div>
    <div class="hero-overlay"></div>
    <div class="hero-grain"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">Calorimetría diferencial de barrido · Machine Learning</div>
        <h1 class="hero-title">
            Honey<span class="hero-title-gold">Check</span>
        </h1>
        <div class="hero-divider"></div>
        <p class="hero-subtitle">
            Detección de adulteración y trazabilidad geográfica de mieles colombianas mediante análisis DSC y modelos de clasificación supervisada.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Sistema Jerárquico V2.0</span>
            <span class="hero-badge">Universidad del Quindío</span>
            <span class="hero-badge">NETZSCH DSC 214 Polyma</span>
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
        <div class="stat-value">98.4%</div>
        <div class="stat-label">Precisión · Autenticidad</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value">82.0%</div>
        <div class="stat-label">Precisión · Origen</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value">62</div>
        <div class="stat-label">Muestras de entrenamiento</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value">p&lt;0.001</div>
        <div class="stat-label">Confianza estadística</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  CARGAR MODELOS
# ═══════════════════════════════════════════════════════════════════
m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()
if m_auth is not None:
    st.success("Sistemas de clasificación calibrados — Entorno listo para análisis.")
else:
    st.warning("Modelos no encontrados en el directorio. La interfaz opera en modo de demostración visual.")

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
    <div style="padding: 48px; text-align:center; border: 1px dashed rgba(192,154,46,0.2); border-radius:2px; 
         background: rgba(192,154,46,0.02); margin-top: 16px;">
        <div style="font-size:40px; margin-bottom:16px; opacity:0.4;">⬡</div>
        <div style="font-family:'Playfair Display',serif; font-size:20px; color:#6A5A3A; margin-bottom:8px;">
            Sistema en espera
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2px; color:#3D2E0A; text-transform:uppercase;">
            Seleccione o arrastre archivos .txt para iniciar el procesamiento
        </div>
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
        <div style="margin-left:auto; font-family:'DM Mono',monospace; font-size:10px; color:#3D2E0A; letter-spacing:2px;">
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
            st.error(f"El archivo no contiene el rango térmico completo requerido (−30 a 190 °C).")
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

        css_map   = ["resultado-real","resultado-adulterada","resultado-mezcla"]
        icon_map  = ["✦", "⚠", "◉"]
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
                        <div style="background:{col}; width:{prob*100:.1f}%; height:100%;"></div>
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
                        {"<div class='geo-winner-tag'>◆ Origen determinado</div>" if activo else ""}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar el espectro térmico: {e}")
    finally:
        os.unlink(tmp_path)

    st.markdown("</div>", unsafe_allow_html=True)  # cierra sample-body

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
