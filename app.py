
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# --- INYECCIÓN DE CSS: ESTILO SAAS PREMIUM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Fondo principal: Blanco/Crema ultimamente suave */
    .stApp {
        background-color: #FCFAFA;
        color: #2B1D14;
        font-family: 'Inter', sans-serif;
    }
    
    /* Eliminar el padding superior excesivo de Streamlit */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* BARRA LATERAL: Tono chocolate oscuro/carbón para alto contraste premium */
    [data-testid="stSidebar"] {
        background-color: #1E1511;
        color: #F8F5F0;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #D4AF37 !important;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Títulos globales */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        color: #1E1511;
    }
    
    /* ZONA DE CARGA DE ARCHIVOS DOMADA */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #D4AF37 !important;
        border-radius: 12px !important;
        padding: 40px !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #B8860B !important;
        background-color: #FFFDF5 !important;
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.15);
    }
    
    /* ALERTAS DE STREAMLIT DOMADAS (Quitar azules y verdes por defecto) */
    [data-testid="stAlert"] {
        background-color: #FFFFFF !important;
        border: 1px solid #EAE1D3 !important;
        border-left: 4px solid #D4AF37 !important;
        border-radius: 8px !important;
        color: #2B1D14 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    /* Tarjetas de métricas superiores */
    .metric-card {
        background: #FFFFFF; 
        border-radius: 12px; 
        padding: 20px;
        border: 1px solid rgba(0,0,0,0.04);
        box-shadow: 0 8px 20px rgba(0,0,0,0.03); 
        text-align: center; 
        margin: 8px 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #D4AF37; font-family: 'Montserrat', sans-serif; }
    .metric-label { font-size: 13px; color: #8C827A; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;}
    
    /* Clases de resultados (Estilo limpio) */
    .resultado-real {
        background: linear-gradient(135deg, #4A7C59, #3A6347);
        color: white; padding: 20px 24px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: 600; margin: 12px 0;
        box-shadow: 0 10px 20px rgba(74, 124, 89, 0.2);
    }
    .resultado-adulterada {
        background: linear-gradient(135deg, #C25953, #9E4641);
        color: white; padding: 20px 24px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: 600; margin: 12px 0;
        box-shadow: 0 10px 20px rgba(194, 89, 83, 0.2);
    }
    .resultado-mezcla {
        background: linear-gradient(135deg, #D4AF37, #B8860B);
        color: white; padding: 20px 24px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: 600; margin: 12px 0;
        box-shadow: 0 10px 20px rgba(212, 175, 55, 0.2);
    }
    .nivel-badge {
        background: #F4EFE6; border-radius: 20px; padding: 6px 16px;
        font-size: 12px; color: #6B5E55; display: inline-block; margin: 8px 0;
        font-weight: 600; border: 1px solid #EAE1D3; text-transform: uppercase; letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER PREMIUM SAAS ---
st.markdown("""
<div style="background: white; padding: 40px 32px; border-radius: 16px; 
     box-shadow: 0 10px 40px rgba(0,0,0,0.04); text-align: center; margin-bottom: 32px;
     border-top: 5px solid #D4AF37;">
    <h1 style="color: #1E1511; font-family: 'Montserrat', sans-serif; font-size: 42px; margin: 0 0 8px 0; font-weight: 700;">
        🍯 HoneyCheck
    </h1>
    <p style="color: #6B5E55; font-size: 18px; font-weight: 500; margin: 0;">
        Detección de adulteración y origen geográfico mediante DSC y ML
    </p>
    <div style="margin-top: 16px; display: inline-block; background: #FAF7F2; padding: 6px 16px; border-radius: 20px; font-size: 12px; color: #8C827A; font-weight: 500;">
        Universidad del Quindío · Sistema Jerárquico V2.0
    </div>
</div>
""", unsafe_allow_html=True)

# Constantes
T_MIN, T_MAX, N_PTS = -30.0, 190.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#4A7C59", "#C25953", "#D4AF37"] 
COLORES_GEO  = ["#8B5A2B", "#CD853F"]

@st.cache_resource
def cargar_modelos():
    base = os.path.dirname(__file__)
    # Se recomienda usar try-except interno para no frenar la UI si faltan archivos en pruebas
    try:
        m_auth   = joblib.load(os.path.join(base, "modelo_svm_optimizado.pkl"))
        sc_auth  = joblib.load(os.path.join(base, "scaler_B_final.pkl"))
        m_geo    = joblib.load(os.path.join(base, "modelo_origen_geografico.pkl"))
        sc_geo   = joblib.load(os.path.join(base, "scaler_geo_A.pkl"))
        pca_geo  = joblib.load(os.path.join(base, "pca_geo.pkl"))
        return m_auth, sc_auth, m_geo, sc_geo, pca_geo
    except:
        return None, None, None, None, None

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

def graficar_termograma(dsc_curve, nombre, color):
    # Configuración estricta de gráfica: Arial, tamaño 12, centrado, SIN título.
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T_GRILLA, dsc_curve, color=color, linewidth=2.0, label=nombre)
    ax.fill_between(T_GRILLA, dsc_curve, alpha=0.05, color=color)
    ax.axhline(0, color="#CCCCCC", linewidth=1.0, linestyle="--")
    
    # Zonas de transición sutiles
    ax.axvspan(-30, 30, alpha=0.08, color="#8CA6B1", label="Fusión")
    ax.axvspan(30, 100, alpha=0.08, color="#D4AF37", label="Transición")
    ax.axvspan(100, 190, alpha=0.08, color="#C25953", label="Caramelización")
        
    ax.set_xlabel("Temperatura (°C)", fontsize=12)
    ax.set_ylabel("DSC (mW/mg)", fontsize=12)
    
    # Título omitido intencionalmente para la exportación manual al documento.
    
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3, color="#EAE1D3", linestyle="-")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8C827A')
    ax.spines['bottom'].set_color('#8C827A')
    
    plt.tight_layout()
    return fig

# --- BARRA LATERAL (MODO OSCURO PREMIUM) ---
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>⚙️ Panel de Control</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="margin:8px 0; padding: 12px 16px; background: rgba(255,255,255,0.03); border-left: 3px solid #D4AF37; border-radius: 0 8px 8px 0;">
            <span style="color:#D4AF37; font-weight:700; font-size: 11px; text-transform:uppercase; letter-spacing:1px;">Nivel 1</span><br>
            <span style="color:#FFF; font-size: 15px; font-weight: 500;">Autenticidad</span><br>
            <span style="color:#8C827A; font-size: 12px;">SVM Lineal · Acc 98.39%</span>
        </div>
        <div style="margin:8px 0; padding: 12px 16px; background: rgba(255,255,255,0.03); border-left: 3px solid #D4AF37; border-radius: 0 8px 8px 0;">
            <span style="color:#D4AF37; font-weight:700; font-size: 11px; text-transform:uppercase; letter-spacing:1px;">Nivel 2</span><br>
            <span style="color:#FFF; font-size: 15px; font-weight: 500;">Origen Geográfico</span><br>
            <span style="color:#8C827A; font-size: 12px;">SVM+PCA · Acc 82.00%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color:#8C827A; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>Clases Detectables</p>", unsafe_allow_html=True)
    for clase, color in zip(CLASES_AUTH, COLORES_AUTH):
        st.markdown(
            f"<div style='margin-bottom:6px;'><span style='color:{color};font-size:14px;'>■</span> <span style='font-size:14px; font-weight:500; color:#E0DCD3;'>{clase}</span></div>",
            unsafe_allow_html=True)
            
    st.markdown("<div style='margin-top:16px;'><span style='color:#8C827A; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>Orígenes (Miel Auténtica)</span></div>", unsafe_allow_html=True)
    for clase, color in zip(CLASES_GEO, COLORES_GEO):
        st.markdown(
            f"<div style='margin-bottom:6px; padding-left:10px;'><span style='color:{color};font-size:14px;'>■</span> <span style='font-size:14px; font-weight:500; color:#E0DCD3;'>{clase}</span></div>",
            unsafe_allow_html=True)

# Cargar modelos
m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()
if m_auth is not None:
    st.success("Sistemas jerárquicos calibrados. Entorno de análisis listo.")
else:
    st.warning("Modelos no encontrados en el directorio. La interfaz operará en modo de demostración visual.")

# Carga de archivos
st.markdown("<h2 style='font-size: 24px; margin-bottom: 16px;'>📁 Carga de Termogramas</h2>", unsafe_allow_html=True)

archivos = st.file_uploader(
    "Archivos .txt generados por NETZSCH DSC 214 Polyma",
    type=["txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if not archivos:
    st.info("El sistema está a la espera de muestras. Seleccione o arrastre archivos .txt para iniciar el procesamiento.")
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip(
        [col1, col2, col3, col4],
        [("98.39%","Precisión Autenticidad"),("82.00%","Precisión Origen"),
         ("62","Muestras Base"),("p<0.001","Confianza Estadística")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# Análisis
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown(f"<h2 style='font-size: 24px; border-bottom: 2px solid #EAE1D3; padding-bottom: 10px;'>🔬 Informe Analítico ({len(archivos)} muestras)</h2>", unsafe_allow_html=True)

for archivo in archivos:
    st.markdown(f"<h3 style='font-size: 18px; margin-top: 30px; color: #6B5E55;'>Muestra: {archivo.name}</h3>", unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(archivo.read())
        tmp_path = tmp.name

    try:
        df_raw     = leer_dsc(tmp_path)
        dsc_interp = interpolar(df_raw)

        if dsc_interp is None:
            st.error(f"El archivo {archivo.name} no contiene el espectro térmico completo requerido (-30 a 190 °C).")
            continue

        if m_auth is None:
            continue # Evitar error si los modelos no están

        # ── NIVEL 1: Autenticidad ────────────────────────────
        feats   = extraer_features(dsc_interp)
        X_feat  = sc_auth.transform(np.array(list(feats.values())).reshape(1, -1))
        pred    = m_auth.predict(X_feat)[0]
        probs   = m_auth.predict_proba(X_feat)[0]

        st.markdown("<div class='nivel-badge'>Fase 1: Evaluación de Autenticidad</div>", unsafe_allow_html=True)

        css   = ["resultado-real","resultado-adulterada","resultado-mezcla"][pred]
        icono = ["✓","!","○"][pred]
        st.markdown(f"<div class='{css}'>{CLASES_AUTH[pred]}</div>", unsafe_allow_html=True)

        col_g, col_m = st.columns([2.5, 1.2])
        with col_g:
            # Gráfica generada con los parámetros solicitados (Arial, 12, sin título)
            fig = graficar_termograma(dsc_interp, archivo.name, COLORES_AUTH[pred])
            st.pyplot(fig, use_container_width=True) 
            plt.close()

        with col_m:
            st.markdown("<div style='padding: 20px; background: #FFFFFF; border: 1px solid #EAE1D3; border-radius: 12px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size: 14px; text-transform: uppercase; color: #8C827A; margin-top:0;'>Distribución Probabilística</h4>", unsafe_allow_html=True)
            for cls, prob, col in zip(CLASES_AUTH, probs, COLORES_AUTH):
                st.markdown(f"""
                <div style="margin: 16px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 13px; font-weight: 600; color: #2B1D14;">{cls}</span>
                        <span style="font-size: 13px; font-weight: 700; color: {col};">{prob*100:.1f}%</span>
                    </div>
                    <div style="background:#F4EFE6; border-radius:4px; height:8px; width: 100%; overflow: hidden;">
                        <div style="background:{col}; width:{prob*100:.1f}%; height:100%; border-radius:4px; transition: width 0.5s ease;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── NIVEL 2: Origen geográfico (solo si es miel) ─────
        if pred == 0:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='nivel-badge'>Fase 2: Trazabilidad Geográfica</div>", unsafe_allow_html=True)

            X_norm   = sc_geo.transform(dsc_interp.reshape(1, -1))
            X_pca    = pca_geo.transform(X_norm)
            pred_geo = m_geo.predict(X_pca)[0]
            prob_geo = m_geo.predict_proba(X_pca)[0]

            col_g1, col_g2 = st.columns(2)
            for col_ui, cls, prob, color in zip([col_g1, col_g2], CLASES_GEO, prob_geo, COLORES_GEO):
                is_winner = prob > 0.5
                bg_color = "#FFFFFF" if is_winner else "#FAF7F2"
                border = f"2px solid {color}" if is_winner else "1px solid #EAE1D3"
                opacity = "1.0" if is_winner else "0.5"
                
                with col_ui:
                    st.markdown(f"""
                    <div style="text-align:center; padding:24px; background:{bg_color}; opacity:{opacity};
                         border-radius:12px; border:{border}; box-shadow: 0 4px 15px rgba(0,0,0,0.02); transition: all 0.3s;">
                        <div style="color:#6B5E55; font-family:'Inter', sans-serif; font-weight:600;
                             font-size:15px; text-transform: uppercase; letter-spacing: 1px;">{cls}</div>
                        <div style="font-size:36px; font-weight:700; font-family:'Montserrat', sans-serif;
                             color:{color}; margin-top:8px;">
                             {prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error procesando el espectro térmico: {e}")
    finally:
        os.unlink(tmp_path)

    st.markdown("<div style='height: 40px; border-bottom: 1px dashed #EAE1D3; margin-bottom: 40px;'></div>", unsafe_allow_html=True)
