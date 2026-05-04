
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
)

# --- INYECCIÓN DE CSS PARA EL TEMA "WAXY, GOLDEN & PURE" ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Open+Sans:wght@400;600&display=swap');

    /* Fondo principal crema suave (cera de abejas) */
    .stApp {
        background-color: #FBF6EA;
        color: #3E2723;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Barra lateral estilo panal ámbar oscuro */
    [data-testid="stSidebar"] {
        background-color: #C08A3E;
        color: #FBF6EA;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Títulos y tipografía global */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        color: #3E2723;
    }
    
    /* Zona de carga de archivos (Dropzone) estilo orgánico */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFDF7;
        border: 2px dashed #D4AF37;
        border-radius: 20px;
        box-shadow: inset 0 0 15px rgba(212, 175, 55, 0.1);
        padding: 30px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #B8860B;
        background-color: #FFF9E6;
    }

    /* Clases de resultados (Rediseñadas con la nueva paleta) */
    .resultado-real {
        background: linear-gradient(135deg, #6B8E23, #556B2F); /* Verde Botánico */
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
        box-shadow: 0 4px 15px rgba(107, 142, 35, 0.3);
        font-family: 'Montserrat', sans-serif;
    }
    .resultado-adulterada {
        background: linear-gradient(135deg, #D96C06, #A64B00); /* Bronce / Naranja quemado */
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
        box-shadow: 0 4px 15px rgba(217, 108, 6, 0.3);
        font-family: 'Montserrat', sans-serif;
    }
    .resultado-mezcla {
        background: linear-gradient(135deg, #C29B0C, #997A00); /* Ámbar oscuro */
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
        box-shadow: 0 4px 15px rgba(194, 155, 12, 0.3);
        font-family: 'Montserrat', sans-serif;
    }
    .resultado-cafetero {
        background: linear-gradient(135deg, #8B5A2B, #6B4226); /* Marrón tierra / Café */
        color: white; padding: 20px 28px; border-radius: 16px;
        text-align: center; font-size: 22px; font-weight: bold; margin: 12px 0;
        font-family: 'Montserrat', sans-serif;
    }
    .resultado-orinoquia {
        background: linear-gradient(135deg, #CD853F, #8B4513); /* Marrón claro / Sabana */
        color: white; padding: 20px 28px; border-radius: 16px;
        text-align: center; font-size: 22px; font-weight: bold; margin: 12px 0;
        font-family: 'Montserrat', sans-serif;
    }
    .metric-card {
        background: #FFFDF7; border-radius: 16px; padding: 16px 20px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 0 4px 10px rgba(0,0,0,0.04); text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #B8860B; font-family: 'Montserrat', sans-serif; }
    .metric-label { font-size: 13px; color: #555; margin-top: 4px; font-weight: 600;}
    .nivel-badge {
        background: #F4E8C1; border-radius: 12px; padding: 8px 16px;
        font-size: 13px; color: #3E2723; display: inline-block; margin: 4px 0;
        font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Estilo del recuadro de éxito */
    .st-emotion-cache-1kyxreq {
        background-color: #EEF4EB;
        border: 2px solid #8FBC8F;
        color: #2F4F4F;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER PREMIUM ---
st.markdown("""
<div style="background:linear-gradient(135deg, #FFD166, #D4AF37); padding:32px;
border-radius:24px; color:#3E2723; text-align:center; margin-bottom:24px;
box-shadow: 0 8px 20px rgba(212, 175, 55, 0.2); border: 2px solid rgba(255,255,255,0.4);">
    <h1 style="margin:0; font-size:46px; font-family: 'Montserrat', sans-serif; font-weight: 700;">🍯 HoneyCheck</h1>
    <p style="margin:12px 0 0; font-size:18px; font-weight: 600; color: #4A3628;">
        Detección de adulteración y origen geográfico de miel mediante DSC y ML
    </p>
    <p style="margin:6px 0 0; font-size:14px; opacity:0.8;">
        Sistema jerárquico de dos niveles · Universidad del Quindío
    </p>
</div>
""", unsafe_allow_html=True)

# Constantes
T_MIN, T_MAX, N_PTS = -30.0, 190.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#6B8E23", "#D96C06", "#C29B0C"] # Verde botánico, Bronce, Ámbar
COLORES_GEO  = ["#8B5A2B", "#CD853F"]

@st.cache_resource
def cargar_modelos():
    base = os.path.dirname(__file__)
    m_auth   = joblib.load(os.path.join(base, "modelo_svm_optimizado.pkl"))
    sc_auth  = joblib.load(os.path.join(base, "scaler_B_final.pkl"))
    m_geo    = joblib.load(os.path.join(base, "modelo_origen_geografico.pkl"))
    sc_geo   = joblib.load(os.path.join(base, "scaler_geo_A.pkl"))
    pca_geo  = joblib.load(os.path.join(base, "pca_geo.pkl"))
    return m_auth, sc_auth, m_geo, sc_geo, pca_geo

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
    # Configuración de fuente y legibilidad
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T_GRILLA, dsc_curve, color=color, linewidth=2.0, label=nombre)
    ax.fill_between(T_GRILLA, dsc_curve, alpha=0.1, color=color)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    
    # Zonas coloreadas con tonos cálidos
    for a, b, c, et in [(-30,30,"#A8D0E6","Fusión agua/azúcares"),
                         (30,100,"#F8E9A1","Transición térmica"),
                         (100,190,"#F76C6C","Caramelización")]:
        ax.axvspan(a, b, alpha=0.15, color=c, label=et)
        
    ax.set_xlabel("Temperatura (°C)", fontsize=12, fontweight='bold', color="#3E2723")
    ax.set_ylabel("DSC (mW/mg)", fontsize=12, fontweight='bold', color="#3E2723")
    
    # Título eliminado para exportación limpia
    
    ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor='#FBF6EA', edgecolor='#D4AF37')
    ax.grid(True, alpha=0.3, color="#D4AF37", linestyle=":")
    
    plt.tight_layout()
    return fig

# --- BARRA LATERAL TEMÁTICA ---
with st.sidebar:
    st.markdown("### 🏵️ Panel de control")
    st.markdown("---")
    st.markdown("**Sistema jerárquico**")
    st.markdown("""
    <div style="margin:8px 0; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
        <span style="color:#FFF;font-weight:bold;">💠 Nivel 1</span><br>
        Autenticidad de la miel<br>
        <small style="color:#EEE;">SVM Lineal · Accuracy 98.39%</small>
    </div>
    <div style="margin:8px 0; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
        <span style="color:#FFF;font-weight:bold;">💠 Nivel 2</span><br>
        Origen geográfico<br>
        <small style="color:#EEE;">SVM + PCA · Accuracy 82.00%</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Sellos de Categoría**")
    for clase, color in zip(CLASES_AUTH, COLORES_AUTH):
        st.markdown(
            f"<div><span style='color:{color};font-size:18px;'>●</span> <span style='font-weight:600;'>{clase}</span></div>",
            unsafe_allow_html=True)
    st.markdown("<br>**Mieles Auténticas:**", unsafe_allow_html=True)
    for clase, color in zip(CLASES_GEO, COLORES_GEO):
        st.markdown(
            f"<div style='padding-left:15px;'><span style='color:{color};font-size:18px;'>●</span> <span style='font-weight:600;'>{clase}</span></div>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:12px; opacity:0.8;'>Universidad del Quindío<br>Programa de Química</div>", unsafe_allow_html=True)

# Cargar modelos
try:
    m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()
    st.success("🏵️ Modelos cargados correctamente. Entorno listo.")
except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.stop()

# Carga de archivos
st.markdown("<h2 style='color:#B8860B;'>🗂️ Cargar termograma(s)</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;'>Sube archivos <code>.txt</code> exportados del NETZSCH DSC 214 Polyma.</p>", unsafe_allow_html=True)

archivos = st.file_uploader(
    "Selecciona archivos DSC",
    type=["txt"],
    accept_multiple_files=True,
)

if not archivos:
    st.info("🐝 Sube al menos un archivo .txt para comenzar el análisis en el laboratorio.")
    st.markdown("---")
    st.markdown("<h2 style='color:#B8860B;'>📊 Rendimiento del sistema</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip(
        [col1, col2, col3, col4],
        [("98.39%","Autenticidad"),("82.00%","Origen geográfico"),
         ("62","Muestras entrenadas"),("p<0.001","Significancia")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# Análisis
st.markdown("---")
st.markdown(f"<h2 style='color:#B8860B;'>🔬 Resultados de Pureza — {len(archivos)} muestra(s)</h2>", unsafe_allow_html=True)

for archivo in archivos:
    st.markdown(f"### 📄 {archivo.name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(archivo.read())
        tmp_path = tmp.name

    try:
        df_raw     = leer_dsc(tmp_path)
        dsc_interp = interpolar(df_raw)

        if dsc_interp is None:
            st.error("⚠️ El archivo no cubre el rango requerido (-30 a 190 °C).")
            continue

        # ── NIVEL 1: Autenticidad ────────────────────────────
        feats   = extraer_features(dsc_interp)
        X_feat  = sc_auth.transform(
            np.array(list(feats.values())).reshape(1, -1))
        pred    = m_auth.predict(X_feat)[0]
        probs   = m_auth.predict_proba(X_feat)[0]

        st.markdown(
            f"<div class='nivel-badge'>🧪 Nivel 1 — Evaluación de Autenticidad</div>",
            unsafe_allow_html=True)

        css   = ["resultado-real","resultado-adulterada","resultado-mezcla"][pred]
        icono = ["🌿","⚠️","🔶"][pred]
        st.markdown(
            f"<div class='{css}'>{icono} {CLASES_AUTH[pred]}</div>",
            unsafe_allow_html=True)

        col_g, col_m = st.columns([2.5, 1])
        with col_g:
            # La gráfica ahora usa Arial 12, está centrada visualmente por las columnas de Streamlit y no tiene título.
            fig = graficar_termograma(dsc_interp, archivo.name, COLORES_AUTH[pred])
            st.pyplot(fig, use_container_width=True) 
            plt.close()

        with col_m:
            st.markdown("**Índice de Confianza**")
            for cls, prob, col in zip(CLASES_AUTH, probs, COLORES_AUTH):
                st.markdown(f"""
                <div style="margin:8px 0; background: #FFFDF7; padding: 10px; border-radius: 12px; border: 1px solid #E8D5B5;">
                    <span style="color:{col};font-weight:bold; font-family:'Montserrat', sans-serif;">{cls}</span>
                    <div style="background:#E8D5B5;border-radius:8px;
                         height:12px;margin-top:6px; overflow: hidden;">
                        <div style="background:{col};width:{prob*100:.1f}%;
                             height:12px;border-radius:8px;"></div>
                    </div>
                    <div style="font-size:14px;color:#3E2723; text-align:right; font-weight:bold; margin-top:4px;">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # ── NIVEL 2: Origen geográfico (solo si es miel) ─────
        if pred == 0:
            st.markdown("---")
            st.markdown(
                "<div class='nivel-badge'>🗺️ Nivel 2 — Trazabilidad de Origen</div>",
                unsafe_allow_html=True)

            X_norm   = sc_geo.transform(dsc_interp.reshape(1, -1))
            X_pca    = pca_geo.transform(X_norm)
            pred_geo = m_geo.predict(X_pca)[0]
            prob_geo = m_geo.predict_proba(X_pca)[0]

            css_geo   = ["resultado-cafetero","resultado-orinoquia"][pred_geo]
            icono_geo = ["⛰️","🌅"][pred_geo]
            st.markdown(
                f"<div class='{css_geo}'>{icono_geo} {CLASES_GEO[pred_geo]}</div>",
                unsafe_allow_html=True)

            st.markdown("**Confianza Geográfica**")
            col_g1, col_g2 = st.columns(2)
            for col, cls, prob, color in zip(
                [col_g1, col_g2], CLASES_GEO, prob_geo, COLORES_GEO
            ):
                with col:
                    st.markdown(f"""
                    <div style="text-align:center;padding:20px;background:#FFFDF7;
                         border-radius:16px;border-bottom:5px solid {color}; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                        <div style="color:{color};font-family:'Montserrat', sans-serif; font-weight:700;
                             font-size:18px;">{cls}</div>
                        <div style="font-size:32px;font-weight:bold;
                             color:{color};margin-top:8px;">
                             {prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error procesando {archivo.name}: {e}")
    finally:
        os.unlink(tmp_path)

    st.markdown("---")
