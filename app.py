
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, tempfile
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

st.set_page_config(
    page_title="HoneyCheck — Autenticidad y Origen de Miel DSC",
    page_icon="🍯",
    layout="wide",
)

st.markdown("""
<style>
    .resultado-real {
        background: linear-gradient(135deg, #1D9E75, #0F6E56);
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
    }
    .resultado-adulterada {
        background: linear-gradient(135deg, #D85A30, #993C1D);
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
    }
    .resultado-mezcla {
        background: linear-gradient(135deg, #7F77DD, #534AB7);
        color: white; padding: 24px 32px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: bold; margin: 12px 0;
    }
    .resultado-cafetero {
        background: linear-gradient(135deg, #2471A3, #1A5276);
        color: white; padding: 20px 28px; border-radius: 16px;
        text-align: center; font-size: 22px; font-weight: bold; margin: 12px 0;
    }
    .resultado-orinoquia {
        background: linear-gradient(135deg, #E67E22, #A04000);
        color: white; padding: 20px 28px; border-radius: 16px;
        text-align: center; font-size: 22px; font-weight: bold; margin: 12px 0;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #1D9E75; }
    .metric-label { font-size: 12px; color: #666; margin-top: 4px; }
    .nivel-badge {
        background: #f0f0f0; border-radius: 8px; padding: 6px 14px;
        font-size: 12px; color: #444; display: inline-block; margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#1D9E75,#0F6E56);padding:28px;
border-radius:16px;color:white;text-align:center;margin-bottom:20px;">
    <h1 style="margin:0;font-size:38px;">🍯 HoneyCheck</h1>
    <p style="margin:8px 0 0;font-size:17px;opacity:0.9;">
        Detección de adulteración y origen geográfico de miel mediante DSC y ML
    </p>
    <p style="margin:4px 0 0;font-size:12px;opacity:0.7;">
        Sistema jerárquico de dos niveles · Universidad del Quindío
    </p>
</div>
""", unsafe_allow_html=True)

# Constantes
T_MIN, T_MAX, N_PTS = -30.0, 190.0, 1000
T_GRILLA = np.linspace(T_MIN, T_MAX, N_PTS)
CLASES_AUTH = ["Miel auténtica", "Jarabe comercial", "Mezcla de azúcares"]
CLASES_GEO  = ["Eje Cafetero", "Orinoquía"]
COLORES_AUTH = ["#1D9E75", "#D85A30", "#7F77DD"]
COLORES_GEO  = ["#2471A3", "#E67E22"]

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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(T_GRILLA, dsc_curve, color=color, linewidth=1.8, label=nombre)
    ax.fill_between(T_GRILLA, dsc_curve, alpha=0.08, color=color)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    for a, b, c, et in [(-30,30,"#3B8BD4","Fusión agua/azúcares"),
                         (30,100,"#EF9F27","Transición térmica"),
                         (100,190,"#E24B4A","Caramelización")]:
        ax.axvspan(a, b, alpha=0.06, color=c, label=et)
    ax.set_xlabel("Temperatura (°C)", fontsize=11)
    ax.set_ylabel("DSC (mW/mg)", fontsize=11)
    ax.set_title(f"Termograma DSC — {nombre}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Panel de control")
    st.markdown("---")
    st.markdown("**Sistema jerárquico de 2 niveles**")
    st.markdown("""
    <div style="margin:8px 0;">
        <span style="color:#1D9E75;font-weight:bold;">● Nivel 1</span>
        Autenticidad de la miel<br>
        <small>SVM Lineal · Accuracy 98.39%</small>
    </div>
    <div style="margin:8px 0;">
        <span style="color:#2471A3;font-weight:bold;">● Nivel 2</span>
        Origen geográfico<br>
        <small>SVM + PCA · Accuracy 82.00%</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Categorías detectables**")
    for clase, color in zip(CLASES_AUTH, COLORES_AUTH):
        st.markdown(
            f"<span style='color:{color};font-weight:bold;'>● {clase}</span>",
            unsafe_allow_html=True)
    st.markdown("**Si es miel auténtica:**")
    for clase, color in zip(CLASES_GEO, COLORES_GEO):
        st.markdown(
            f"<span style='color:{color};font-weight:bold;'>  ↳ {clase}</span>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Universidad del Quindío · Programa de Química")

# Cargar modelos
try:
    m_auth, sc_auth, m_geo, sc_geo, pca_geo = cargar_modelos()
    st.success("✅ Modelos cargados correctamente (Nivel 1 + Nivel 2)")
except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.stop()

# Carga de archivos
st.markdown("## 📂 Cargar termograma(s)")
st.markdown("Sube archivos `.txt` exportados del NETZSCH DSC 214 Polyma.")

archivos = st.file_uploader(
    "Selecciona archivos DSC",
    type=["txt"],
    accept_multiple_files=True,
)

if not archivos:
    st.info("👆 Sube al menos un archivo .txt para comenzar el análisis.")
    st.markdown("---")
    st.markdown("## 📊 Rendimiento del sistema")
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
st.markdown(f"## 🔬 Resultados — {len(archivos)} muestra(s)")

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
            f"<div class='nivel-badge'>🔬 Nivel 1 — Autenticidad</div>",
            unsafe_allow_html=True)

        css   = ["resultado-real","resultado-adulterada","resultado-mezcla"][pred]
        icono = ["✅","⚠️","🔶"][pred]
        st.markdown(
            f"<div class='{css}'>{icono} {CLASES_AUTH[pred]}</div>",
            unsafe_allow_html=True)

        col_g, col_m = st.columns([2, 1])
        with col_g:
            fig = graficar_termograma(dsc_interp, archivo.name,
                                      COLORES_AUTH[pred])
            st.pyplot(fig); plt.close()

        with col_m:
            st.markdown("**Confianza — Nivel 1**")
            for cls, prob, col in zip(CLASES_AUTH, probs, COLORES_AUTH):
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <span style="color:{col};font-weight:bold;">{cls}</span>
                    <div style="background:#eee;border-radius:8px;
                         height:10px;margin-top:3px;">
                        <div style="background:{col};width:{prob*100:.1f}%;
                             height:10px;border-radius:8px;"></div>
                    </div>
                    <span style="font-size:12px;color:#555;">{prob*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Features clave**")
            st.markdown(f"- dsc_max: `{feats['dsc_max']:.4f}` mW/mg")
            st.markdown(f"- dsc_slope_mean: `{feats['dsc_slope_mean']:.6f}`")
            st.markdown(f"- n_picos: `{int(feats['n_picos'])}`")
            st.markdown(f"- onset_temp: `{feats['onset_temp']:.2f}` °C")

        # ── NIVEL 2: Origen geográfico (solo si es miel) ─────
        if pred == 0:
            st.markdown("---")
            st.markdown(
                "<div class='nivel-badge'>🗺️ Nivel 2 — Origen geográfico</div>",
                unsafe_allow_html=True)

            X_norm   = sc_geo.transform(dsc_interp.reshape(1, -1))
            X_pca    = pca_geo.transform(X_norm)
            pred_geo = m_geo.predict(X_pca)[0]
            prob_geo = m_geo.predict_proba(X_pca)[0]

            css_geo   = ["resultado-cafetero","resultado-orinoquia"][pred_geo]
            icono_geo = ["🟦","🟠"][pred_geo]
            st.markdown(
                f"<div class='{css_geo}'>{icono_geo} {CLASES_GEO[pred_geo]}</div>",
                unsafe_allow_html=True)

            st.markdown("**Confianza — Nivel 2**")
            col_g1, col_g2 = st.columns(2)
            for col, cls, prob, color in zip(
                [col_g1, col_g2], CLASES_GEO, prob_geo, COLORES_GEO
            ):
                with col:
                    st.markdown(f"""
                    <div style="text-align:center;padding:16px;background:#f8f8f8;
                         border-radius:12px;border-left:4px solid {color};">
                        <div style="color:{color};font-weight:bold;
                             font-size:15px;">{cls}</div>
                        <div style="font-size:28px;font-weight:bold;
                             color:{color};margin-top:6px;">
                             {prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error procesando {archivo.name}: {e}")
    finally:
        os.unlink(tmp_path)

    st.markdown("---")
