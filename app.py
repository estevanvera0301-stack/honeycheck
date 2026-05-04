
# ═══════════════════════════════════════════════════════════════════
#  HERO — Panal SVG claro
# ═══════════════════════════════════════════════════════════════════
# Hero con abeja SVG — style inline (Streamlit no filtra style en divs)
st.markdown("""
<div style="position:relative; width:100%; min-height:420px; overflow:hidden; margin:-1rem -1rem 0 -1rem; display:flex; align-items:center;">

    <!-- Fondo gradiente -->
    <div style="position:absolute; inset:0; background-color:#FFFDF4; background-image: radial-gradient(ellipse 65% 80% at 85% 50%, rgba(245,212,122,0.50) 0%, transparent 70%), radial-gradient(ellipse 45% 60% at 10% 30%, rgba(255,245,210,0.70) 0%, transparent 65%), radial-gradient(ellipse 30% 50% at 50% 90%, rgba(232,168,32,0.12) 0%, transparent 60%);"></div>

    <!-- Línea dorada izquierda -->
    <div style="position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg, transparent, #E8A820, #C8820A, transparent); z-index:6;"></div>

    <!-- Abeja SVG ilustrada — derecha -->
    <div style="position:absolute; right:60px; top:50%; transform:translateY(-50%); width:340px; height:340px; opacity:0.20; pointer-events:none; z-index:5;">
        <svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" fill="#C8820A" style="width:100%; height:100%;">
          <!-- abdomen -->
          <ellipse cx="200" cy="245" rx="52" ry="85"/>
          <!-- rayas abdomen -->
          <rect x="150" y="220" width="100" height="16" rx="8" fill="#E8A820" opacity="0.75"/>
          <rect x="150" y="244" width="100" height="16" rx="8" fill="#E8A820" opacity="0.75"/>
          <rect x="154" y="268" width="92"  height="16" rx="8" fill="#E8A820" opacity="0.75"/>
          <rect x="162" y="292" width="76"  height="13" rx="6" fill="#E8A820" opacity="0.60"/>
          <!-- tórax -->
          <ellipse cx="200" cy="160" rx="44" ry="40"/>
          <!-- cabeza -->
          <circle cx="200" cy="98" r="38"/>
          <!-- ojos -->
          <circle cx="182" cy="90" r="8" fill="#FFF8E1" opacity="0.6"/>
          <circle cx="218" cy="90" r="8" fill="#FFF8E1" opacity="0.6"/>
          <!-- antenas -->
          <line x1="186" y1="63" x2="155" y2="22" stroke="#C8820A" stroke-width="7" stroke-linecap="round"/>
          <circle cx="150" cy="17" r="10"/>
          <line x1="214" y1="63" x2="245" y2="22" stroke="#C8820A" stroke-width="7" stroke-linecap="round"/>
          <circle cx="250" cy="17" r="10"/>
          <!-- alas superiores -->
          <ellipse cx="132" cy="148" rx="78" ry="32" transform="rotate(-28 132 148)" fill="#C8820A" opacity="0.35"/>
          <ellipse cx="268" cy="148" rx="78" ry="32" transform="rotate(28 268 148)"  fill="#C8820A" opacity="0.35"/>
          <!-- alas inferiores -->
          <ellipse cx="145" cy="185" rx="50" ry="20" transform="rotate(-18 145 185)" fill="#C8820A" opacity="0.20"/>
          <ellipse cx="255" cy="185" rx="50" ry="20" transform="rotate(18 255 185)"  fill="#C8820A" opacity="0.20"/>
          <!-- patas -->
          <line x1="162" y1="165" x2="110" y2="210" stroke="#C8820A" stroke-width="5" stroke-linecap="round"/>
          <line x1="162" y1="150" x2="105" y2="170" stroke="#C8820A" stroke-width="5" stroke-linecap="round"/>
          <line x1="162" y1="178" x2="108" y2="230" stroke="#C8820A" stroke-width="4" stroke-linecap="round"/>
          <line x1="238" y1="165" x2="290" y2="210" stroke="#C8820A" stroke-width="5" stroke-linecap="round"/>
          <line x1="238" y1="150" x2="295" y2="170" stroke="#C8820A" stroke-width="5" stroke-linecap="round"/>
          <line x1="238" y1="178" x2="292" y2="230" stroke="#C8820A" stroke-width="4" stroke-linecap="round"/>
          <!-- aguijón -->
          <polygon points="200,328 193,348 207,348" opacity="0.8"/>
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
            <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">
                Sistema Jerárquico V2.0</span>
            <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">
                Universidad del Quindío</span>
            <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color:#5C3D00; border:1px solid #E8A820; padding:7px 16px; border-radius:2px; background:rgba(255,255,255,0.65); backdrop-filter:blur(8px); text-transform:uppercase;">
                NETZSCH DSC 214 Polyma</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
