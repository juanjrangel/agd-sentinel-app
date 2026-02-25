import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES ---
st.set_page_config(
    page_title="AGD-Sentinel | Ingeotecnia",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS "Nuclear"
st.markdown("""
    <style>
    /* 1. Fondo oscuro global y texto base */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* CORRECCIÓN DEL ENCABEZADO */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    /* 2. Forzar texto blanco */
    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: #FAFAFA !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 3. INPUTS NUMÉRICOS MEJORADOS */
    input[type="number"] {
        color: #FFFFFF !important;
        background-color: #262730 !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #464B5C !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    input {
        caret-color: #FF4B4B !important;
    }
    input::-webkit-outer-spin-button,
    input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    /* 4. CORRECCIÓN DE TODOS LOS TOOLTIPS NATIVOS */
    div[data-testid="stTooltipContent"] {
        background-color: #262730 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #464B5C !important;
    }

    /* 5. Estilo de Tarjetas de Métricas */
    [data-testid="stMetricLabel"] {
        color: #A3A8B8 !important; 
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; 
        font-size: 1.8rem !important;
    }

    /* 6. SEMÁFORO LED PERSONALIZADO */
    .led-container {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        position: relative;
        margin-bottom: 10px;
    }
    .led-label {
        color: #A3A8B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .led-value {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    .led-dot {
        width: 15px;
        height: 15px;
        border-radius: 50%;
        display: inline-block;
        cursor: help;
        position: relative;
    }
    .led-dot .tooltip-box {
        visibility: hidden;
        width: 200px;
        background-color: #FFFFFF;
        color: #000000 !important;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        font-size: 0.8rem;
        font-weight: normal;
        border: 2px solid #ccc;
    }
    .led-dot .tooltip-box strong {
        color: #000000 !important;
    }
    .led-dot:hover .tooltip-box {
        visibility: visible;
        opacity: 1;
    }

    /* 7. Sliders y Widgets */
    .stSlider label {
        color: #FAFAFA !important;
        font-weight: 500 !important;
    }
    
    /* 8. Barra Lateral */
    section[data-testid="stSidebar"] {
        background-color: #262730 !important;
    }
    
    /* 9. Botón de Machine Learning */
    div.stButton > button:first-child {
        background-color: #FF4B4B; 
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 0.75em 1.5em;
        border-radius: 10px;
        border: none;
        width: 100%;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        box-shadow: 0px 6px 20px rgba(255, 75, 75, 0.6);
        transform: translateY(-2px);
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNCIONES AUXILIARES ---

def get_metric_status(metric_name, value):
    """Retorna color y mensaje para el semáforo de ML"""
    value = round(value, 3) 
    if metric_name == "R2":
        if value >= 0.90:
            return "#00CC96", "EXCELENTE", "El modelo entiende perfectamente la historia del terreno."
        elif value >= 0.75:
            return "#FFA500", "ACEPTABLE", "Ajuste razonable, pero hay datos dispersos."
        else:
            return "#FF4B4B", "BAJO", "El dato ingresado contradice la tendencia histórica."
    elif metric_name == "RMSE":
        if value <= 0.05:
            return "#00CC96", "EXCELENTE", "Margen de error mínimo (< 0.05)."
        elif value <= 0.10:
            return "#FFA500", "ACEPTABLE", "Margen de error moderado."
        else:
            return "#FF4B4B", "ALTO", "Alta incertidumbre en la predicción."
    elif metric_name == "MAE":
        if value <= 0.04:
            return "#00CC96", "EXCELENTE", "Desviación promedio insignificante."
        elif value <= 0.08:
            return "#FFA500", "ACEPTABLE", "Desviación promedio normal."
        else:
            return "#FF4B4B", "ALTO", "El modelo no logra conectar bien los puntos."
    return "#FFFFFF", "N/A", "Sin datos"

def render_led_metric(label, value, metric_type):
    """Genera el HTML del Semáforo LED"""
    color, status, desc = get_metric_status(metric_type, value)
    led_style = f"background-color: {color}; box-shadow: 0 0 10px {color};"
    
    html = f"""
    <div class="led-container">
        <div class="led-label">{label}</div>
        <div class="led-value">
            {value:.3f}
            <div class="led-dot" style="{led_style}">
                <div class="tooltip-box">
                    <strong>DIAGNÓSTICO: {status}</strong><br>
                    <hr style="margin: 5px 0; border-color: #ccc;">
                    {desc}
                </div>
            </div>
        </div>
    </div>
    """
    return html

def calcular_parametros_estimados(spt_n, vs_velocity, resistivity):
    """Simula correlación datos de campo -> parámetros (MODO AUTOMÁTICO)"""
    phi = 18 + (spt_n * 0.4) 
    phi = min(max(phi, 15), 35) 
    
    cohesion = (vs_velocity * 0.08) - 2
    cohesion = min(max(cohesion, 5), 40)
    
    gamma = 16 + (spt_n * 0.1)
    gamma = min(max(gamma, 16), 21)
    
    if resistivity < 100:
        factor = 0.85 + (resistivity / 1000)
        cohesion *= factor
        phi *= factor
    
    return round(phi, 2), round(cohesion, 2), round(gamma, 2)

def factor_seguridad_talud_infinito(c, phi, gamma, alpha_deg, hw_ratio, k_sismo):
    """Ecuación Talud Infinito"""
    alpha = np.radians(alpha_deg)
    phi_rad = np.radians(phi)
    gamma_w = 9.81  
    z = 5.0 
    hw = z * hw_ratio 
    
    resistencia = c + (gamma * z - gamma_w * hw) * (np.cos(alpha)**2) * np.tan(phi_rad)
    actuante = (gamma * z * np.sin(alpha) * np.cos(alpha)) + (k_sismo * gamma * z * np.cos(alpha))
    
    if actuante <= 0.1: return 10.0 
    return resistencia / actuante

# --- 3. INTERFAZ (SIDEBAR CON INPUTS SINCRONIZADOS ROBUSTOS) ---

def dual_input(label, min_val, max_val, default, key_base, step=1.0, help_text=None):
    """Crea un slider y un input numérico que se sincronizan bidireccionalmente."""
    slider_key = f"{key_base}_slider"
    input_key = f"{key_base}_input"
    
    if key_base not in st.session_state:
        st.session_state[key_base] = default
        if slider_key not in st.session_state: st.session_state[slider_key] = default
        if input_key not in st.session_state: st.session_state[input_key] = default

    def update_from_slider():
        val = st.session_state[slider_key]
        st.session_state[key_base] = val
        st.session_state[input_key] = val

    def update_from_input():
        val = st.session_state[input_key]
        st.session_state[key_base] = val
        st.session_state[slider_key] = val

    st.markdown(f"<label style='font-size: 1rem; font-weight: 500;'>{label}</label>", unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)
        
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.slider(
            label="hidden",
            min_value=float(min_val), max_value=float(max_val), step=float(step),
            key=slider_key,
            value=float(st.session_state[key_base]), 
            on_change=update_from_slider,
            label_visibility="collapsed"
        )
    with col2:
        st.number_input(
            label="hidden",
            min_value=float(min_val), max_value=float(max_val), step=float(step),
            key=input_key,
            value=float(st.session_state[key_base]),
            on_change=update_from_input,
            label_visibility="collapsed"
        )
    return st.session_state[key_base]

with st.sidebar:
    st.title("🎛️ Panel de Campo")
    st.markdown("---")
    
    # ---------------- SECCIÓN 1: SENSORES ----------------
    # Tooltip integrado en el título
    st.markdown("### 1. 📡 Datos de Sensores", help="Parámetros medidos en campo (Input para Estimación).")
    
    input_spt = dual_input(
        "SPT (Golpes N)", 2, 50, 15, "spt", step=1.0,
        help_text="< 10: Suelo suelto | > 30: Suelo denso"
    )
    input_vs = dual_input(
        "Velocidad Onda Vs (m/s)", 100, 500, 218, "vs", step=1.0,
        help_text="Rigidez del suelo. Crítico < 200 m/s"
    )
    input_resistivity = dual_input(
        "Resistividad (Ohm-m)", 10, 500, 200, "res", step=10.0,
        help_text="Bajos valores = Alta saturación"
    )
    
    # Cálculo Automático (Ocurre de fondo para tener los valores listos)
    est_phi, est_c, est_gamma = calcular_parametros_estimados(input_spt, input_vs, input_resistivity)

    # ---------------- SECCIÓN 2: CONDICIONANTES ----------------
    st.markdown("---")
    st.markdown("### 2. ⛈️ Condicionantes")
    
    input_alpha = dual_input(
        "Pendiente (°)", 5.0, 90.0, 25.0, "alpha", step=1.0
    )
    input_rain_pct = dual_input(
        "Saturación (%)", 0.0, 100.0, 30.0, "rain", step=5.0
    )
    input_seismic = st.number_input(
        "Sismo (Coef. k)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.00, 
        step=0.01,
        format="%.2f",
        help="Aceleración sísmica horizontal."
    )

    # ---------------- SECCIÓN 3: LABORATORIO ----------------
    st.markdown("---")
    st.markdown("### 3. 🧪 Datos de Laboratorio")
    
    # Interruptor Principal
    manual_mode = st.toggle("¿Usar datos de Laboratorio?", value=False, 
                            help="Active esto si tiene valores exactos de Cohesión, Fricción y Peso Unitario. El sistema ignorará los sensores y usará estos datos.")

    if manual_mode:
        st.success("Modo Manual ACTIVADO: Ingrese valores abajo.")
        final_c = st.number_input("Cohesión (c) [kPa]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f")
        final_phi = st.number_input("Ángulo de Fricción (ϕ) [°]", min_value=0.0, max_value=45.0, value=25.0, step=0.1, format="%.1f")
        final_gamma = st.number_input("Peso Unitario (γ) [kN/m³]", min_value=10.0, max_value=25.0, value=18.0, step=0.1, format="%.1f")
        
        source_label = "LABORATORIO (REAL)"
        source_color = "#00CC96" 
    else:
        st.caption("Modo Manual DESACTIVADO: Usando estimación por sensores.")
        # Asignamos los valores estimados a las variables finales
        final_c = est_c
        final_phi = est_phi
        final_gamma = est_gamma
        
        source_label = "ESTIMACIÓN (CORRELACIÓN)"
        source_color = "#FFA500" 

# --- 4. CÁLCULOS CENTRALES (Usando las variables finales) ---

input_hw_ratio = input_rain_pct / 100.0

# Calculamos FS usando las variables finales (sean estimadas o manuales)
fs_actual = factor_seguridad_talud_infinito(final_c, final_phi, final_gamma, input_alpha, input_hw_ratio, input_seismic)

# Monte Carlo (Adaptado a la fuente)
n_sim = 1000
np.random.seed(42)
fs_sims = []
for _ in range(n_sim):
    c_s = max(final_c + np.random.normal(0, 2), 0.1)
    p_s = max(final_phi + np.random.normal(0, 1.5), 5)
    fs_sims.append(factor_seguridad_talud_infinito(c_s, p_s, final_gamma, input_alpha, input_hw_ratio, input_seismic))
prob_falla = np.mean(np.array(fs_sims) < 1.1) * 100

# --- 5. VISUALIZACIÓN ---

st.title("Sistema de Predicción de Deterioro Geotécnico (AGD)")
st.markdown(f"Monitorización en tiempo real sector **Cortinas, Toledo**. Escala 1:5000.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Estado Actual")
    if fs_actual < 1.1:
        color, icon, txt = "#FF4B4B", "🚨", "CRÍTICO"
    elif fs_actual < 1.5:
        color, icon, txt = "#FFA500", "⚠️", "ALERTA"
    else:
        color, icon, txt = "#00CC96", "✅", "ESTABLE"
        
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
        <h1 style="margin:0; font-size: 3.5rem;">{fs_actual:.2f}</h1>
        <p style="margin:0; font-size: 1.2rem; font-weight: bold;">{icon} {txt}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VISUALIZACIÓN DE PARÁMETROS CON FUENTE
    st.markdown(f"### Parámetros del Suelo")
    st.markdown(f"<small style='color: {source_color}; font-weight: bold;'>FUENTE: {source_label}</small>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Cohesión (c)", f"{final_c} kPa")
    c1.metric("Peso Unit. (γ)", f"{final_gamma} kN/m³")
    c2.metric("Fricción (ϕ)", f"{final_phi} °")
    c2.metric("Prob. Falla", f"{prob_falla:.1f}%")

with col2:
    st.subheader("Distribución de Probabilidad")
    fig = px.histogram(x=fs_sims, nbins=40, color_discrete_sequence=['#636EFA'])
    fig.add_vline(x=1.1, line_dash="dash", line_color="red")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title="Factor de Seguridad", yaxis_title="Frecuencia")
    st.plotly_chart(fig, width="stretch")

# --- 6. MACHINE LEARNING ---

st.markdown("---")
st.subheader("🤖 Proyección de Deterioro y Recuperación Futuro (IA)")

btn_col, graph_col = st.columns([1, 3])

with btn_col:
    st.write("")
    if st.button("EJECUTAR MODELO\nPREDICTIVO AGD"):
        # Lógica ML
        years_hist = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1,1)
        fs_hist = np.array([1.25, 1.20, 1.10, 1.00, 0.90, 1.05, 1.15])
        X_train = np.vstack([years_hist, [[2026]]])
        y_train = np.append(fs_hist, fs_actual)
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_poly, y_train)
        y_pred = model.predict(X_poly)
        
        # Métricas
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        st.markdown("##### Calidad del Ajuste")
        st.markdown(render_led_metric("R² (Precisión)", r2, "R2"), unsafe_allow_html=True)
        st.markdown(render_led_metric("RMSE (Error)", rmse, "RMSE"), unsafe_allow_html=True)
        st.markdown(render_led_metric("MAE (Absoluto)", mae, "MAE"), unsafe_allow_html=True)

        with graph_col:
            years_fut = np.arange(2017, 2031).reshape(-1,1)
            fs_fut = model.predict(poly.transform(years_fut))
            
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=years_hist.flatten(), y=fs_hist, mode='markers', name='Histórico'))
            fig_ml.add_trace(go.Scatter(x=[2026], y=[fs_actual], mode='markers', marker_symbol='star', marker_size=15, name='Actual'))
            fig_ml.add_trace(go.Scatter(x=years_fut.flatten(), y=fs_fut, mode='lines', name='Tendencia IA'))
            fig_ml.add_hline(y=1.1, line_color="red", line_dash="dash")
            fig_ml.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                xaxis_title="Año", yaxis_title="Factor de Seguridad", height=450)
            st.plotly_chart(fig_ml, width="stretch")
            
            # --- CORRECCIÓN LÓGICA DE LA ALERTA ---
            if fs_actual < 1.1:
                # Caso 1: Falla Inmediata
                st.error(f"⚠️ PREDICCIÓN: Inestabilidad DETECTADA ACTUALMENTE (2026) y persistente en el futuro.")
            else:
                # Caso 2: Buscar falla futura
                future_risk = years_fut.flatten()[np.where((years_fut.flatten() > 2026) & (fs_fut < 1.1))]
                if len(future_risk) > 0:
                    st.error(f"⚠️ PREDICCIÓN: Posible inestabilidad detectada a partir del año **{future_risk[0]}**.")
                else:
                    st.success("✅ PREDICCIÓN: Tendencia estable estimada para los próximos 5 años.")
    else:
        with graph_col:

            st.info("👈 Ejecute el modelo para ver la proyección.")
