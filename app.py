import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. PAGE CONFIGURATION AND VISUAL STYLES ---
st.set_page_config(
    page_title="AGD-Sentinel | Geotechnics",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Injection for UI styling
st.markdown("""
    <style>
    /* 1. Global dark background and base text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* HEADER CORRECTION */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    /* 2. Force white text */
    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: #FAFAFA !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 3. IMPROVED NUMERIC INPUTS */
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

    /* 4. NATIVE TOOLTIPS CORRECTION */
    div[data-testid="stTooltipContent"] {
        background-color: #262730 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #464B5C !important;
    }

    /* 5. Metric Cards Style */
    [data-testid="stMetricLabel"] {
        color: #A3A8B8 !important; 
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; 
        font-size: 1.8rem !important;
    }

    /* 6. CUSTOM LED INDICATOR FOR ML METRICS */
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

    /* 7. Sliders and Widgets */
    .stSlider label {
        color: #FAFAFA !important;
        font-weight: 500 !important;
    }
    
    /* 8. Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730 !important;
    }
    
    /* 9. Machine Learning Button */
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

# --- 2. AUXILIARY FUNCTIONS ---

def get_metric_status(metric_name, value):
    """Returns color and status message for the ML evaluation LED indicator."""
    value = round(value, 3) 
    if metric_name == "R2":
        if value >= 0.90:
            return "#00CC96", "EXCELLENT", "The model perfectly understands the terrain's history."
        elif value >= 0.75:
            return "#FFA500", "ACCEPTABLE", "Reasonable fit, but data is scattered."
        else:
            return "#FF4B4B", "POOR", "Input data contradicts the historical trend."
    elif metric_name == "RMSE":
        if value <= 0.05:
            return "#00CC96", "EXCELLENT", "Minimal error margin (< 0.05)."
        elif value <= 0.10:
            return "#FFA500", "ACCEPTABLE", "Moderate error margin."
        else:
            return "#FF4B4B", "HIGH", "High uncertainty in prediction."
    elif metric_name == "MAE":
        if value <= 0.04:
            return "#00CC96", "EXCELLENT", "Insignificant average deviation."
        elif value <= 0.08:
            return "#FFA500", "ACCEPTABLE", "Normal average deviation."
        else:
            return "#FF4B4B", "HIGH", "The model fails to connect the data points well."
    return "#FFFFFF", "N/A", "No data"

def render_led_metric(label, value, metric_type):
    """Generates the HTML string for the custom LED metric indicator."""
    color, status, desc = get_metric_status(metric_type, value)
    led_style = f"background-color: {color}; box-shadow: 0 0 10px {color};"
    
    html = f"""
    <div class="led-container">
        <div class="led-label">{label}</div>
        <div class="led-value">
            {value:.3f}
            <div class="led-dot" style="{led_style}">
                <div class="tooltip-box">
                    <strong>DIAGNOSIS: {status}</strong><br>
                    <hr style="margin: 5px 0; border-color: #ccc;">
                    {desc}
                </div>
            </div>
        </div>
    </div>
    """
    return html

def calcular_parametros_estimados(spt_n, vs_velocity, resistivity):
    """Simulates field data correlation to geotechnical parameters (AUTOMATIC MODE)."""
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
    """Calculates the pseudo-static Factor of Safety (FS) using the Infinite Slope Limit Equilibrium Model."""
    alpha = np.radians(alpha_deg)
    phi_rad = np.radians(phi)
    
    gamma_w = 9.81  # Unit weight of water (kN/m³)
    z = 5.0  # Assumed depth of the failure surface (m)
    hw = z * hw_ratio  # Height of the groundwater table above the failure surface
    
    # Resisting forces (Shear strength based on Mohr-Coulomb criterion)
    resistencia = c + (gamma * z - gamma_w * hw) * (np.cos(alpha)**2) * np.tan(phi_rad)
    
    # Driving forces (Gravity and pseudo-static seismic loads)
    actuante = (gamma * z * np.sin(alpha) * np.cos(alpha)) + (k_sismo * gamma * z * np.cos(alpha))
    
    # Prevent division by zero for completely flat terrains
    if actuante <= 0.1: return 10.0 
    
    return resistencia / actuante

# --- 3. USER INTERFACE (SIDEBAR WITH ROBUST SYNCHRONIZED INPUTS) ---

def dual_input(label, min_val, max_val, default, key_base, step=1.0, help_text=None):
    """Creates a slider and numeric input that are bidirectionally synchronized via Streamlit Session State to optimize memory handling."""
    slider_key = f"{key_base}_slider"
    input_key = f"{key_base}_input"
    
    # Initialize Session State variables to prevent recursive callback errors
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
            on_change=update_from_slider,
            label_visibility="collapsed"
        )
    with col2:
        st.number_input(
            label="hidden",
            min_value=float(min_val), max_value=float(max_val), step=float(step),
            key=input_key,
            on_change=update_from_input,
            label_visibility="collapsed"
        )
    return st.session_state[key_base]

with st.sidebar:
    st.title("🎛️ Field Panel")
    st.markdown("---")
    
    # ---------------- SECTION 1: SENSORS ----------------
    st.markdown("### 1. 📡 Sensor Data", help="Field measured parameters (Input for Estimation).")
    
    input_spt = dual_input(
        "SPT (N-blows)", 2, 50, 15, "spt", step=1.0,
        help_text="< 10: Loose soil | > 30: Dense soil"
    )
    input_vs = dual_input(
        "Shear Wave Velocity Vs (m/s)", 100, 500, 218, "vs", step=1.0,
        help_text="Soil stiffness. Critical < 200 m/s"
    )
    input_resistivity = dual_input(
        "Resistivity (Ohm-m)", 10, 500, 200, "res", step=10.0,
        help_text="Low values = High saturation"
    )
    
    # Automatic geotechnical parameter estimation running in the background
    est_phi, est_c, est_gamma = calcular_parametros_estimados(input_spt, input_vs, input_resistivity)

    # ---------------- SECTION 2: BOUNDARY CONDITIONS ----------------
    st.markdown("---")
    st.markdown("### 2. ⛈️ Boundary Conditions")
    
    input_alpha = dual_input(
        "Slope Angle (°)", 5.0, 90.0, 25.0, "alpha", step=1.0
    )
    input_rain_pct = dual_input(
        "Saturation (%)", 0.0, 100.0, 30.0, "rain", step=5.0
    )
    input_seismic = st.number_input(
        "Seismic (k Coeff.)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.00, 
        step=0.01,
        format="%.2f",
        help="Horizontal seismic acceleration."
    )

    # ---------------- SECTION 3: LABORATORY DATA ----------------
    st.markdown("---")
    st.markdown("### 3. 🧪 Laboratory Data")
    
    # Main toggle switch for manual physical properties override
    manual_mode = st.toggle("Use Laboratory Data?", value=False, 
                            help="Activate this if you have exact values for Cohesion, Friction, and Unit Weight. The system will ignore sensors and use these data.")

    if manual_mode:
        st.success("Manual Mode ACTIVATED: Enter values below.")
        final_c = st.number_input("Cohesion (c) [kPa]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f")
        final_phi = st.number_input("Friction Angle (ϕ) [°]", min_value=0.0, max_value=45.0, value=25.0, step=0.1, format="%.1f")
        final_gamma = st.number_input("Unit Weight (γ) [kN/m³]", min_value=10.0, max_value=25.0, value=18.0, step=0.1, format="%.1f")
        
        source_label = "LABORATORY (REAL)"
        source_color = "#00CC96" 
    else:
        st.caption("Manual Mode DEACTIVATED: Using sensor estimation.")
        # Assign estimated parameters to the final variables for physical modeling
        final_c = est_c
        final_phi = est_phi
        final_gamma = est_gamma
        
        source_label = "ESTIMATION (CORRELATION)"
        source_color = "#FFA500" 

# --- 4. CORE COMPUTATIONAL ENGINE ---

input_hw_ratio = input_rain_pct / 100.0

# Compute the current Factor of Safety using deterministic Limit Equilibrium calculations
fs_actual = factor_seguridad_talud_infinito(final_c, final_phi, final_gamma, input_alpha, input_hw_ratio, input_seismic)

# Monte Carlo Simulation: Introducing Gaussian noise to physical parameters to assess failure probability
n_sim = 1000
np.random.seed(42)
fs_sims = []
for _ in range(n_sim):
    c_s = max(final_c + np.random.normal(0, 2), 0.1)
    p_s = max(final_phi + np.random.normal(0, 1.5), 5)
    fs_sims.append(factor_seguridad_talud_infinito(c_s, p_s, final_gamma, input_alpha, input_hw_ratio, input_seismic))
prob_falla = np.mean(np.array(fs_sims) < 1.1) * 100

# --- 5. REAL-TIME DASHBOARD VISUALIZATION ---

st.title("Geotechnical Degradation Predictive System (AGD)")
st.markdown(f"Real-time monitoring sector **Cortinas, Toledo**. Scale 1:5000.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current State")
    if fs_actual < 1.1:
        color, icon, txt = "#FF4B4B", "🚨", "CRITICAL"
    elif fs_actual < 1.5:
        color, icon, txt = "#FFA500", "⚠️", "WARNING"
    else:
        color, icon, txt = "#00CC96", "✅", "STABLE"
        
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
        <h1 style="margin:0; font-size: 3.5rem;">{fs_actual:.2f}</h1>
        <p style="margin:0; font-size: 1.2rem; font-weight: bold;">{icon} {txt}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render physical parameters emphasizing the active data source
    st.markdown(f"### Soil Parameters")
    st.markdown(f"<small style='color: {source_color}; font-weight: bold;'>SOURCE: {source_label}</small>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Cohesion (c)", f"{final_c} kPa")
    c1.metric("Unit Weight (γ)", f"{final_gamma} kN/m³")
    c2.metric("Friction Angle (ϕ)", f"{final_phi} °")
    c2.metric("Failure Prob.", f"{prob_falla:.1f}%")

with col2:
    st.subheader("Probability Distribution")
    fig = px.histogram(x=fs_sims, nbins=40, color_discrete_sequence=['#636EFA'])
    fig.add_vline(x=1.1, line_dash="dash", line_color="red")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title="Factor of Safety", yaxis_title="Frequency")
    st.plotly_chart(fig, width="stretch")

# --- 6. MACHINE LEARNING PREDICTIVE ENGINE ---

st.markdown("---")
st.subheader("🤖 Future Degradation and Recovery Projection (AI)")

btn_col, graph_col = st.columns([1, 3])

with btn_col:
    st.write("")
    if st.button("EXECUTE AGD\nPREDICTIVE MODEL"):
        
        # Execute Polynomial Regression Pipeline
        # Historical temporal data (Features)
        years_hist = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1,1)
        # Historical Factor of Safety data (Target)
        fs_hist = np.array([1.25, 1.20, 1.10, 1.00, 0.90, 1.05, 1.15])
        
        # Append current physical state to training matrix
        X_train = np.vstack([years_hist, [[2026]]])
        y_train = np.append(fs_hist, fs_actual)
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_poly, y_train)
        y_pred = model.predict(X_poly)
        
        # Compute standard statistical evaluation metrics
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        st.markdown("##### Fit Quality")
        st.markdown(render_led_metric("R² (Accuracy)", r2, "R2"), unsafe_allow_html=True)
        st.markdown(render_led_metric("RMSE (Error)", rmse, "RMSE"), unsafe_allow_html=True)
        st.markdown(render_led_metric("MAE (Absolute)", mae, "MAE"), unsafe_allow_html=True)

        with graph_col:
            years_fut = np.arange(2017, 2031).reshape(-1,1)
            fs_fut = model.predict(poly.transform(years_fut))
            
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=years_hist.flatten(), y=fs_hist, mode='markers', name='Historical'))
            fig_ml.add_trace(go.Scatter(x=[2026], y=[fs_actual], mode='markers', marker_symbol='star', marker_size=15, name='Current'))
            fig_ml.add_trace(go.Scatter(x=years_fut.flatten(), y=fs_fut, mode='lines', name='AI Trend'))
            fig_ml.add_hline(y=1.1, line_color="red", line_dash="dash")
            fig_ml.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                xaxis_title="Year", yaxis_title="Factor of Safety", height=450)
            st.plotly_chart(fig_ml, width="stretch")
            
            if fs_actual < 1.1:
                # Case 1: Immediate and persistent critical instability detected
                st.error(f"⚠️ PREDICTION: Instability CURRENTLY DETECTED (2026) and persistent in the future.")
            else:
                # Case 2: Project intersection with the critical failure threshold (FS < 1.1)
                future_risk = years_fut.flatten()[np.where((years_fut.flatten() > 2026) & (fs_fut < 1.1))]
                if len(future_risk) > 0:
                    st.error(f"⚠️ PREDICTION: Possible instability detected starting from the year **{future_risk[0]}**.")
                else:
                    st.success("✅ PREDICTION: Stable trend estimated for the next 5 years.")
    else:
        with graph_col:
            st.info("👈 Execute the model to view the projection.")
