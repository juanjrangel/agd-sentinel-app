"""
AGD-Sentinel: Full Hybrid Physics-Informed Reproducibility Script
-----------------------------------------------------------------
This script implements the complete hybrid pipeline:
1. Geotechnical Limit Equilibrium Method (LEM) Engine.
2. Temporal degradation of c' and phi' (alpha, delta).
3. Monte Carlo sampling with bivariate correlation (rho = -0.62).
4. Exact T* computation (Year FS drops to 1.0).
5. SHAP Global Explainability on the physical outputs.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print(" AGD-SENTINEL: FULL HYBRID PHYSICS-INFORMED REPRODUCIBILITY")
print("="*65)

# ==============================================================================
# 1. PHYSICAL LEM ENGINE & DEGRADATION FUNCTION
# ==============================================================================
def calculate_failure_year(c0, phi0, gamma, beta, kh, z, start_year=2000.0):
    """
    Calculates the exact year of failure by stepping through time and 
    applying degradation to cohesion and friction until FS <= 1.0.
    """
    alpha = 0.35  # kPa/year (Cohesion degradation rate)
    delta = 0.20  # deg/year (Friction degradation rate)
    
    beta_rad = np.radians(beta)
    
    # Stresses (Simplified Pseudo-Static Infinite Slope)
    normal_stress = gamma * z * (np.cos(beta_rad)**2)
    driving_stress = (gamma * z * np.sin(beta_rad) * np.cos(beta_rad)) + (kh * gamma * z * (np.cos(beta_rad)**2))
    
    # Time-stepping simulation (Monthly resolution = 1/12 year)
    dt = 1/12
    max_years = 100.0
    
    for t in np.arange(0, max_years, dt):
        # Apply linear weathering degradation
        c_t = c0 - (alpha * t)
        phi_t = phi0 - (delta * t)
        
        # Physical bounds constraint (soil cannot have negative strength)
        if c_t <= 1.0: c_t = 1.0
        if phi_t <= 5.0: phi_t = 5.0
            
        phi_rad = np.radians(phi_t)
        
        # Resisting stress calculation
        resisting_stress = c_t + (normal_stress * np.tan(phi_rad))
        
        # Factor of Safety calculation
        fs = resisting_stress / driving_stress
        
        # Add explicit observation noise (sigma_FS = 0.05) as stated in methodology
        fs_measured = fs + np.random.normal(0, 0.05)
        
        if fs_measured <= 1.0:
            return start_year + t
            
    return start_year + max_years # Return max if no failure is reached

# ==============================================================================
# 2. BIVARIATE CORRELATED SAMPLING (Pristine State t=0)
# ==============================================================================
print("[*] Stage 1: Initializing Monte Carlo Sampling (Pristine State)...")
N = 1000
np.random.seed(2026)

# Pristine geotechnical baseline parameters (Year 2000)
mean_c0, std_c0 = 28.0, 3.0
mean_phi0, std_phi0 = 35.0, 2.5
rho = -0.62 

print(f"    -> Applied Bivariate Correlation (c0', phi0'): rho = {rho}")
cov_matrix = [
    [std_c0**2, rho * std_c0 * std_phi0], 
    [rho * std_c0 * std_phi0, std_phi0**2]
]

c_phi_samples = np.random.multivariate_normal([mean_c0, mean_phi0], cov_matrix, N)
c0_samples = np.clip(c_phi_samples[:, 0], 15, 45)
phi0_samples = np.clip(c_phi_samples[:, 1], 25, 45)

gamma_samples = np.random.uniform(16, 22, N)
beta_samples = np.random.uniform(20, 35, N)
kh_samples = np.random.uniform(0.05, 0.15, N)
z_samples = np.random.uniform(2.0, 12.0, N)

X_df = pd.DataFrame({
    'Cohesion_0 (kPa)': c0_samples,
    'Friction_0 (deg)': phi0_samples,
    'Unit Weight (kN/m3)': gamma_samples,
    'Seismic Coef (kh)': kh_samples,
    'Slope Angle (deg)': beta_samples,
    'Depth (m)': z_samples
})

# ==============================================================================
# 3. PHYSICAL SIMULATION (Data Generation)
# ==============================================================================
print("[*] Stage 2: Executing Physical LEM Engine (Degradation Loop)...")
T_predictions = []

for idx, row in X_df.iterrows():
    t_fail = calculate_failure_year(
        c0=row['Cohesion_0 (kPa)'],
        phi0=row['Friction_0 (deg)'],
        gamma=row['Unit Weight (kN/m3)'],
        beta=row['Slope Angle (deg)'],
        kh=row['Seismic Coef (kh)'],
        z=row['Depth (m)']
    )
    T_predictions.append(t_fail)

X_df['Predicted_T_star'] = T_predictions
T_median = np.median(T_predictions)
print(f"    -> Computed Median Critical Year (T*): {T_median:.2f}")

# ==============================================================================
# 4. GLOBAL EXPLAINABILITY (SHAP PIPELINE)
# ==============================================================================
print("[*] Stage 3: Running SHAP Global Explainability Pipeline on Physical Data...")
features = X_df.drop(columns=['Predicted_T_star'])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, X_df['Predicted_T_star'])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': features.columns,
    'SHAP Importance': mean_abs_shap
})
shap_importance['Relative Weight (%)'] = (shap_importance['SHAP Importance'] / shap_importance['SHAP Importance'].sum()) * 100
shap_importance = shap_importance.sort_values(by='Relative Weight (%)', ascending=False).reset_index(drop=True)

print("\n--- SHAP GLOBAL FEATURE IMPORTANCE RANKING ---")
for index, row in shap_importance.iterrows():
    print(f"{row['Feature']:>25} : {row['Relative Weight (%)']:5.1f}%")
print("----------------------------------------------")

csv_filename = "resultados_fisicos_completos.csv"
X_df.to_csv(csv_filename, index=False)
print(f"[*] SUCCESS: Full physical dataset exported to '{csv_filename}'")
print("="*65)
