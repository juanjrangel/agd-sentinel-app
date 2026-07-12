# AGD-Sentinel: Geotechnical Degradation Prediction Framework ⛰️

AGD-Sentinel is an open-source hybrid deterministic-statistical software framework designed to project the temporal degradation of infinite slopes. The framework integrates pseudo-static Limit Equilibrium Methods (LEM) with polynomial regression machine learning algorithms to estimate the temporal decline of the Factor of Safety (FS).

This repository contains the source code, predictive models, and deployment configurations referenced in our manuscript submitted to *Results in Engineering*.

## 🌐 1. Cloud Deployment (Online Version)

To ensure universal accessibility, rapid testing, and peer-review evaluation, the application is deployed through Streamlit Cloud. No local installation is required.

* **Live application:** https://agd-sentinel.streamlit.app/

## 💻 2. Local Execution (For Users with Python Installed)

If Python 3.9 or higher is already installed on your system, the source code can be executed directly from this lightweight repository.

**Step-by-step instructions:**

1. Clone or download this repository and extract it to your local machine.

2. Open a terminal or command prompt and navigate to the extracted folder.

3. Install the required mathematical and visualization dependencies by running:

   ```text
   pip install -r requirements.txt
   ```

4. The application can be launched using either of the following methods:

   * **Option A:** Double-click the `run_AGD_Sentinel.bat` file included in the repository.

   * **Option B:** Execute the command `streamlit run app.py` directly from the terminal.

## 📊 3. Automated Scripts (Reviewer Reproducibility)

To fulfill rapid testing requirements and verify the underlying physics-informed polynomial engine, the specific scripts below are provided. Each script reproduces specific figures and metrics from the manuscript:

* `master_run.py` (seed 42): Reproduces Figs. 2, 3, 6 + master_metrics.json
* `reviewer_analyses.py` (seed 7): Reproduces Figs. 4, 8, 9; baselines (Table 2); Sobol values
* `missing_analyses.py` (seed 11): Reproduces Fig. 7 (PDP); critical-region, coverage, timings
* `agd_corrected.py` (seed 2026): Reproduces Supplementary Figs. S1-S3
* `deterioration_laws.py` (deterministic, no seed): Reproduces Fig. 5 (fig_laws.png)

For detailed data configuration, literature sources, and methodology context required by the reviewers, please refer strictly to the `README_data.md` file included in this repository.

## 📥 4. Portable Offline Version (No Python Required)

For field applications in environments without internet connectivity, a fully isolated and preconfigured offline environment has been packaged (approximately 700 MB).

1. Download the complete offline package: https://www.dropbox.com/scl/fo/jmno52o6nv2f3ujq7prfq/AD9Serks9dcd2FZe9MAW9To?rlkey=lw9hajcv47s6wdxkrk0gkllcx&e=1&dl=0

2. Extract the downloaded `.zip` folder.

3. Double-click the `run_AGD_Sentinel.bat` script.

## 🔬 Scientific Framework

**AGD-Sentinel combines:**

* Pseudo-static infinite slope stability analysis
* Time-dependent geotechnical degradation functions
* Polynomial regression (degree *n = 3*)
* Monte Carlo uncertainty propagation
* Interactive visualization through Streamlit and Plotly

## 📂 Repository Structure

```text
/.devcontainer                 Codespace configuration
LICENSE                        Software license (MIT)
README.md                      Main project documentation
README_data.md                 Reviewer documentation and configuration details
agd_corrected.py               Supplementary analysis script
app.py                         Main Streamlit application
cortinas_campaign_data.csv     Geotechnical campaign data
cortinas_config.json           Model configuration and seeds
deterioration_laws.py          Alternative deterioration laws script
fig_laws.png                   Alternative laws figure
master_metrics.json            Consolidated outputs
master_run.py                  Main manuscript figures script
missing_analyses.py            Reviewer specific checks script
requirements.txt               Python dependencies
reviewer_analyses.py           Sensitivity and baseline analyses script
run_AGD_Sentinel.bat           Windows execution script
```

## ⚠️ Disclaimer

**AGD-Sentinel provides conditional temporal projections based on assumed geotechnical degradation rates and boundary conditions.** Results should be interpreted as engineering support scenarios rather than deterministic forecasts of landslide occurrence.

## 📖 Citation

If you use AGD-Sentinel in academic work, please cite:

> Buenahora Ballesteros, C.A., 2026.
>
> **AGD-Sentinel v1.0: Physics-informed temporal slope failure projection framework.**

## 📄 License

This project is open-source and distributed under the **MIT License**.
