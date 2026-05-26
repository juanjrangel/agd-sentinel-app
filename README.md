# AGD-Sentinel: Geotechnical Degradation Prediction Framework ⛰️

AGD-Sentinel is an open-source hybrid deterministic-statistical software framework designed to project the temporal degradation of infinite slopes. The framework integrates pseudo-static Limit Equilibrium Methods (LEM) with polynomial regression machine learning algorithms to estimate the temporal decline of the Factor of Safety (FS).

This repository contains the source code, predictive models, and deployment configurations referenced in our manuscript submitted to *Computers & Geosciences*.

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

## ⚡ 3. Automated Quick Test (Reviewer Example)

To fulfill rapid testing requirements and verify the underlying physics-informed polynomial engine without launching the full web interface, an automated test script is provided.

**To run the test:**

1. Navigate to the repository folder in your terminal.

2. Execute the test script:

   ```text
   python quick_test.py
   ```

3. The console will output the prediction report, demonstrating the temporal FS degradation and failure year projection using the Cortinas case study baseline data.

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
/.devcontainer         Codespace configuration
app.py                 Main Streamlit application
quick_test.py          Automated test script (Cortinas dataset)
requirements.txt       Python dependencies
run_AGD_Sentinel.bat   Windows execution script
README.md              Project documentation
LICENSE                Software license (MIT)
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
