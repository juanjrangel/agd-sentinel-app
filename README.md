# AGD-Sentinel: Geotechnical Degradation Prediction Framework ⛰️

AGD-Sentinel is an open-source hybrid deterministic-statistical software framework designed to project the temporal degradation of infinite slopes. The framework integrates pseudo-static Limit Equilibrium Methods (LEM) with polynomial regression machine learning algorithms to estimate the temporal decline of the Factor of Safety (FS).

This repository contains the source code, predictive models, and deployment configurations referenced in our manuscript submitted to *Computers & Geosciences*.

## 🌐 1. Cloud Deployment (Online Version)

To ensure universal accessibility, rapid testing, and peer-review evaluation, the application is deployed through Streamlit Cloud. No local installation is required.

* **Live application:** [https://agd-sentinel.streamlit.app/](https://agd-sentinel.streamlit.app/)

## 💻 2. Local Execution (For Users with Python Installed)

If Python 3.9 or higher is already installed on your system, the source code can be executed directly from this lightweight repository, which contains only the application code and not a full bundled Python distribution.

**Step-by-step instructions:**

1. Clone or download this repository as a `.zip` archive and extract it to your local machine.

2. Open a terminal or command prompt and navigate to the extracted folder.

3. Install the required mathematical and visualization dependencies by running:

   ```text
   pip install -r requirements.txt
   ```

4. Once installation is complete, the application can be launched using either of the following methods:

   * **Option A:** Double-click the `run_AGD_Sentinel.bat` file included in the repository.

   * **Option B:** Execute the command `streamlit run app.py` directly from the terminal.

5. The local server will start automatically, and the interactive dashboard will open in your default web browser.

## 📥 3. Portable Offline Version (For Users Without Python Installed)

For field applications in environments without internet connectivity, or for users who prefer not to install Python and its dependencies manually, a fully isolated and preconfigured offline environment has been packaged (approximately 700 MB).

This package includes a portable Python distribution (WinPython) and does not require any external configuration.

**Step-by-step instructions:**

1. Download the complete offline package: [https://bit.ly/4beO4Fv](https://bit.ly/4beO4Fv)

2. Extract the downloaded `.zip` folder to your local hard drive. *(**Important:** Do not execute the software directly from inside the `.zip` archive without extracting it first.)*

3. Open the extracted folder and simply double-click the `run_AGD_Sentinel.bat` script.

4. The batch launcher is preconfigured to use the internal WinPython environment included in the folder. It automatically isolates system variables, starts the local server, and opens the application in your web browser.

## 🔬 Scientific Framework

**AGD-Sentinel combines:**

* Pseudo-static infinite slope stability analysis
* Time-dependent geotechnical degradation functions
* Polynomial regression (degree *n = 3*)
* Monte Carlo uncertainty propagation
* Interactive visualization through Streamlit and Plotly

The computational workflow generates synthetic Factor of Safety trajectories from a physics-based LEM engine and approximates their temporal evolution using polynomial regression implemented with scikit-learn.

Monte Carlo simulations introduce Gaussian perturbations into geotechnical parameters (cohesion, friction angle, and seismic coefficient) to quantify uncertainty in projected failure-year estimates.

## 📂 Repository Structure

```text
/data               Synthetic datasets
/docs               Documentation
/results            Monte Carlo outputs
app.py              Main Streamlit application
requirements.txt    Python dependencies
README.md           Project documentation
LICENSE             Software license
```

## ⚙️ Main Dependencies

* Python 3.11
* NumPy
* scikit-learn
* Plotly
* Streamlit

*(All required libraries are listed in `requirements.txt`.)*

## ⚠️ Disclaimer

**AGD-Sentinel provides conditional temporal projections based on assumed geotechnical degradation rates and boundary conditions.** Results should be interpreted as engineering support scenarios rather than deterministic forecasts of landslide occurrence.

## 📖 Citation

If you use AGD-Sentinel in academic work, please cite:

> Buenahora Ballesteros, C.A., 2026.
>
> **AGD-Sentinel v1.0: Physics-informed temporal slope failure projection framework.**
> Zenodo. [https://doi.org/XXXXXXXX](https://doi.org/XXXXXXXX)

## 📄 License

This project is open-source and distributed under the **MIT License**.
