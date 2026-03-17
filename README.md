# AGD-Sentinel: Geotechnical Degradation Predictive System ⛰️

AGD-Sentinel is an open-source, deterministic-statistical hybrid software designed to predict the temporal degradation of infinite slopes. It integrates pseudo-static Limit Equilibrium Models (LEM) with Polynomial Regression Machine Learning algorithms to forecast the Factor of Safety (FS) decay over time.

This repository contains the source code, predictive models, and deployment configurations referenced in our publication submitted to *Computers & Geosciences*.

---

## 🌐 1. Cloud Deployment (Online Version)
For universal access, rapid testing, and peer-review evaluation, the application is deployed via Streamlit Cloud. **No local installation is required.**

* **Live App:** https://agd-sentinel.streamlit.app/

---

## 💻 2. Local Execution (For users WITH Python installed)
If you already have a working Python 3.9+ environment on your machine, you can run the source code directly from this lightweight repository (which only contains the code, not the heavy Python engine).

**Step-by-step instructions:**
1. Clone or download this repository as a `.zip` file and extract it on your computer.
2. Open your terminal or command prompt and navigate to the extracted folder.
3. Install the required mathematical and visualization dependencies by running the following command: `pip install -r requirements.txt`
4. Once the installation is complete, you have two options to launch the application:
   * **Option A:** Double-click the `run_AGD_Sentinel.bat` file included in this repository. 
   * **Option B:** Run the command `streamlit run app.py` directly in your terminal.
5. The local server will start, and the interactive dashboard will automatically open in your default web browser.

---

## 📥 3. Offline Portable Version (For users WITHOUT Python installed)
For fieldwork deployment in environments without internet connectivity, or for users who do not wish to install Python and libraries manually, we have packaged a fully isolated, pre-configured offline environment (approx. 700 MB). 

This package includes a portable Python distribution (WinPython) and requires **zero external configuration**.

**Step-by-step instructions:**
1. **Download the Full Offline Package:** https://bit.ly/4beO4Fv
2. Extract the downloaded `.zip` folder to your local drive. *(Important: Do not run the software from inside the `.zip` file without extracting it first).*
3. Open the extracted folder and simply **double-click the `run_AGD_Sentinel.bat` script**.
4. The batch file is strictly pre-configured to utilize the internal WinPython engine included in the folder. It will automatically isolate system variables, start the local server, and open the application in your browser.

---

## 🔬 Scientific Foundation
The predictive machine learning engine utilizes a `scikit-learn` polynomial fit (degree n=3) to project the Factor of Safety (FS) decay curve. The validation and robustness of the model rely on Monte Carlo simulations, which introduce Gaussian noise to the initial geotechnical parameters (Cohesion, Internal Friction Angle, and Seismic Coefficients) to simulate inherent geological uncertainties.

---

## 📄 License
This project is open-source and available under the MIT License.
