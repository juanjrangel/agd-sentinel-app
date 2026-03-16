# AGD-Sentinel: Geotechnical Degradation Predictive System ⛰️

AGD-Sentinel is an open-source, deterministic-statistical hybrid software designed to predict the temporal degradation of infinite slopes using pseudo-static Limit Equilibrium Models (LEM) and Polynomial Regression Machine Learning algorithms.

This repository contains the source code and deployment files referenced in our publication in *Computers & Geosciences*.

## 🌐 1. Cloud Deployment (Online Version)
The application is deployed via Streamlit Cloud for universal access without local installation.
* **Live App:** https://agd-sentinel.streamlit.app/

## 💻 2. Local & Offline Execution
For fieldwork or environments without internet access, AGD-Sentinel can be executed locally. 

**Prerequisites:**
* Python 3.9 or higher (A portable distribution like WinPython is highly recommended for isolated environments).
* The libraries specified in `requirements.txt`.

**Execution Steps:**
1. Clone or download this repository as a `.zip` file.
2. Ensure your local Python environment has the required dependencies installed (`pip install -r requirements.txt`).
3. Double-click the provided `run_AGD_Sentinel.bat` file. This script will automatically launch the local Streamlit server and open the interactive dashboard in your default web browser.

## 🔬 Scientific Foundation
The predictive engine utilizes a `scikit-learn` polynomial fit ($n=3$) to project the Factor of Safety (FS) decay curve. The validation of the model relies on Monte Carlo simulations introducing Gaussian noise to the initial geotechnical parameters (Cohesion, Internal Friction Angle, and Seismic Coefficients).

## 📄 License
This project is open-source and available under the MIT License.
