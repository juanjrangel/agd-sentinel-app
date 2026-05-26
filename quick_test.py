# quick_test.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print("="*60)
print(" INITIATING QUICK TEST: AGD-SENTINEL (Cortinas Sector)")
print("="*60)

# Historical data for Cortinas, Toledo (Integrated for testing purposes)
historical_years = np.array([2016, 2017, 2018, 2019, 2020]).reshape(-1, 1)
historical_fs = np.array([1.50, 1.42, 1.31, 1.18, 1.05])

print("[*] Training polynomial regression model (n=3)...")
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(historical_years)
model = LinearRegression().fit(X_poly, historical_fs)

# Future prediction
future_years = np.array([2021, 2022]).reshape(-1, 1)
future_fs = model.predict(poly.transform(future_years))

print("\n--- PREDICTION RESULTS ---")
for year, fs in zip(future_years.flatten(), future_fs):
    if fs < 1.0:
        print(f"⚠️ Year {year}: FS = {fs:.2f} -> ALERT: Imminent failure projected.")
    else:
        print(f"✅ Year {year}: FS = {fs:.2f} -> Stable.")
print("="*60)
