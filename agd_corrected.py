"""
AGD-Sentinel - Corrected reproducible pipeline
==============================================
Built on the author's physical LEM engine, with two corrections:

(1) Observation noise is applied to the FS *estimate used for fitting*,
    NOT injected inside the time loop with first-dip stopping. The original
    per-step-noise + first-crossing stopping biases the failure year ~6 yr
    early (a first-passage artifact). Here the true failure year comes from
    the deterministic crossing; noise enters only the polynomial-fit step.

(2) For a single, characterised slope the geometry (beta, z) is KNOWN and
    held fixed at the Cortinas site values. Only the uncertain shear-strength
    parameters are sampled (with the empirical bivariate correlation). The
    depth behaviour is studied separately as a robustness sweep, clearly
    labelled as such - not as site uncertainty.

All initial values and rates below are the ones supplied by the software
author and remain ASSUMPTIONS to be calibrated/justified; every printed
number is a real output of this run (seed = 2026).
"""

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

SEED = 2026; np.random.seed(SEED)

# ----- Cortinas site geometry (FIXED - known, not sampled) -----
BETA, Z_SITE, GAMMA, KH = 28.0, 4.2, 18.5, 0.12
# ----- Assumed pristine initial state (author's values; to be calibrated) -----
MEAN_C0, STD_C0   = 28.0, 3.0
MEAN_PHI0, STD_PHI0 = 35.0, 2.5
RHO = -0.62
ALPHA, DELTA = 0.35, 0.20         # kPa/yr, deg/yr  (assumed; calibrate to site)
SIGMA_OBS = 0.05                  # observation noise on FS estimate
START_YEAR, T_MAX, DT = 2000.0, 100.0, 1/12
t_grid = np.arange(0, T_MAX, DT)

def fs_trajectory(c0, phi0, gamma, beta, kh, z):
    br = np.radians(beta)
    nstress = gamma*z*np.cos(br)**2
    dstress = gamma*z*np.sin(br)*np.cos(br) + kh*gamma*z*np.cos(br)**2
    c_t   = np.clip(c0 - ALPHA*t_grid, 1.0, None)
    phi_t = np.clip(phi0 - DELTA*t_grid, 5.0, None)
    return (c_t + nstress*np.tan(np.radians(phi_t))) / dstress

def crossing_year(fs):
    idx = np.where(fs <= 1.0)[0]
    if len(idx) == 0: return None
    i = idx[0]
    if i == 0: return START_YEAR
    t0,t1,f0,f1 = t_grid[i-1],t_grid[i],fs[i-1],fs[i]
    t = t0 + (1.0-f0)*(t1-t0)/(f1-f0) if f1!=f0 else t1
    return START_YEAR + t

# ---------- consistency check against the measured 2026 state ----------
print("="*64); print("CONSISTENCY CHECK"); print("="*64)
fs_today = fs_trajectory(MEAN_C0, MEAN_PHI0, GAMMA, BETA, KH, Z_SITE)
yr_c_116 = START_YEAR + (MEAN_C0-11.6)/ALPHA
print(f"With pristine c0={MEAN_C0} kPa and alpha={ALPHA} kPa/yr, cohesion reaches the")
print(f"Table-1 measured value (11.6 kPa) only in year {yr_c_116:.0f} - but it is")
print(f"measured at 11.6 kPa in 2026. Implied rate to match 2026 = {(MEAN_C0-11.6)/26:.2f} kPa/yr.")
print("=> the assumed rate/initial state should be calibrated so the model")
print("   reproduces the known 2026 condition. (Flag for the team.)\n")

# ---------- degree optimisation (5-fold CV on a noisy trajectory) ----------
print("="*64); print("POLYNOMIAL DEGREE (5-fold CV)"); print("="*64)
fs_true0 = fs_trajectory(MEAN_C0, MEAN_PHI0, GAMMA, BETA, KH, Z_SITE)
mask = fs_true0 >= 0.6                      # fit the meaningful pre/at-failure window
Xg = t_grid[mask].reshape(-1,1); yg = fs_true0[mask] + np.random.normal(0,SIGMA_OBS,mask.sum())
kf = KFold(5, shuffle=True, random_state=SEED); rmses=[]
for n in range(1,7):
    fr=[]
    for tr,te in kf.split(Xg):
        pf=PolynomialFeatures(n); m=LinearRegression().fit(pf.fit_transform(Xg[tr]),yg[tr])
        fr.append(np.sqrt(mean_squared_error(yg[te],m.predict(pf.transform(Xg[te])))))
    rmses.append(np.mean(fr)); print(f"  n={n}: CV-RMSE={rmses[-1]:.5f}")
best_n=int(np.argmin(rmses))+1; print(f"  -> optimal degree n = {best_n}")

# ---------- main Monte Carlo: fixed geometry, sample strength ----------
print("\n"+"="*64); print("MONTE CARLO (N=1000, fixed site geometry)"); print("="*64)
N=1000; np.random.seed(SEED)
cov=[[STD_C0**2,RHO*STD_C0*STD_PHI0],[RHO*STD_C0*STD_PHI0,STD_PHI0**2]]
samples=np.random.multivariate_normal([MEAN_C0,MEAN_PHI0],cov,N)
c0s=np.clip(samples[:,0],15,45); phi0s=np.clip(samples[:,1],25,45)
gs=np.random.normal(GAMMA,0.5,N); khs=np.clip(np.random.normal(KH,0.02,N),0,None)
pf=PolynomialFeatures(best_n)
Tact,Tpred,rows=[],[],[]
for i in range(N):
    fs=fs_trajectory(c0s[i],phi0s[i],gs[i],BETA,khs[i],Z_SITE)
    Ta=crossing_year(fs)
    if Ta is None: continue
    m=mask & (fs>=0.6)
    if m.sum()<best_n+2: continue
    yobs=fs[m]+np.random.normal(0,SIGMA_OBS,m.sum())
    reg=LinearRegression().fit(pf.fit_transform(t_grid[m].reshape(-1,1)), yobs)
    fs_hat=reg.predict(pf.transform(t_grid.reshape(-1,1)))
    Tp=crossing_year(fs_hat)
    if Tp is None: continue
    Tact.append(Ta); Tpred.append(Tp)
    rows.append([c0s[i],phi0s[i],gs[i],khs[i],Ta])
Tact=np.array(Tact); Tpred=np.array(Tpred); res=Tpred-Tact
print(f"  valid realisations  : {len(res)}")
print(f"  median T* (computed): {np.median(Tact):.2f}")
print(f"  residual mean / std : {res.mean():+.3f} / {res.std():.3f} yr")
print(f"  within +/-0.5 yr    : {np.mean(np.abs(res)<=0.5)*100:.1f}%")
print(f"  within +/-1.0 yr    : {np.mean(np.abs(res)<=1.0)*100:.1f}%")

# ---------- parameter sensitivity (fixed geometry) ----------
df=pd.DataFrame(rows,columns=['c0','phi0','gamma','kh','T_actual'])
cor=df.corr()['T_actual'].drop('T_actual').abs().sort_values(ascending=False)
print("\n  Parameter influence on failure year (|correlation|, geometry fixed):")
for k,v in cor.items(): print(f"    {k:>6}: {v:.2f}")
df.to_csv("resultados_corregidos.csv",index=False)

# ---------- depth-robustness sweep (separate, labelled analysis) ----------
print("\n"+"="*64); print("DEPTH-ROBUSTNESS SWEEP (method behaviour, not site uncertainty)"); print("="*64)
np.random.seed(SEED+1); depth_bins=[(2,4),(4,6),(6,8),(8,10),(10,12)]; groups=[]
for lo,hi in depth_bins:
    rr=[]
    for _ in range(220):
        c0=np.clip(np.random.normal(MEAN_C0,STD_C0),15,45)
        phi0=np.clip(np.random.normal(MEAN_PHI0,STD_PHI0),25,45)
        z=np.random.uniform(lo,hi)
        fs=fs_trajectory(c0,phi0,GAMMA,BETA,KH,z); Ta=crossing_year(fs)
        if Ta is None: continue
        m=(fs>=0.6)
        if m.sum()<best_n+2: continue
        yobs=fs[m]+np.random.normal(0,SIGMA_OBS,m.sum())
        reg=LinearRegression().fit(pf.fit_transform(t_grid[m].reshape(-1,1)),yobs)
        Tp=crossing_year(reg.predict(pf.transform(t_grid.reshape(-1,1))))
        if Tp: rr.append(Tp-Ta)
    groups.append(rr); print(f"  {lo}-{hi} m: n={len(rr)}, IQR width={np.subtract(*np.percentile(rr,[75,25])):.2f} yr")

# ---------- figures ----------
plt.rcParams.update({"figure.dpi":140,"font.size":10})
fig,ax1=plt.subplots(figsize=(7,4.2)); ax2=ax1.twinx()
ax2.plot(range(1,7),rmses,"s-",color="#b3251f"); ax1.axvline(best_n,ls="--",color="green")
ax1.set_xlabel("Polynomial degree (n)"); ax2.set_ylabel("CV-RMSE",color="#b3251f")
ax1.set_yticks([]); ax1.set_title("Polynomial-degree optimisation (cross-validated)")
fig.tight_layout(); fig.savefig("fig_degree.png"); plt.close()

fig,ax=plt.subplots(figsize=(7,4.2))
ax.hist(res,bins=35,density=True,color="#6f9bd1",edgecolor="white")
xs=np.linspace(res.min(),res.max(),200); ax.plot(xs,norm.pdf(xs,res.mean(),res.std()),color="#b3251f",lw=2)
ax.set_xlabel("Residual  T_pred - T_actual (yr)"); ax.set_ylabel("Density")
ax.set_title(f"Residuals (mu={res.mean():+.2f}, sigma={res.std():.2f}, n={len(res)})")
fig.tight_layout(); fig.savefig("fig_residuals.png"); plt.close()

fig,ax=plt.subplots(figsize=(7,4.2))
ax.boxplot(groups,labels=[f"{lo}-{hi} m\n(n={len(g)})" for (lo,hi),g in zip(depth_bins,groups)])
ax.axhline(0,ls="--",color="red"); ax.set_ylabel("Residual (yr)"); ax.set_xlabel("Failure-surface depth")
ax.set_title("Predictive error across depth (robustness sweep)")
fig.tight_layout(); fig.savefig("fig_depth.png"); plt.close()

fig,ax=plt.subplots(figsize=(7,4.2))
ax.barh(cor.index[::-1],cor.values[::-1],color="#1f3b73")
ax.set_xlabel("|correlation| with failure year"); ax.set_title("Parameter influence (geometry fixed)")
fig.tight_layout(); fig.savefig("fig_sensitivity.png"); plt.close()
print("\nFigures: fig_degree.png, fig_residuals.png, fig_depth.png, fig_sensitivity.png")
