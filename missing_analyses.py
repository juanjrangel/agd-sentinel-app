"""
Additional analyses for missing reviewer points (R1-3,9b,11b,12,13,15,19)
Real Cortinas configuration. Seed = 11.
"""
import numpy as np, time
np.random.seed(11)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def FS(c,phi,gamma,z,beta,kh):
    b=np.radians(beta);p=np.radians(phi)
    cb,sb=np.cos(b),np.sin(b)
    return (c+(gamma*z*cb**2-kh*gamma*z*sb*cb)*np.tan(p))/(gamma*z*sb*cb+kh*gamma*z*cb**2)

C0,PHI0=18.5,30.8; ALPHA,DELTA=1.60,2.375; GAMMA,KH=19.68,0.0
T_MAX,DT=12.0,1/12; t=np.arange(0,T_MAX,DT); X=t.reshape(-1,1)
def traj(c0,p0,g,z,b,a=ALPHA,d=DELTA):
    return FS(np.clip(c0-a*t,1,None),np.clip(p0-d*t,5,None),g,z,b,KH)
def cross(fs):
    i=np.where(fs<=1.0)[0]
    if len(i)==0:return None
    i=i[0]
    if i==0:return 0.0
    t0,t1,f0,f1=t[i-1],t[i],fs[i-1],fs[i]
    return t0+(1-f0)*(t1-t0)/(f1-f0) if f1!=f0 else t1

fs0=traj(C0,PHI0,GAMMA,13,30); Ttrue=cross(fs0)
pf2=PolynomialFeatures(2)

# ---- R1-9b: rho sensitivity ----
print("[R1-9b] SENSITIVITY TO rho (-0.3 .. -0.8), N=1000 each")
print(f"{'rho':>6}{'mean t*':>9}{'std t*':>8}")
for rho in [-0.3,-0.5,-0.62,-0.8]:
    rng=np.random.default_rng(11)
    cov=[[9,rho*3*2.5],[rho*3*2.5,6.25]]
    s=rng.multivariate_normal([C0,PHI0],cov,1000)
    Ts=[cross(traj(np.clip(s[i,0],10,30),np.clip(s[i,1],20,40),
        rng.normal(GAMMA,0.5),rng.uniform(8,16),rng.uniform(26,34))) for i in range(1000)]
    Ts=[x for x in Ts if x is not None]
    print(f"{rho:>6}{np.mean(Ts):>9.3f}{np.std(Ts):>8.3f}")

# ---- R1-12: local error in critical region FS in [0.9,1.1] ----
print("\n[R1-12] LOCAL SURROGATE ERROR IN CRITICAL REGION (FS in [0.9,1.1]) vs GLOBAL")
ge,le=[],[]
for r in range(200):
    np.random.seed(100+r)
    yo=fs0+np.random.normal(0,0.05,len(t))
    m=LinearRegression().fit(pf2.fit_transform(X),yo)
    yh=m.predict(pf2.transform(X))
    ge.append(np.sqrt(np.mean((yh-fs0)**2)))
    msk=(fs0>=0.9)&(fs0<=1.1)
    le.append(np.sqrt(np.mean((yh[msk]-fs0[msk])**2)))
print(f"  global RMSE = {np.mean(ge):.4f} FS units;  critical-region RMSE = {np.mean(le):.4f} FS units")

# ---- R1-13: actual coefficients + physical consistency ----
print("\n[R1-13] FITTED QUADRATIC COEFFICIENTS (sigma=0.05, one representative fit)")
np.random.seed(5)
yo=fs0+np.random.normal(0,0.05,len(t))
m=LinearRegression().fit(pf2.fit_transform(X),yo)
b0,b1,b2=m.intercept_,m.coef_[1],m.coef_[2]
print(f"  FS(t) = {b0:.4f} + ({b1:.4f})t + ({b2:.5f})t^2")
print(f"  physical checks: beta1<0 (declining) -> {b1<0}; |beta2| small (near-linear decay) -> {abs(b2)<0.01}")
print(f"  dFS/dt at t in [0,12]: from {b1:.4f} to {b1+2*b2*12:.4f} -> monotonic decreasing: {(b1+2*b2*12)<0}")

# ---- R1-3: monotonicity violation frequency across noisy fits ----
print("\n[R1-3] MONOTONICITY CHECK (fraction of noisy fits with dFS/dt>0 anywhere in domain)")
viol=0
for r in range(500):
    np.random.seed(2000+r)
    yo=fs0+np.random.normal(0,0.05,len(t))
    m=LinearRegression().fit(pf2.fit_transform(X),yo)
    d=m.coef_[1]+2*m.coef_[2]*t
    if np.any(d>0):viol+=1
print(f"  violations: {viol}/500 = {viol/5:.1f}%  (quadratic on this smooth decay)")

# ---- R1-15: prediction-interval coverage ----
print("\n[R1-15] PREDICTION-INTERVAL COVERAGE (nominal 90%, noise-driven crossing distribution)")
# For 200 'worlds': generate noisy obs, fit, get crossing; build PI from bootstrap of residual noise
hits=0;trials=200
for w in range(trials):
    rng=np.random.default_rng(3000+w)
    yo=fs0+rng.normal(0,0.05,len(t))
    m=LinearRegression().fit(pf2.fit_transform(X),yo)
    # bootstrap crossing distribution under refit noise
    boot=[]
    for b in range(100):
        yb=m.predict(pf2.transform(X))+rng.normal(0,0.05,len(t))
        mb=LinearRegression().fit(pf2.fit_transform(X),yb)
        Tb=cross(mb.predict(pf2.transform(X)))
        if Tb is not None:boot.append(Tb)
    lo,hi=np.percentile(boot,[5,95])
    if lo<=Ttrue<=hi:hits+=1
print(f"  empirical coverage of 90% PI: {hits/trials*100:.1f}%  ({hits}/{trials})")

# ---- R1-19: timings ----
print("\n[R1-19] COMPUTATIONAL COST (this hardware, single core)")
t0=time.perf_counter()
for _ in range(10000): FS(18.5,30.8,GAMMA,13,30,KH)
t_lem=(time.perf_counter()-t0)/10000
t0=time.perf_counter()
for _ in range(1000): LinearRegression().fit(pf2.fit_transform(X),fs0)
t_fit=(time.perf_counter()-t0)/1000
t0=time.perf_counter()
rng=np.random.default_rng(1)
for i in range(1000):
    cross(traj(rng.normal(C0,3),rng.normal(PHI0,2.5),rng.normal(GAMMA,0.5),rng.uniform(8,16),rng.uniform(26,34)))
t_mc=time.perf_counter()-t0
print(f"  single LEM evaluation : {t_lem*1e6:.1f} microseconds")
print(f"  polynomial fit (144 pts): {t_fit*1e3:.2f} ms")
print(f"  full 1000-run Monte Carlo: {t_mc:.2f} s")

# ---- R1-11b: partial dependence (marginal shapes) ----
print("\n[R1-11b] PARTIAL DEPENDENCE OF t* (mean over 300 random co-samples)")
rng=np.random.default_rng(7)
def pd_curve(var,grid):
    out=[]
    for v in grid:
        Ts=[]
        for _ in range(300):
            c0=v if var=='c0' else np.clip(rng.normal(C0,3),10,30)
            p0=v if var=='phi0' else np.clip(rng.normal(PHI0,2.5),20,40)
            z=v if var=='z' else rng.uniform(8,16)
            b=v if var=='beta' else rng.uniform(26,34)
            g=rng.normal(GAMMA,0.5)
            Tp=cross(traj(c0,p0,g,z,b))
            if Tp is not None:Ts.append(Tp)
        out.append(np.mean(Ts))
    return out
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":140,"font.size":8})
fig,axs=plt.subplots(1,4,figsize=(11,2.8))
grids={'c0':np.linspace(12,25,7),'phi0':np.linspace(24,38,7),'z':np.linspace(8,16,7),'beta':np.linspace(26,34,7)}
labels={'c0':"c'0 (kPa)",'phi0':"phi'0 (deg)",'z':"z (m)",'beta':"beta (deg)"}
for ax,(k,g) in zip(axs,grids.items()):
    ax.plot(g,pd_curve(k,g),"o-",color="#1f3b73")
    ax.set_xlabel(labels[k]);ax.set_ylabel("mean t* (yr)")
fig.suptitle("Partial dependence of projected time-to-critical")
fig.tight_layout();fig.savefig("figPDP.png");plt.close()
print("  figure written: figPDP.png")
