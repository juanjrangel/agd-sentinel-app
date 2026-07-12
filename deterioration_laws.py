"""
Alternative deterioration laws (R4-8): linear, exponential, power-law (accelerating),
ALL calibrated to pass through the two measured states (2017: c=18.5,phi=30.8 ->
2021: c=12.1, phi=21.3). Compare FS=1.0 crossing of each law. Seed-free (deterministic).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def FS(c,phi,gamma,z,beta,kh=0.0):
    b=np.radians(beta);p=np.radians(phi)
    cb,sb=np.cos(b),np.sin(b)
    return (c+(gamma*z*cb**2-kh*gamma*z*sb*cb)*np.tan(p))/(gamma*z*sb*cb+kh*gamma*z*cb**2)

C0,PHI0=18.5,30.8; C1,PHI1=12.1,21.3; DT_CAL=4.0   # 2017 -> 2021
GAMMA,Z,B=19.68,13.0,30.0
t=np.arange(0,8.0,1/24)  # 2017..2025 fine grid

# Law 1: LINEAR  c=c0-a t (a=(c0-c1)/4)
aL=(C0-C1)/DT_CAL; dL=(PHI0-PHI1)/DT_CAL
cL=np.clip(C0-aL*t,1,None); pL=np.clip(PHI0-dL*t,5,None)
# Law 2: EXPONENTIAL  c=c0 exp(-k t), k=ln(c0/c1)/4
kC=np.log(C0/C1)/DT_CAL; kP=np.log(PHI0/PHI1)/DT_CAL
cE=C0*np.exp(-kC*t); pE=PHI0*np.exp(-kP*t)
# Law 3: POWER (accelerating, p=2)  c=c0-A t^2, A=(c0-c1)/16
AC=(C0-C1)/DT_CAL**2; AP=(PHI0-PHI1)/DT_CAL**2
cP=np.clip(C0-AC*t**2,1,None); pP=np.clip(PHI0-AP*t**2,5,None)
# Law 4: POWER (decelerating, p=0.5)  c=c0-A sqrt(t), A=(c0-c1)/2
A5c=(C0-C1)/DT_CAL**0.5; A5p=(PHI0-PHI1)/DT_CAL**0.5
cD=np.clip(C0-A5c*np.sqrt(t),1,None); pD=np.clip(PHI0-A5p*np.sqrt(t),5,None)

def cr(fs):
    i=np.where(fs<=1.0)[0]
    if len(i)==0:return None
    i=i[0]
    if i==0:return 0.0
    t0,t1,f0,f1=t[i-1],t[i],fs[i-1],fs[i]
    return t0+(1-f0)*(t1-t0)/(f1-f0)

laws={"Power (p = 0.5, decelerating)":(cD,pD),"Exponential":(cE,pE),"Linear":(cL,pL),"Power (p = 2, accelerating)":(cP,pP)}
print(f"{'Law':>30}{'crossing t (yr)':>17}{'year':>9}")
res={}
for n,(c,p) in laws.items():
    fs=FS(c,p,GAMMA,Z,B); T=cr(fs); res[n]=(fs,T)
    print(f"{n:>30}{T:>17.2f}{2017+T:>9.2f}")
# verify all pass through both calibration points
for n,(c,p) in laws.items():
    i4=int(4/(1/24))
    assert abs(c[0]-C0)<1e-9 and abs(c[i4]-C1)<0.02, n
print("calibration check: all laws pass through both measured states  OK")

plt.rcParams.update({"figure.dpi":150,"font.size":9})
fig,ax=plt.subplots(figsize=(6.8,4.2))
cols={"Linear":"#1f6b45","Exponential":"#1f3b73","Power (p = 2, accelerating)":"#b3251f"}
for n,(fs,T) in res.items():
    ax.plot(2017+t,fs,color=cols[n],lw=2,label=f"{n}  (crossing {2017+T:.1f})")
    ax.plot(2017+T,1.0,"o",color=cols[n],ms=6)
ax.axhline(1.0,ls="--",color="k",lw=1)
ax.axvline(2021.6,color="red",alpha=0.30,lw=6)
ax.text(2021.72,1.28,"documented failure\n(Aug 2021)",color="red",fontsize=8)
# mark the two calibration states
for yy,fsv in [(2017,FS(C0,PHI0,GAMMA,Z,B)),(2021,FS(C1,PHI1,GAMMA,Z,B))]:
    ax.plot(yy,fsv,"ks",ms=7,mfc="white",zorder=5)
ax.annotate("measured 2017",(2017,FS(C0,PHI0,GAMMA,Z,B)),textcoords="offset points",xytext=(8,6),fontsize=8)
ax.annotate("measured 2021",(2021,FS(C1,PHI1,GAMMA,Z,B)),textcoords="offset points",xytext=(8,-12),fontsize=8)
ax.set_xlim(2016.8,2025);ax.set_ylim(0.55,1.45)
ax.set_xlabel("Year");ax.set_ylabel("Factor of Safety (static)")
ax.set_title("Structural uncertainty: alternative deterioration laws, same calibration")
ax.legend(fontsize=8,loc="lower left")
fig.tight_layout();fig.savefig("fig_laws.png");plt.close()
print("fig_laws.png written")
