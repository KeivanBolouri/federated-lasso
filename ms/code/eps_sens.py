import numpy as np, pandas as pd
from fedlasso import fedavg, consensus_admm, local_solve, tune_lambda, soft, select_threshold
R="/home/claude/work/repo/"
sites,val=[],[]
for f in [3,2,1]:
    sites.append((pd.read_csv(f"{R}node{f}_X_train.csv").to_numpy(float),
                  pd.read_csv(f"{R}node{f}_y_train.csv").to_numpy(float).ravel()))
    val.append((pd.read_csv(f"{R}node{f}_X_val.csv").to_numpy(float),
                pd.read_csv(f"{R}node{f}_y_val.csv").to_numpy(float).ravel()))
P=sites[0][0].shape[1]; m=np.array([len(y) for _,y in sites]); w=m/m.sum()
lams=[tune_lambda(X,y,Xv,yv)[0] for (X,y),(Xv,yv) in zip(sites,val)]
lam_bar=float(np.dot(w,lams))
tg=np.concatenate([[0.0],np.logspace(np.log10(lam_bar*1e-3),np.log10(lam_bar*5),39)])
EPS=[1e-6,1e-4,1e-2,1e-1]
rows=[]
for E in (1,5,20):
    b,r,_,_=fedavg(sites,w,lams,E,P)
    tau,_,_=select_threshold(b,val,w,tg); bst=soft(b,tau)
    for name,v in [("FedAvg",b),("FedAvg-ST",bst)]:
        rows.append(dict(method=name,E=E,**{f"eps_{e:g}":int((np.abs(v)>e).sum()) for e in EPS}))
z,_=consensus_admm(sites,w,lam_bar,P)
rows.append(dict(method="ADMM",E=np.nan,**{f"eps_{e:g}":int((np.abs(z)>e).sum()) for e in EPS}))
pd.DataFrame(rows).to_csv("eps_sensitivity.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
