import sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import indep_b4 as ib
# (a) kernel vs finite-difference Jacobian of probe outputs
chi=5; A,R,B,eta = ib.setup(chi, 0.7, 3, lr="ntk")
def f(y):
    a=y[:45].reshape(9,chi); r=y[45:70].reshape(chi,chi); b=y[70:].reshape(9,chi)
    W=a@r@b.T
    return np.einsum("ni,ij,nj->n", ib.PA, W, ib.PB)
y0=np.concatenate([A.ravel(),R.ravel(),B.ravel()])
J=np.zeros((128,y0.size)); h=1e-6
for k in range(y0.size):
    e=np.zeros_like(y0); e[k]=h; J[:,k]=(f(y0+e)-f(y0-e))/(2*h)
etav=np.concatenate([np.full(45,eta[0]),np.full(25,eta[1]),np.full(45,eta[2])])
Kfd=(J*etav)@J.T; Kan=ib.kernel(A,R,B,eta)
print("kernel rel err", np.linalg.norm(Kfd-Kan)/np.linalg.norm(Kan))
# (c) deep S2S widths and timing
for chi,sig in [(36,0.01),(64,0.004)]:
    t=time.time(); r=ib.run(chi,sig,0); print(chi,sig,r["label"],"widths",np.round(r["widths"],3),"times",np.round(r["times"],2),"rhos",np.round(r["rhos"],1),"dK %.3g l_end %.2g nfev %d %.1fs"%(r["dK"],r["l_end"],r["nfev"],time.time()-t))
for chi,sig in [(64,0.7723),(128,0.6),(3,2.52)]:
    t=time.time(); r=ib.run(chi,sig,0); print(chi,sig,r["label"],"dK %.4g l_end %.3g l_best %.3g nfev %d status %d %.1fs"%(r["dK"],r["l_end"],r["l_best"],r["nfev"],r["status"],time.time()-t))
