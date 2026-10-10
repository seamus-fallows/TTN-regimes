import sys, json, glob, numpy as np, torch
sys.path.insert(0,'C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality')
from rules import get_tree
from ttn.network import TTNSpec, bond_spectra
from ttn.teachers import pattern_teacher
SP='C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_locality_fix'
recs=[json.loads(l) for f in glob.glob(SP+'/out/spot.part*') for l in open(f)]
def pick(kind,n,S,start,sigma,seed):
    for r in recs:
        if r['kind']==kind and r['n']==n and r['S']==S and r['start']==start and abs(r['sigma']-sigma)<1e-9 and r['seed']==seed and r.get('traj'): return r
for args in [('binary',8,[0,2,4],'small_balanced',0.05,0),('binary',8,[0,2,4],'small_balanced',0.5,0),('ht',4,[0,1,2,3],'small_balanced',0.06,0),('ht',4,[0,1,2,3],'small_balanced',0.6,0)]:
    r=pick(*args)
    if r is None: print('missing',args); continue
    tree=get_tree(r['kind'],r['n']); spec=TTNSpec.uniform(tree,r['d'],r['chi'])
    nodes=tree.steiner_nodes(sorted(r['S'])); st=[v for v in tree.edges if v in nodes and tree.parent[v] in nodes]; ns=[v for v in tree.edges if v not in st]
    d=np.load(r['traj']); t,Y=d['t'],d['y']
    t1,t2=r['events'][0][0], r['events'][1][0]
    idx=np.where((t>t1)&(t<=t2*1.5))[0]; idx=idx[np.linspace(0,len(idx)-1,min(len(idx),300)).astype(int)]
    rel=[]; s2s=[]; leak=[]
    for i in idx:
        bs=bond_spectra(spec, spec.unflatten(torch.from_numpy(Y[:,i])))
        s2=np.array([float(bs[v][1]) for v in st]); s2s.append(s2.mean())
        rel.append((s2.max()-s2.min())/s2.mean())
        leak.append(max(float(bs[v][1]) for v in ns) if ns else 0)
    rel=np.array(rel); s2s=np.array(s2s)
    m=s2s>0.01*0.5
    print(args, 'k=',len(nodes),'Steiner edges',len(st),' rel spread of s2 across Steiner edges: median %.2e, max (s2>1%% s2*) %.2e'%(np.median(rel), rel[m].max() if m.any() else np.nan), ' max non-Steiner s2/s2* %.2e'%(max(leak)/0.5))
