import os as _os_bhs, pathlib as _pl_bhs
import os
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[2])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
# Oracle of the convert_athenak.py resampler: at cell centres (in the finest block that covers the point) the interpolation must return the cell value, exactly.
# Usage: uv run --with h5py --with numpy python -I tools/check_convert_athenak.py   (needs an AthenaK .bin in data/athenak; see the paths below)
import sys, numpy as np
sys.path.insert(0,os.path.join(_BHS_ROOT, 'code', 'tools')); sys.path.insert(0,os.path.join(_BHS_DATA, 'athenak/vis/python'))
import convert_athenak as c, bin_convert
d=bin_convert.read_binary(os.path.join(_BHS_DATA, 'athenak/run_d/T10k_d/bin/torus.mhd_w_bcc.00200.bin')); d['mb_data']={k:np.asarray(v) for k,v in d['mb_data'].items()}
lk=c.build_lookup(d); geo=np.asarray(d['mb_geometry']); lg=np.asarray(d['mb_logical']); nx=d['nx1_mb']
rng=np.random.default_rng(3); pts=[]; ref=[]
for _ in range(2000):
    m=rng.integers(0,len(geo)); i,j,k=rng.integers(1,nx-1,3); dx=(geo[m,1]-geo[m,0])/nx
    p=np.array([geo[m,0]+(i+.5)*dx,geo[m,2]+(j+.5)*dx,geo[m,4]+(k+.5)*dx])
    # only matters if this block is the finest one covering p
    fin=max(lg[mm,3] for mm in range(len(geo)) if geo[mm,0]<=p[0]<geo[mm,1] and geo[mm,2]<=p[1]<geo[mm,3] and geo[mm,4]<=p[2]<geo[mm,5])
    if lg[m,3]!=fin: continue
    pts.append(p); ref.append(d['mb_data']['dens'][m][k,j,i])
P,done=c.sample(d,lk,np.array(pts),['dens']); ref=np.array(ref)
print('n=%d, covered=%d, max relative error at cell centres = %.2e'%(len(pts),done.sum(),np.abs(P[:,0]-ref).max()/ref.min()), 'max abs', np.abs(P[:,0]-ref).max())
print('levels',sorted(lk), 'sizes',{L:lk[L][2] for L in lk})
