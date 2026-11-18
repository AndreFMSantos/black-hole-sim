#!/usr/bin/env python3
"""Compares an ipole image (HDF5, unpolarised) with ours (--radio-out PFM, Jy/uas^2): flux, best orientation, L1 and per-pixel ratios. Usage: compare_images.py ipole.h5 ours.pfm (numpy, h5py)."""
import sys, h5py, numpy as np
def pfm(fn):
    with open(fn,'rb') as f:
        t=f.readline().strip(); w,h=map(int,f.readline().split()); s=float(f.readline())
        a=np.frombuffer(f.read(),dtype='<f4' if s<0 else '>f4')
    c=3 if t==b'PF' else 1
    return a.reshape(h,w,c)[...,0][::-1]  # PFM rows go bottom-up
ip=h5py.File(sys.argv[1],'r'); us=pfm(sys.argv[2])
I=np.array(ip['unpol']) if 'unpol' in ip else np.array(ip['pol'])[...,0]
sc=float(ip['header']['scale'][()]) if 'scale' in ip['header'] else 1.0
nx=I.shape[0]; fov=160.0; pix=(fov/nx)**2  # uas^2
Ij=I*sc/pix  # Jy per pixel -> Jy/uas^2
print('ipole raw shape',I.shape,'scale',sc,'sum Jy',I.sum()*sc,' ours sum Jy',us.sum()*pix)
best=None
for name,cand in [('id',Ij),('T',Ij.T),('flipud',Ij[::-1]),('fliplr',Ij[:,::-1]),('T flipud',Ij.T[::-1]),('T fliplr',Ij.T[:,::-1]),('rot180',Ij[::-1,::-1]),('T rot180',Ij.T[::-1,::-1])]:
    k=(cand*us).sum()/(us*us).sum(); d=np.abs(cand-k*us).sum()/np.abs(cand).sum()
    if best is None or d<best[1]: best=(name,d,k,cand)
name,d,k,cand=best
print(f'orientation {name}: ipole/ours best scale {k:.4f}, L1 difference after scaling {d:.4f}')
for t in (0.1,0.5):
    m=cand>t*cand.max(); print(f'  pixels above {t} of peak: ratio ipole/ours median {np.median(cand[m]/us[m]):.4f}, p10 {np.percentile(cand[m]/us[m],10):.4f}, p90 {np.percentile(cand[m]/us[m],90):.4f}')
np.save('ip.npy',cand); np.save('us.npy',us)
