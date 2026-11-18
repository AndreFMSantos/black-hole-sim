# Independent oracle of the --hdr10 output: compares a frame (route-shot --hdr10 1000 --bloom 0 --glare 0) with PQ/BT.2020 computed in numpy from the linear EXR of the same frame. Usage: see notes/physics/25-video-hdr.md
import numpy as np, sys
from PIL import Image
import OpenEXR, Imath
S=sys.argv[1]
f=OpenEXR.InputFile(S+'/hx/frame_00100.exr'); dw=f.header()['dataWindow']; w=dw.max.x-dw.min.x+1; h=dw.max.y-dw.min.y+1
pt=Imath.PixelType(Imath.PixelType.FLOAT)
lin=np.stack([np.frombuffer(f.channel(c,pt),dtype=np.float32).reshape(h,w) for c in 'RGB'],-1).astype(np.float64)
n=lin*120.0; Y=n@np.array([.2126,.7152,.0722]); peak=1000.; Y0=.4*peak; W=peak-Y0
s=np.where(Y>Y0,(Y0+W*(1-np.exp(-(Y-Y0)/W)))/np.maximum(Y,1e-12),1.0); n=n*s[...,None]
M=np.array([[.6274,.3293,.0433],[.0691,.9195,.0114],[.0164,.0880,.8956]]); b=n@M.T
y=np.clip(b/1e4,0,1)**0.1593017578125; pq=((0.8359375+18.8515625*y)/(1+18.6875*y))**78.84375
ref=np.round(np.clip(pq,0,1)*255)
got=np.asarray(Image.open(S+'/hdr_shot.png').convert('RGB')).astype(float)
d=np.abs(ref-got); print('shape',lin.shape,got.shape,'max |diff| 8-bit codes: %.0f, mean %.4f, frac>1: %.5f'%(d.max(),d.mean(),(d>1).mean()))
print('PQ(100 nits)=%.4f PQ(203)=%.4f PQ(1000)=%.4f (ST2084: 0.5081 0.5807 0.7518)'%tuple(((0.8359375+18.8515625*((x/1e4)**0.1593017578125))/(1+18.6875*((x/1e4)**0.1593017578125)))**78.84375 for x in (100,203,1000)))
print('max linear',lin.max(),'max PQ code',got.max())
idx=np.argsort(-lin.sum(-1).ravel())[::4000][:10]
for i in idx:
    yy,xx=divmod(i,w); print((yy,xx),'lin',lin[yy,xx].round(3),'ref',ref[yy,xx],'got',got[yy,xx])
