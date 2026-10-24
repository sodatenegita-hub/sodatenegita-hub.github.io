import numpy as np, sys
from PIL import Image
WHITE=np.array([255,255,255.]);CREAM=np.array([255,244,230.])
ORANGE=np.array([247,148,29.])
def fit_circle(a):
    m=(np.abs(a-CREAM).sum(2)<4); m[:, :1300]=False
    pts=[]
    for y in range(470,1340,20):
        r=np.where(m[y])[0]
        if len(r): pts.append((r.min(),y))
    xs=np.array([x for x in range(1600,2560,40)]); 
    for x in xs:
        c=np.where(m[:,x])[0]
        if len(c): pts.append((x,c.min()))
    P=np.array(pts,float); A=np.c_[2*P,np.ones(len(P))]; b=(P**2).sum(1)
    cx,cy,c=np.linalg.lstsq(A,b,rcond=None)[0]; return cx,cy,np.sqrt(c+cx*cx+cy*cy)
def recolor(src,dst,bg,circ,sub=None,box=None):
    a=np.array(Image.open(src).convert('RGB')).astype(float)
    H,W,_=a.shape; cx,cy,r=fit_circle(a)
    yy,xx=np.mgrid[0:H,0:W]; d=np.sqrt((xx-cx)**2+(yy-cy)**2)-r   # <0 inside
    inside=np.clip(0.5-d,0,1)[...,None]          # anti-aliased circle coverage
    oldbg=WHITE*(1-inside)+CREAM*inside
    newbg=np.array(bg,float)*(1-inside)+np.array(circ,float)*inside
    # how much of each pixel is background: compare to old background, fg assumed darkest channel
    PAL=np.array([[247,148,29],[113,102,88],[57,181,74],[241,89,42],[163,154,142],[141,198,63],[46,74,98]],float)
    best=None
    for f in PAL:
        v=f-oldbg; pv=a-oldbg
        t=np.clip((pv*v).sum(2)/np.maximum((v*v).sum(2),1),0,1)
        res=((pv-t[...,None]*v)**2).sum(2)
        if best is None: best=res; T=t
        else:
            m=res<best; best=np.where(m,res,best); T=np.where(m,t,T)
    w=(1-T)[...,None]
    out=a+w*(newbg-oldbg)
    if sub is not None:   # 小さい文字の帯だけ、オレンジを別の色へ
        x0,y0,x1,y1=box
        reg=a[y0:y1,x0:x1]; t=np.clip((255-reg[...,2])/(255-29),0,1)[...,None]
        isor=(np.abs(reg[...,0]-255)<12)[...,None]   # 白〜オレンジの線上
        nb=newbg[y0:y1,x0:x1]
        rep=nb*(1-t)+np.array(sub,float)*t
        out[y0:y1,x0:x1]=np.where(isor&(t>0.02),rep,out[y0:y1,x0:x1])
    Image.fromarray(np.clip(out,0,255).round().astype('uint8')).save(dst)
if __name__=='__main__':
    pass
