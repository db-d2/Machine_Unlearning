#!/usr/bin/env python3
"""Reproducible source for Figure 1 (eval_pipeline). Emits eval_pipeline.svg.
All numbers are the current paper values; edit them here, never in a raster asset."""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_pipeline.svg")

# ---- palette ----
BG="#e9eef2"; CARD="#ffffff"; BORDER="#d9e1e9"; INK="#1f2a33"; MUTED="#5c6b77"
BADGE="#26333f"; RETAIN="#34618c"; UNSEEN="#bcd6ea"; FORGET="#db6b4f"
ENC="#4e8fc0"; DEC="#cf7357"; BAND="#cfe0ee"; GREEN="#2e8b62"; GREY="#8a97a2"; ORANGE="#d9963f"
SANS="'Helvetica Neue',Helvetica,Arial,sans-serif"; MONO="'DejaVu Sans Mono','Menlo',monospace"
W,H=1400,772
E=[]
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def rr(x,y,w,h,fill,stroke="none",rx=12,sw=1,op=1):
    E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')
def tx(x,y,s,size=13,fill=INK,w="normal",fam=SANS,anc="start",it=False,ls=None):
    st=f' letter-spacing="{ls}"' if ls else ""
    E.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anc}" font-style="{"italic" if it else "normal"}"{st}>{esc(s)}</text>')
def ln(x1,y1,x2,y2,stroke=MUTED,sw=1.6,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def arrow(x1,y1,x2,y2):
    ln(x1,y1,x2,y2,MUTED,1.8); E.append(f'<polygon points="{x2},{y2} {x2-8},{y2-4} {x2-8},{y2+4}" fill="{MUTED}"/>') if x2>x1 else E.append(f'<polygon points="{x2},{y2} {x2},{y2-6} {x2-8 if x2>x1 else x2+0},{y2}" fill="{MUTED}"/>')
def badge(x,y,letter):
    rr(x,y,22,22,BADGE,rx=5); tx(x+11,y+16,letter,14,"#fff",w="700",anc="middle")
def swatch(x,y,color,label):
    rr(x,y,13,13,color,rx=3); tx(x+19,y+11,label,11.5,MUTED)

E.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SANS}">')
rr(0,0,W,H,BG,rx=0)
# soft shadow filter
E.append('<defs><filter id="sh" x="-4%" y="-4%" width="108%" height="112%"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#243244" flood-opacity="0.10"/></filter></defs>')

def card(x,y,w,h): E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{CARD}" stroke="{BORDER}" stroke-width="1" filter="url(#sh)"/>')

# ============ PANEL A : DATASETS ============
ax,ay,aw,ah=22,22,772,338; card(ax,ay,aw,ah)
badge(ax+22,ay+20,"A"); tx(ax+54,ay+38,"Datasets",19,INK,"700")
tx(ax+22,ay+68,"Three cohorts, two modalities. Each structured forget set is a rare",12.5,MUTED)
tx(ax+22,ay+85,"biologically coherent subpopulation.",12.5,MUTED)
def grid(gx,gy,cols,rows,cell,gap,fn):
    for r in range(rows):
        for c in range(cols):
            E.append(f'<rect x="{gx+c*(cell+gap)}" y="{gy+r*(cell+gap)}" width="{cell}" height="{cell}" rx="1.4" fill="{fn(c,r)}"/>')
def dcard(dx,title,sub,line1,line2,fn,light_grid=False):
    dy=ay+104; dw=232; dh=182
    E.append(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="10" fill="#f6f9fb" stroke="{BORDER}" stroke-width="1"/>')
    tx(dx+16,dy+26,title,14,INK,"700"); tx(dx+16,dy+43,sub,10.5,MUTED,fam=MONO)
    grid(dx+16,dy+56,18,7,9,2.2,fn)
    tx(dx+16,dy+152,line1,10.5,INK,fam=MONO)
    tx(dx+16,dy+170,"forget: "+line2,10.5,FORGET,fam=MONO)
def pbmc(c,r): return FORGET if (14<=c<=16 and 0<=r<=1) else (UNSEEN if c>=16 else RETAIN)
def tm(c,r):   return FORGET if (2<=c<=4 and 3<=r<=4) else (UNSEEN if c>=16 else RETAIN)
def tcga(c,r): return FORGET if (0<=c<=4 and 0<=r<=2) else (UNSEEN if (c+r)%2==0 else "#9fc0dd")
dcard(ax+22,"PBMC-33k","single-cell RNA-seq","33,088 cells · 2,000 HVGs","30 megakaryocytes",pbmc)
dcard(ax+22+250,"Tabula Muris","single-cell RNA-seq","41,647 cells · 2,000 HVGs","82 cardiac muscle",tm)
dcard(ax+22+500,"TCGA-BRCA","bulk RNA-seq","1,089 patients · 978 L1000","158 Basal subtype",tcga)
ly=ay+ah-22
swatch(ax+22,ly-9,RETAIN,"retain cells"); swatch(ax+150,ly-9,UNSEEN,"unseen (matched negatives)"); swatch(ax+360,ly-9,FORGET,"forget set (structured)")

# ============ PANEL B : VAE ============
bx,by,bw,bh=810,22,568,338; card(bx,by,bw,bh)
badge(bx+22,by+20,"B"); tx(bx+54,by+38,"VAE",19,INK,"700")
tx(bx+22,by+66,"scVI-style encoder-decoder with shared output parameters.",12,MUTED)
tx(bx+22,by+83,"Latent dim 32.",12,MUTED)
tx(bx+120,by+118,"encoder",12.5,ENC,"700",anc="middle")
tx(bx+400,by+118,"decoder (shared)",12.5,DEC,"700",anc="middle")
mid=by+205
def blk(x,w,h,fill,label):
    rr(x,mid-h/2,w,h,fill,rx=4); tx(x+w/2,mid+4,label,11,"#fff",w="700",anc="middle",fam=MONO)
def io(x,label,under):
    rr(x,mid-26,34,52,"#e4e9ee",rx=4,stroke=BORDER,sw=1); tx(x+17,mid+5,label,13,INK,"700",anc="middle")
    tx(x+17,mid+48,under,10,MUTED,anc="middle",fam=MONO)
io(bx+26,"X","genes")
enc_x=bx+80
for w_,h_,lab in [(34,84,"1024"),(30,60,"512"),(26,40,"128")]:
    blk(enc_x,w_,h_,ENC,lab); enc_x+=w_+8
rr(bx+228,mid-22,34,44,"#cfe6f2",rx=4,stroke=ENC,sw=1.2); tx(bx+245,mid+4,"Z",13,INK,"700",anc="middle"); tx(bx+245,mid+48,"latent",10,MUTED,anc="middle",fam=MONO)
dec_x=bx+280
for w_,h_,lab in [(26,40,"128"),(30,60,"512"),(34,84,"1024")]:
    blk(dec_x,w_,h_,DEC,lab); dec_x+=w_+8
io(bx+420,"X̂","reconstructed")
# small flow arrows
for a,b in [(bx+60,bx+78),(bx+206,bx+226),(bx+264,bx+278),(bx+406,bx+418)]:
    ln(a,mid,b,mid,MUTED,1.4)
tx(bx+22,by+bh-64,"L = −E[log p(x|z)] + KL[q(z|x) ‖ N(0,I)]",12,INK,fam=MONO)
# legend
E.append(f'<circle cx="{bx+470}" cy="{by+150}" r="6" fill="{FORGET}"/>'); tx(bx+482,by+155,"forget",11,MUTED)
E.append(f'<circle cx="{bx+470}" cy="{by+172}" r="6" fill="{RETAIN}"/>'); tx(bx+482,by+177,"retain",11,MUTED)
tx(bx+bw-20,by+bh-38,"both pass through",10,MUTED,anc="end"); tx(bx+bw-20,by+bh-24,"the same parameters",10,MUTED,anc="end")

# arrows between A/B -> C
arrow(ax+aw-2,by+140,bx+2,by+140)  # train (A->B)
tx((ax+aw+bx)/2,by+132,"train",10.5,MUTED,anc="middle",fam=MONO)
E.append(f'<path d="M {bx+40} {by+bh+2} L {bx+40} {by+bh+18} L {ax+aw*0.34} {by+bh+18} L {ax+aw*0.34} 378" fill="none" stroke="{MUTED}" stroke-width="1.6"/>')
E.append(f'<polygon points="{ax+aw*0.34},380 {ax+aw*0.34-4},372 {ax+aw*0.34+4},372" fill="{MUTED}"/>')

# ============ PANEL C : MEMBERSHIP INFERENCE ============
cx,cy,cw,ch=22,382,500,368; card(cx,cy,cw,ch)
badge(cx+22,cy+20,"C"); tx(cx+54,cy+38,"Membership inference",19,INK,"700")
tx(cx+22,cy+66,"Trained MLP attacker on 70-dim VAE latent features",12,MUTED)
tx(cx+22,cy+83,"classifies forget vs. matched-negative samples.",12,MUTED)
vy=cy+150
rr(cx+24,vy-24,58,48,"#e4e9ee",rx=6,stroke=BORDER,sw=1); tx(cx+53,vy+4,"VAE",13,INK,"700",anc="middle")
arrow(cx+84,vy,cx+120,vy); tx(cx+102,vy-8,"query",9.5,MUTED,anc="middle",fam=MONO)
# dot rows
tx(cx+128,vy-16,"forget set",10,FORGET,fam=MONO)
for i in range(5): E.append(f'<circle cx="{cx+134+i*15}" cy="{vy-4}" r="5" fill="{FORGET}"/>')
tx(cx+128,vy+20,"matched neg. (k=10)",10,RETAIN,fam=MONO)
for i in range(5): E.append(f'<circle cx="{cx+134+i*15}" cy="{vy+32}" r="5" fill="{RETAIN if i<4 else UNSEEN}"/>')
arrow(cx+220,vy,cx+250,vy)
rr(cx+250,vy-30,96,60,ENC,rx=8); tx(cx+298,vy-2,"MLP",13,"#fff","700",anc="middle"); tx(cx+298,vy+16,"attacker",12,"#fff","700",anc="middle")
arrow(cx+348,vy,cx+380,vy)
tx(cx+386,vy-4,"AUC",13,INK,"700"); tx(cx+386,vy+16,"advantage =",10,MUTED,fam=MONO); tx(cx+386,vy+31,"2 · |AUC − 0.5|",10,INK,fam=MONO)
# reference box
ry=cy+ch-92
E.append(f'<rect x="{cx+22}" y="{ry}" width="{cw-44}" height="66" rx="9" fill="#f6f9fb" stroke="{BORDER}" stroke-width="1"/>')
tx(cx+38,ry+24,"Reference (PBMC, structured):",12,INK,"700")
tx(cx+38,ry+48,"baseline AUC ",11,MUTED,fam=MONO); tx(cx+150,ry+48,"0.79",11,FORGET,"700",fam=MONO)
tx(cx+200,ry+48,"retrain AUC ",11,MUTED,fam=MONO); tx(cx+300,ry+48,"0.58",11,RETAIN,"700",fam=MONO)
tx(cx+350,ry+48,"chance ",11,MUTED,fam=MONO); tx(cx+408,ry+48,"0.50",11,MUTED,"700",fam=MONO)
arrow(cx+cw-2,vy+40,540,vy+40)  # C -> D

# ============ PANEL D : UNLEARNING & EVALUATION ============
dx,dy,dw,dh=540,382,838,368; card(dx,dy,dw,dh)
badge(dx+22,dy+20,"D"); tx(dx+54,dy+38,"Unlearning & evaluation",19,INK,"700")
tx(dx+22,dy+66,"Nine post-hoc methods plus two training-time approaches and a DP-SGD",12,MUTED)
tx(dx+22,dy+83,"(from scratch) baseline. Only retrain reaches the target region.",12,MUTED)
tx(dx+22,dy+116,"11 METHODS, GROUPED",10.5,MUTED,"700",fam=MONO,ls="0.8")
groups=[("Fine-tune","retain FT · grad ascent · Moon"),("Fisher-based","Fisher scrub · SSD"),
        ("Distillation","SCRUB · contrastive latent"),("Adversarial","frozen critics · extra-gradient"),
        ("Training-time","synthetic aug. · RAU"),("Scratch (baseline)","DP-SGD · full retrain")]
gy=dy+140
for g,m in groups:
    tx(dx+22,gy,g,12,INK,"700"); tx(dx+150,gy,m,11.5,MUTED,fam=MONO); gy+=34
# scatter plot
sx,sy,sw,sh=dx+430,dy+150,360,168
E.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="#fbfcfd" stroke="{BORDER}" stroke-width="1"/>')
tx(sx,sy-10,"PRIVACY–UTILITY PLANE (PBMC)",10.5,MUTED,"700",fam=MONO,ls="0.6")
def X(r): return sx+ (r-0.5)/0.5*sw
def Y(a): return sy+sh - a*1.0*sh
# target band advantage <= 0.258
E.append(f'<rect x="{sx}" y="{Y(0.258)}" width="{sw}" height="{sy+sh-Y(0.258)}" fill="{BAND}" opacity="0.55"/>')
# axes ticks
for a in [0.0,0.25,0.5,0.75,1.0]:
    tx(sx-6,Y(a)+4,f"{a:.2f}",9,MUTED,anc="end",fam=MONO); ln(sx,Y(a),sx+3,Y(a),BORDER,1)
for r in [0.5,0.75,1.0]:
    tx(X(r),sy+sh+14,f"{r:.2f}",9,MUTED,anc="middle",fam=MONO)
tx(sx-30,sy+sh/2,"privacy advantage",9.5,MUTED,anc="middle")  # rotated below
E[-1]=E[-1].replace('text-anchor="middle"',f'text-anchor="middle" transform="rotate(-90 {sx-30} {sy+sh/2})"')
tx(sx+sw/2,sy+sh+28,"utility (marker r)",9.5,MUTED,anc="middle")
pts=[("retrain",0.829,0.156,GREEN,7),("Fisher",0.628,0.615,"#c0392b",7),("DP-SGD",0.787,0.045,ORANGE,6),
     ("extra-grad",0.789,0.281,GREY,5),("retain-FT",0.832,0.333,GREY,5),("SSD",0.831,0.435,GREY,5),
     ("SCRUB",0.832,0.411,GREY,5),("contrastive",0.832,0.673,GREY,5),("grad-asc",0.832,0.396,GREY,5),
     ("Moon",0.831,0.480,GREY,5)]
for name,r,a,col,rad in pts:
    E.append(f'<circle cx="{X(r):.1f}" cy="{Y(a):.1f}" r="{rad}" fill="{col}" opacity="0.9"/>')
tx(X(0.628)-10,Y(0.615)-10,"Fisher (Streisand)",9.5,"#c0392b","700",anc="end")
tx(X(0.829)+12,Y(0.156)+4,"retrain",10,GREEN,"700")
E.append('</svg>')
open(OUT,"w").write("\n".join(E))
print("wrote", OUT, len(E), "elements")
