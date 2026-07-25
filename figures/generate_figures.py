"""
SmartMQ Figure Generation — All paper figures.
Run from SmartMQ_Code/ directory:
  python figures/generate_figures.py
Output: figures/output/ directory
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np, math, random, pandas as pd
from collections import deque
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from core import run_smartmq, summarize, generate_signal, PARAMS
from core import run_zhang23, matched_zhang23
from core.smartmq import compute_embedding, compute_trend_memory, cosine_sim, sem_score, clip01, embedding_series

# ─── Output directory ───────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# ─── Style ──────────────────────────────────────────────────────
NAVY='#1B3A6B'; TEAL='#0D7377'; RED='#C0392B'
ORANGE='#D4870A'; GREY='#5D6D7E'; LGREY='#CCD1D1'
SC_COLORS={"flat":NAVY,"step":RED,"periodic":TEAL,"drift":ORANGE}
SC_MARKERS={"flat":"o","step":"s","periodic":"^","drift":"D"}
SC_LABELS=["Flat","Step","Periodic","Drift"]
SCENARIOS=[("flat",1000),("step",1000),("periodic",1000),("drift",500)]

plt.rcParams.update({
    'pdf.fonttype':42,'ps.fonttype':42,
    'font.family':'DejaVu Sans','font.size':23,
    'axes.labelsize':25,'axes.titlesize':23,
    'xtick.labelsize':22,'ytick.labelsize':22,
    'legend.fontsize':23,'axes.facecolor':'white',
    'figure.facecolor':'white','axes.grid':True,
    'grid.alpha':0.20,'grid.linewidth':0.7,'grid.color':LGREY,
    'axes.spines.top':False,'axes.spines.right':False,
    'savefig.dpi':600,'savefig.bbox':'tight','savefig.facecolor':'white',
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'))
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=600)
    plt.close(fig)
    print(f'  ✅ {name}.pdf + .png')

# ─── Baseline helpers ────────────────────────────────────────────
def run_sod(sig, delta=0.30):
    ref=sig[0]; dec=["PUBLISH"]
    for x in sig[1:]:
        if abs(x-ref)>delta: dec.append("PUBLISH"); ref=x
        else: dec.append("SUPPRESS")
    return dec

def run_msod(sig, delta=0.30, max_interval=30):
    ref=sig[0]; last=0; dec=["PUBLISH"]
    for t,x in enumerate(sig[1:],1):
        if abs(x-ref)>delta or (t-last)>=max_interval:
            dec.append("PUBLISH"); ref=x; last=t
        else: dec.append("SUPPRESS")
    return dec

def run_vbt(sig, thr=0.040, window=10, max_interval=30):
    from collections import deque as dq
    buf=dq(maxlen=window); last=0; dec=[]
    for t,x in enumerate(sig):
        buf.append(x)
        lv=float(np.var(list(buf))) if len(buf)>=3 else 0.0
        if t==0 or lv>thr or (t-last)>=max_interval:
            dec.append("PUBLISH"); last=t
        else: dec.append("SUPPRESS")
    return dec

def pr(dec): return sum(1 for d in dec if d=="PUBLISH")/len(dec)

def mae_calc(sig, dec):
    last=sig[0]; s=0.0
    for i,d in enumerate(dec):
        if d=="PUBLISH": last=sig[i]
        s+=abs(sig[i]-last)
    return s/len(dec)

def matched_delta(sig, target):
    lo,hi=0.0,20.0
    for _ in range(60):
        mid=(lo+hi)/2
        if pr(run_sod(sig,mid))>target: lo=mid
        else: hi=mid
    return run_sod(sig,(lo+hi)/2)

def matched_msod(sig, target, max_interval=30):
    lo,hi=0.0,20.0
    for _ in range(60):
        mid=(lo+hi)/2
        if pr(run_msod(sig,mid,max_interval))>target: lo=mid
        else: hi=mid
    return run_msod(sig,(lo+hi)/2,max_interval)

def matched_vbt(sig, target, window=10, max_interval=30):
    lo,hi=0.0,10.0
    for _ in range(60):
        mid=(lo+hi)/2
        if pr(run_vbt(sig,mid,window,max_interval))>target: lo=mid
        else: hi=mid
    return run_vbt(sig,(lo+hi)/2,window,max_interval)

def selectivity_indep(sig, dec, P=PARAMS):
    """Decision-independent semantic selectivity: change vs the previous sample
    (fixed reference, independent of any method's decisions)."""
    prev=None; sm=1.0; Z=[]
    for emb in embedding_series(sig,P):
        if prev is None: sm=1.0
        else:
            raw=1.0-cosine_sim(emb,prev)
            sm=P["SEM_ALPHA"]*sm+(1-P["SEM_ALPHA"])*raw
        Z.append(sm); prev=list(emb)
    pub=[z for z,d in zip(Z,dec) if d=="PUBLISH"]
    sup=[z for z,d in zip(Z,dec) if d=="SUPPRESS"]
    if not pub or not sup: return float('nan')
    return float(np.mean(pub)/(np.mean(sup)+1e-12))

def compute_aoi(dec):
    aoi=[]; last=0
    for t,d in enumerate(dec):
        if d=="PUBLISH": last=t
        aoi.append(t-last)
    return aoi

print("Pre-computing base results...")
base_traces={}; base_metrics={}
for sc,n in SCENARIOS:
    sig=generate_signal(sc); tr=run_smartmq(sig); m=summarize(tr)
    base_traces[sc]=tr; base_metrics[sc]=m
    print(f"  {sc}: pub={m['pub_rate']:.1%}  selectivity={m['selectivity']:.3f}x")

# ═══ FIG 1: Baseline comparison ═════════════════════════════════
print("\n[1/6] figure_3 — Baseline comparison")
N_SEEDS=20; results={}
METHODS=['SmartMQ','SOD','MSOD','VBT','Zhang23']
for sc,n in SCENARIOS:
    sig=generate_signal(sc); m_sm=base_metrics[sc]
    # default-threshold decisions (efficiency panel + natural operating points)
    dec_so=run_sod(sig); dec_ms=run_msod(sig); dec_vb=run_vbt(sig); dec_zh=run_zhang23(sig)
    # matched-rate MAE vs single-purpose baselines
    dec_so_m=matched_delta(sig,m_sm['pub_rate'])
    dec_ms_m=matched_msod(sig,m_sm['pub_rate'])
    # 20-seed decision-independent selectivity, all methods matched to SmartMQ rate
    sel={m:[] for m in METHODS}
    for s in range(N_SEEDS):
        ss=generate_signal(sc,seed=s); tr=run_smartmq(ss,seed=s)
        tgt=summarize(tr)['pub_rate']
        rows={'SmartMQ':[r['decision'] for r in tr],
              'SOD':matched_delta(ss,tgt),'MSOD':matched_msod(ss,tgt),
              'VBT':matched_vbt(ss,tgt),'Zhang23':matched_zhang23(ss,tgt)}
        for m,d in rows.items(): sel[m].append(selectivity_indep(ss,d))
    # selectivity at each method's natural (default) operating point, for panel (d)
    selectivity_def={'SmartMQ':selectivity_indep(sig,[r['decision'] for r in base_traces[sc]]),
              'SOD':selectivity_indep(sig,dec_so),'MSOD':selectivity_indep(sig,dec_ms),
              'VBT':selectivity_indep(sig,dec_vb),'Zhang23':selectivity_indep(sig,dec_zh)}
    results[sc]={
        'pr':{'SmartMQ':m_sm['pub_rate'],'SOD':pr(dec_so),'MSOD':pr(dec_ms),'VBT':pr(dec_vb),'Zhang23':pr(dec_zh)},
        'mae_sm':m_sm['mae'],'mae_sod_m':mae_calc(sig,dec_so_m),'mae_msod_m':mae_calc(sig,dec_ms_m),
        'selectivity_mean':{m:float(np.nanmean(sel[m])) for m in METHODS},
        'selectivity_std':{m:float(np.nanstd(sel[m])) for m in METHODS},
        'selectivity_def':selectivity_def,
    }
    print(f"  {sc}: D_indep " + " ".join(f"{m}={results[sc]['selectivity_mean'][m]:.2f}" for m in METHODS))

fig,axes=plt.subplots(2,2,figsize=(13,9.3))
fig.subplots_adjust(hspace=0.45,wspace=0.35,left=0.08,right=0.97,top=0.88,bottom=0.09)
x=np.arange(4); w=0.15
m_col={'SmartMQ':NAVY,'SOD':RED,'MSOD':ORANGE,'VBT':TEAL,'Zhang23':GREY}

ax=axes[0,0]
for i,(m,col) in enumerate(m_col.items()):
    ax.bar(x+i*w-2*w,[results[sc]['pr'][m]*100 for sc,_ in SCENARIOS],w,label=m,color=col,alpha=0.85,edgecolor='white',lw=0.4)
ax.axhline(y=20,color=LGREY,lw=1.2,ls='--'); ax.set_ylabel('Publish rate (%)')
ax.set_xticks(x); ax.set_xticklabels(SC_LABELS); ax.set_ylim(0,82)
ax.set_title('(a)',loc='left',fontweight='bold')

ax=axes[0,1]
gap_sod=[results[sc]['mae_sm']-results[sc]['mae_sod_m'] for sc,_ in SCENARIOS]
gap_msod=[results[sc]['mae_sm']-results[sc]['mae_msod_m'] for sc,_ in SCENARIOS]
bw=0.32
b1=ax.bar(x-bw/2,[g*100 for g in gap_sod],bw,color=RED,alpha=0.85,edgecolor='white',label='vs SOD (matched)')
b2=ax.bar(x+bw/2,[g*100 for g in gap_msod],bw,color=ORANGE,alpha=0.85,edgecolor='white',label='vs MSOD (matched)')
ax.axhline(y=20.0,color=GREY,lw=1.8,ls='--',label='Noise floor (σ=0.2°C)')
for i,(gs,gm) in enumerate(zip(gap_sod,gap_msod)):
    ax.text(x[i]-bw/2,gs*100+0.4,f'+{gs*100:.1f}',ha='center',va='bottom',rotation=90,fontsize=21,color=RED)
    ax.text(x[i]+bw/2,gm*100+0.4,f'+{gm*100:.1f}',ha='center',va='bottom',rotation=90,fontsize=21,color=ORANGE)
ax.set_ylabel('MAE overhead (×10⁻² °C)'); ax.set_xticks(x); ax.set_xticklabels(SC_LABELS)
ax.legend(fontsize=22,loc='upper left'); ax.set_ylim(0,24)
ax.set_title('(b)',loc='left',fontweight='bold')

ax=axes[1,0]
for i,(m,col) in enumerate(m_col.items()):
    dm=[results[sc]['selectivity_mean'][m] for sc,_ in SCENARIOS]
    ds=[results[sc]['selectivity_std'][m] for sc,_ in SCENARIOS]
    ax.bar(x+i*w-2*w,dm,w,yerr=ds,capsize=2,color=col,alpha=0.85,edgecolor='white',lw=0.4,
           label=m,error_kw=dict(lw=0.8,color=GREY))
ax.axhline(y=1.0,color=LGREY,lw=1.2,ls='--')
ax.set_ylabel(r'Semantic selectivity $\mathcal{Z}$ (×)'); ax.set_xticks(x); ax.set_xticklabels(SC_LABELS)
ax.set_ylim(0,2.5); ax.set_title('(c)',loc='left',fontweight='bold')

ax=axes[1,1]
for sc,_ in SCENARIOS:
    r=results[sc]
    for m,col in m_col.items():
        ax.scatter(r['pr'][m]*100,r['selectivity_def'][m],color=col,s=85,marker=SC_MARKERS[sc],
                   alpha=0.85,zorder=5,edgecolor='white',lw=0.4)
ax.axhline(y=1.0,color=LGREY,lw=1.2,ls='--')
ax.set_xlabel('Publish rate (%)'); ax.set_ylabel(r'Semantic selectivity $\mathcal{Z}$'); ax.set_ylim(0.8,2.4)
leg=[Patch(color=c,label=m) for m,c in m_col.items()]+[
     Line2D([0],[0],marker='o',color='w',markerfacecolor=GREY,ms=8,label='Flat'),
     Line2D([0],[0],marker='s',color='w',markerfacecolor=GREY,ms=8,label='Step'),
     Line2D([0],[0],marker='^',color='w',markerfacecolor=GREY,ms=8,label='Periodic'),
     Line2D([0],[0],marker='D',color='w',markerfacecolor=GREY,ms=8,label='Drift')]
ax.legend(handles=leg[-4:],fontsize=17,ncol=1,loc='upper right',framealpha=0.92,title='Scenario',title_fontsize=16); ax.set_title('(d)',loc='left',fontweight='bold')
_mh=[Patch(color=c,label=m) for m,c in m_col.items()]
fig.legend(_mh,list(m_col.keys()),loc='lower center',ncol=5,bbox_to_anchor=(0.5,1.0),fontsize=21,framealpha=0.9)
save(fig,'figure_3')

# ═══ FIG 2: Sensitivity ═════════════════════════════════════════
print("[2/6] figure_5 — Sensitivity")
SENS=[("W_S","$w_s$ (semantic)"),("W_T","$w_t$ (trend)"),
      ("W_R","$w_r$ (ref dev)"),("W_A","$w_a$ (age)"),
      ("U_HIGH","$\\theta_H$"),("U_LOW","$\\theta_L$"),
      ("W_E","$w_e$ (energy)"),("SEM_K","$k$ (mapping)")]
FACTORS=[0.5,0.75,0.9,1.0,1.1,1.25,1.5]
FLABELS=["-50%","-25%","-10%","base","+10%","+25%","+50%"]

fig,axes=plt.subplots(2,4,figsize=(14,9))
fig.subplots_adjust(hspace=0.55,wspace=0.35,left=0.07,right=0.98,top=0.90,bottom=0.12)
for idx,(pname,plabel) in enumerate(SENS):
    row,col=divmod(idx,4); ax=axes[row,col]; ss=[]
    for sc,n in SCENARIOS:
        sig=generate_signal(sc)
        bdc=selectivity_indep(sig,[r['decision'] for r in run_smartmq(sig)])
        ratios=[selectivity_indep(sig,[r['decision'] for r in run_smartmq(sig,param_overrides={pname:PARAMS[pname]*f})])/(bdc+1e-12) for f in FACTORS]
        ax.plot(range(len(FACTORS)),ratios,color=SC_COLORS[sc],marker=SC_MARKERS[sc],markersize=6,lw=2,label=sc.capitalize())
        ss.append(max(ratios)-min(ratios))
    sl="HIGH" if max(ss)>0.25 else "MED"
    ax.axhline(y=1.0,color=LGREY,lw=1.2,ls='--'); ax.axhline(y=1.1,color=LGREY,lw=0.7,ls=':'); ax.axhline(y=0.9,color=LGREY,lw=0.7,ls=':')
    ax.set_title(plabel,fontweight='bold'); ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(['-50%','','','base','','','+50%'],fontsize=20,rotation=0,ha='center')
    ax.set_ylim(0.50,1.85); ax.set_ylabel(r'$\mathcal{Z}$ / baseline' if col==0 else '')
    sc_=RED if sl=="HIGH" else ORANGE
    ax.text(0.5,0.95,f'Sens: {sl}',transform=ax.transAxes,ha='center',va='top',fontsize=20,color=sc_,fontweight='bold')
_h,_l=axes[0,0].get_legend_handles_labels()
fig.legend(_h,_l,loc='lower center',ncol=4,bbox_to_anchor=(0.5,1.0),fontsize=21,framealpha=0.9)
save(fig,'figure_5')

# ═══ FIG 3: AoI ═════════════════════════════════════════════════
print("[3/6] figure_7 — AoI")
AOI_TH=30
aoi_results={}
for sc,n in SCENARIOS:
    sig=generate_signal(sc)
    dec_sm=[r["decision"] for r in base_traces[sc]]
    dec_so=run_sod(sig); dec_ms=run_msod(sig,max_interval=AOI_TH); dec_vb=run_vbt(sig,max_interval=AOI_TH)
    aoi_results[sc]={
        'SmartMQ':{'mean':np.mean(compute_aoi(dec_sm)),'peak':max(compute_aoi(dec_sm)),'viol':sum(1 for a in compute_aoi(dec_sm) if a>AOI_TH)/n},
        'SOD':    {'mean':np.mean(compute_aoi(dec_so)),'peak':max(compute_aoi(dec_so)),'viol':sum(1 for a in compute_aoi(dec_so) if a>AOI_TH)/n},
        'MSOD':   {'mean':np.mean(compute_aoi(dec_ms)),'peak':max(compute_aoi(dec_ms)),'viol':sum(1 for a in compute_aoi(dec_ms) if a>AOI_TH)/n},
        'VBT':    {'mean':np.mean(compute_aoi(dec_vb)),'peak':max(compute_aoi(dec_vb)),'viol':sum(1 for a in compute_aoi(dec_vb) if a>AOI_TH)/n},
    }
fig,axes=plt.subplots(1,3,figsize=(14,5.7))
fig.subplots_adjust(wspace=0.38,left=0.08,right=0.97,top=0.84,bottom=0.15)
methods=['SmartMQ','SOD','MSOD','VBT']; m_col2={'SmartMQ':NAVY,'SOD':RED,'MSOD':ORANGE,'VBT':TEAL}
x=np.arange(4); bw=0.18
for pidx,(metric,ylabel,title) in enumerate([('mean','Mean AoI (samples)','(a) Mean AoI'),
                                               ('peak','Peak AoI (samples)','(b) Peak AoI'),
                                               ('viol','Violation rate (%)','(c) AoI violation rate')]):
    ax=axes[pidx]
    for i,(m,col) in enumerate(m_col2.items()):
        vals=[aoi_results[sc][m][metric]*(100 if metric=='viol' else 1) for sc,_ in SCENARIOS]
        bars=ax.bar(x+i*bw-1.5*bw,vals,bw,color=col,alpha=0.85,edgecolor='white',lw=0.4,label=m)
        if metric=='mean':
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,f'{v:.1f}',ha='center',va='bottom',rotation=90,fontsize=17,color='#2C3E50')
    if metric in ['mean','peak']: ax.axhline(y=AOI_TH,color=GREY,lw=1.5,ls='--')
    ax.set_ylabel(ylabel); ax.set_xticks(x); ax.set_xticklabels(SC_LABELS); ax.tick_params(axis='x',labelsize=17); ax.set_title(title,loc='left',fontweight='bold')
_h6,_l6=axes[0].get_legend_handles_labels()
fig.legend(_h6,_l6,loc='lower center',ncol=4,bbox_to_anchor=(0.5,1.0),fontsize=21,framealpha=0.9)
save(fig,'figure_7')

# ═══ FIG 4: Composite signal ════════════════════════════════════
print("[4/6] figure_4 — Composite signal")
random.seed(42); np.random.seed(42)
phases_def=[("Flat",200,lambda t:20.0+random.gauss(0,0.2)),
            ("Step",150,lambda t:21.0+random.gauss(0,0.2)),
            ("Periodic",200,lambda t:20.5+0.5*math.sin(2*math.pi*t/40)+random.gauss(0,0.2)),
            ("Drift",250,lambda t:20.0+t*0.012+random.gauss(0,0.2))]
composite=[]; boundaries=[]; cum=0
for nm,n,gfn in phases_def:
    boundaries.append((cum,cum+n,nm)); [composite.append(gfn(i)) for i in range(n)]; cum+=n
tr_c=run_smartmq(composite)
dec_sm_c=[r["decision"] for r in tr_c]; sems_c=[r["smoothed_sem"] for r in tr_c]; utils_c=[r["U"] for r in tr_c]
dec_so_c=run_sod(composite); dec_vb_c=run_vbt(composite)
phase_col={'Flat':'#EBF5FB','Step':'#FEF9E7','Periodic':'#EAFAF1','Drift':'#FDF2F8'}
t_arr_c=np.arange(len(composite))

fig,axes=plt.subplots(3,1,figsize=(14,11.3),gridspec_kw={'height_ratios':[3,1.2,1.2],'hspace':0.62})
fig.subplots_adjust(left=0.08,right=0.97,top=0.93,bottom=0.07)
ax=axes[0]
for start,end,nm in boundaries:
    ax.axvspan(start,end,alpha=0.14,color=phase_col[nm],zorder=0)
    ax.text((start+end)/2,max(composite)+0.06,nm,ha='center',va='bottom',fontsize=21,color=GREY,style='italic')
    if start>0: ax.axvline(x=start,color=LGREY,lw=1,ls='--',alpha=0.8)
ax.plot(t_arr_c,composite,color=TEAL,lw=1.2,label='Sensor signal',zorder=2)
pub_sm=[t for t,d in enumerate(dec_sm_c) if d=="PUBLISH"]
ax.scatter(pub_sm,[composite[t] for t in pub_sm],color=RED,s=20,zorder=5,label=f'SmartMQ ({pr(dec_sm_c):.0%} pub)',alpha=0.9)
pub_so=[t for t,d in enumerate(dec_so_c) if d=="PUBLISH"]
ax.scatter(pub_so,[composite[t] for t in pub_so],color=NAVY,s=10,marker='^',alpha=0.35,zorder=4,label=f'SOD ({pr(dec_so_c):.0%} pub)')
ax.set_ylabel('Temperature (°C)'); ax.set_ylim(19.3,23.7); ax.legend(fontsize=20,framealpha=0.9,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3); ax.set_title('(a)',loc='left',fontweight='bold')

def rolling(dec,w):
    return np.convolve(np.array([1 if d=="PUBLISH" else 0 for d in dec],dtype=float),np.ones(w)/w,mode='same')
ax2=axes[1]
for start,end,nm in boundaries: ax2.axvspan(start,end,alpha=0.10,color=phase_col[nm])
ax2.plot(t_arr_c,rolling(dec_sm_c,40),color=RED,lw=2,label='SmartMQ')
ax2.plot(t_arr_c,rolling(dec_so_c,40),color=NAVY,lw=1.8,ls='--',alpha=0.8,label='SOD')
ax2.plot(t_arr_c,rolling(dec_vb_c,40),color=TEAL,lw=1.8,ls=':',alpha=0.8,label='VBT')
ax2.set_ylabel('Rolling publish rate'); ax2.set_ylim(-0.03,0.92); ax2.legend(fontsize=20,ncol=3,framealpha=0.9,loc='lower center',bbox_to_anchor=(0.5,1.0))
ax2.set_title('(b)',loc='left',fontweight='bold')

ax3=axes[2]
for start,end,nm in boundaries: ax3.axvspan(start,end,alpha=0.10,color=phase_col[nm])
ax3.plot(t_arr_c,utils_c,color=NAVY,lw=1.8,label='Utility $U_t$',alpha=0.9)
ax3.axhline(y=0.40,color=RED,lw=1.5,ls='--',alpha=0.9,label='$\\theta_H=0.40$')
ax3.axhline(y=0.24,color=TEAL,lw=1.5,ls='--',alpha=0.9,label='$\\theta_L=0.24$')
ax3.set_ylabel('Utility $U_t$'); ax3.set_xlabel('Sample index')
ax3.set_ylim(0.06,0.80); ax3.legend(fontsize=20,ncol=3,framealpha=0.9,loc='lower center',bbox_to_anchor=(0.5,1.0)); ax3.set_title('(c)',loc='left',fontweight='bold')
save(fig,'figure_4')

# ═══ FIG 5: Hardware ════════════════════════════════════════════
print("[5/6] figure_8 — Hardware")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sht30_temperature_humidity.csv")
df=pd.read_csv(DATA_PATH)
temp=df['temperature'].values; hum=df['humidity'].values; t_hw=np.arange(len(df))
tr_T=run_smartmq(temp); m_T=summarize(tr_T)
tr_H=run_smartmq(hum);  m_H=summarize(tr_H)
dec_T=[r["decision"] for r in tr_T]; dec_H=[r["decision"] for r in tr_H]
selectivity_T=selectivity_indep(temp,dec_T); selectivity_H=selectivity_indep(hum,dec_H)

fig,axes=plt.subplots(3,1,figsize=(14,11.3),gridspec_kw={'height_ratios':[2.5,2.5,1.2],'hspace':0.62})
fig.subplots_adjust(left=0.09,right=0.97,top=0.93,bottom=0.07)
ax1=axes[0]
ax1.plot(t_hw,temp,color=TEAL,lw=1.1,label='Temperature signal',zorder=2)
pub_T=[t for t,d in enumerate(dec_T) if d=="PUBLISH"]
ax1.scatter(pub_T,[temp[t] for t in pub_T],color=RED,s=18,zorder=5,alpha=0.9,
    label=f'Published ({m_T["pub_rate"]:.1%} pub rate, selectivity = {selectivity_T:.2f}x)')
ax1.set_ylabel('Temperature (°C)'); ax1.set_ylim(21.4,31); ax1.legend(fontsize=20,framealpha=0.9,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2); ax1.text(0.012,0.94,'(a)',transform=ax1.transAxes,fontweight='bold',fontsize=23,va='top')

ax2=axes[1]
ax2.plot(t_hw,hum,color=NAVY,lw=1.1,label='Humidity signal',zorder=2)
pub_H=[t for t,d in enumerate(dec_H) if d=="PUBLISH"]
ax2.scatter(pub_H,[hum[t] for t in pub_H],color=ORANGE,s=18,zorder=5,alpha=0.9,
    label=f'Published ({m_H["pub_rate"]:.1%} pub rate, selectivity = {selectivity_H:.2f}x), same parameters')
ax2.set_ylabel('Humidity (%)'); ax2.set_ylim(37,99); ax2.legend(fontsize=20,framealpha=0.9,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2); ax2.text(0.012,0.94,'(b)',transform=ax2.transAxes,fontweight='bold',fontsize=23,va='top')

ax3=axes[2]
cats=['Temperature\n(SmartMQ)','Humidity\n(SmartMQ, same parameters)']
vals=[selectivity_T,selectivity_H]
bars=ax3.bar(range(2),vals,color=[TEAL,NAVY],alpha=0.85,width=0.4,edgecolor='white')
ax3.axhline(y=1.0,color=LGREY,lw=1.5,ls='--')
for bar,v in zip(bars,vals):
    ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,f'{v:.2f}x',
            ha='center',va='bottom',fontsize=22,fontweight='bold',color='#2C3E50')
ax3.set_ylabel(r'Selectivity $\mathcal{Z}$',labelpad=8); ax3.set_xticks(range(2)); ax3.set_xticklabels(cats,fontsize=21)
ax3.set_ylim(0,max(vals)*1.28)
ax3.set_title('(c) Sensor-agnostic validation, identical parameters across sensor types',loc='left',fontweight='bold',fontsize=22)
save(fig,'figure_8')

# ═══ FIG 6: Window sensitivity ══════════════════════════════════
print("[6/6] figure_6 — Window sensitivity (main text, Fig. 6)")
WPARAMS=[("SLOPE_LONG_W","Trend window $W_{\\mathrm{tr}}$",[5,10,15,20,25,30,35]),
         ("AUTOCORR_W",  "Autocorr window $W_{\\mathrm{ac}}$",[3,5,8,10,12,15,20]),
         ("VAR_W",       "Volatility window $W_{\\mathrm{var}}$",[3,5,8,10,12,15,20])]
fig,axes=plt.subplots(1,3,figsize=(14,5.7))
fig.subplots_adjust(wspace=0.30,left=0.075,right=0.985,top=0.82,bottom=0.17)
_sc_handles=None
for idx,(pname,plabel,wvals) in enumerate(WPARAMS):
    ax=axes[idx]
    for sc,n in SCENARIOS:
        sig=generate_signal(sc); selectivities=[selectivity_indep(sig,[r['decision'] for r in run_smartmq(sig,param_overrides={pname:wv})]) for wv in wvals]
        ax.plot(wvals,selectivities,color=SC_COLORS[sc],marker=SC_MARKERS[sc],markersize=7,lw=2,label=sc.capitalize())
    if _sc_handles is None: _sc_handles=ax.get_legend_handles_labels()
    dline=ax.axvline(x=PARAMS[pname],color=GREY,lw=1.5,ls='--',alpha=0.85)
    ax.axhline(y=1.5,color=LGREY,lw=1,ls=':',alpha=0.8)
    ax.set_xlabel(plabel,fontsize=20); ax.set_ylabel(r'Semantic selectivity $\mathcal{Z}$ (×)' if idx==0 else ''); ax.set_ylim(0.5,4.3)
    ax.set_title(f'({chr(97+idx)})',loc='left',fontweight='bold')
    ax.legend([dline],[f'Default={PARAMS[pname]}'],fontsize=19,framealpha=0.9,loc='upper left')
fig.legend(_sc_handles[0][:4],_sc_handles[1][:4],loc='lower center',ncol=4,bbox_to_anchor=(0.5,1.0),fontsize=20,framealpha=0.9)
save(fig,'figure_6')

print(f"\n✅ All 6 figures saved to: figures/output")
