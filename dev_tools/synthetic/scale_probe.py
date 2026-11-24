import sys, dataclasses, numpy as np
sys.path.insert(0,'/home/claude/kpi')
import core_models as cm, data_pipeline as dp
bank='/home/claude/dev_work/synth/synth_data'
name=sys.argv[1]; k=2
reg=dp.load_registry(bank+'/registry.json')
P=dp.load_panel(reg,name,{'synth':bank+'/data'},allow_lockbox=False)
cfg=cm.load_config()
ep=dp.build_episodes(P,k)
base=cm.run_panel(P,k,cfg,('XGBoost','SER','CovarianceLocalMedian','QueryMedian','RidgeSupports'),pool=None,ep=ep)
P2=dataclasses.replace(P,y=P.y*0.001); cm.clear_caches()
small=cm.run_panel(P2,k,cfg,('XGBoost','SER','CovarianceLocalMedian','QueryMedian','RidgeSupports'),pool=None,ep=ep)
truth=P.y[ep.i,ep.q]
print(name,'overall SD',float(np.nanstd(P.y)),'per-drug SD range',float(np.nanmin(np.nanstd(P.y,0))),float(np.nanmax(np.nanstd(P.y,0))))
for m in base.preds:
    a=base.preds[m]; b=small.preds[m]*1000
    d=np.abs(a-b); rel=d/ (np.abs(truth)+1e-9)
    print(f"{m:22s} max|diff|={np.nanmax(d):.4g}  frac episodes differing >1e-6: {np.nanmean(d>1e-6):.3f}  MAE x1={np.nanmean(np.abs(a-truth)):.4f}  MAE x0.001(rescaled)={np.nanmean(np.abs(b-truth)):.4f}")
print('gate x1   :', [(d['expert'],d['weight']) for d in base.decisions['SER_gate']])
print('gate x1e-3:', [(d['expert'],d['weight']) for d in small.decisions['SER_gate']])
print('xgb rows  x1   :', [r['xgb_row'] for r in base.decisions['baseline_selection']])
print('xgb rows  x1e-3:', [r['xgb_row'] for r in small.decisions['baseline_selection']])
