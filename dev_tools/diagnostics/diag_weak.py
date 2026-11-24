"""Development-only diagnosis of SER vs XGBoost on dev panels (no lockbox access)."""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/kpi')
import data_pipeline as dp
DATA = '/mnt/user-data/uploads/DH307 Last Final/SER_Code_and_All_Cleaned_Study_Data/SER/data'
RUNS = '/mnt/user-data/uploads/DH307 Last Final/kinase-profile-inference/runs/dev'
reg = dp.load_registry('/home/claude/kpi/configs/panels.json')
roots = {'archive': DATA}
jobs = [('Jung2025_ABL1',2),('Jung2025_ABL1',3),('Chakraborty2024_SRC',2),('Chakraborty2024_SRC',3),
        ('Oh2026_ALK',2),('Lee2026_EGFR',2),('Robichaux2021_EGFR',2),('Estevam2025_MET',2)]
out = {}
panels = {}
for name,k in jobs:
    if name not in panels:
        panels[name] = dp.load_panel(reg, name, roots, allow_lockbox=False)
    P = panels[name]
    ep = dp.build_episodes(P, k)
    df = pd.read_csv(f'{RUNS}/{name}_k{k}_predictions.csv.gz', usecols=['episode_id','fold','XGBoost','SER','CovarianceLocalMedian','RidgeSupports','QueryMedian'])
    df = df.set_index('episode_id').loc[ep.ids]
    truth = P.y[ep.i, ep.q]
    res = {m: np.abs(df[m].to_numpy()-truth) for m in ['XGBoost','SER','CovarianceLocalMedian','RidgeSupports','QueryMedian']}
    mae = {m: float(np.nanmean(v)) for m,v in res.items()}
    # blend curve: anchor + w*(alt-anchor)
    xg = df['XGBoost'].to_numpy()
    blends = {}
    for alt in ['CovarianceLocalMedian','RidgeSupports']:
        a = df[alt].to_numpy(); a = np.where(np.isfinite(a), a, xg)
        blends[alt] = {w: float(np.mean(np.abs(xg + w*(a-xg) - truth))) for w in [0,0.25,0.5,0.75,1.0]}
    # sizes
    ser = df['SER'].to_numpy()
    move = float(np.mean(np.abs(ser-xg)))
    # per-query-drug gain
    q = ep.q
    perq = {P.drugs[d]: (float(np.mean(res['XGBoost'][q==d])), float(np.mean(res['SER'][q==d]))) for d in np.unique(q)}
    # data structure: drug correlations, SD, query-median skill
    Y = P.y
    with np.errstate(all='ignore'):
        sd = np.nanstd(Y, axis=0)
        C = pd.DataFrame(Y).corr(min_periods=30).to_numpy()
    off = C[~np.eye(len(C), dtype=bool)]
    res_info = {'n_var': P.n, 'n_drugs': P.n_drugs, 'n_sites': P.n_sites,
                'censored_frac': float((P.censor!=0).mean()), 'drug_sd': [round(float(s),3) for s in sd],
                'mean_offdiag_corr': float(np.nanmean(off)), 'min_corr': float(np.nanmin(off)),
                'mae': {m: round(v,4) for m,v in mae.items()},
                'skill_xgb_vs_qmed_%': round(100*(1-mae['XGBoost']/mae['QueryMedian']),1),
                'ser_gain_%': round(100*(1-mae['SER']/mae['XGBoost']),2),
                'ridge_gain_%': round(100*(1-mae['RidgeSupports']/mae['XGBoost']),2),
                'local_gain_%': round(100*(1-mae['CovarianceLocalMedian']/mae['XGBoost']),2),
                'mean_abs_ser_minus_xgb': round(move,4), 'ser_move_over_xgb_mae_%': round(100*move/mae['XGBoost'],1),
                'blend_local': {str(w): round(v,4) for w,v in blends['CovarianceLocalMedian'].items()},
                'blend_ridge': {str(w): round(v,4) for w,v in blends['RidgeSupports'].items()},
                'per_query_drug_xgb_ser': {d:(round(a,3),round(b,3)) for d,(a,b) in perq.items()}}
    # noise floor from replicates (Jung only)
    if name=='Jung2025_ABL1':
        z = np.load(f'{DATA}/{name}/response_matrix.npz', allow_pickle=True)
        R = z['replicates']
        with np.errstate(all='ignore'):
            d = np.abs(R[:,:,0]-R[:,:,1])
        res_info['replicate_mean_abs_diff'] = float(np.nanmean(d))
        res_info['replicate_note'] = 'mean |rep1-rep2|; a prediction cannot beat ~half-to-equal of this on average'
    out[f'{name}_k{k}'] = res_info
    print(f'{name} k{k}', json.dumps(res_info)[:2600], flush=True)
json.dump(out, open('/home/claude/dev_work/diag_weak.json','w'), indent=1)
