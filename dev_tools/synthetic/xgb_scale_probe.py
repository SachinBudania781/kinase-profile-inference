import sys, dataclasses, numpy as np
sys.path.insert(0,'/home/claude/kpi')
import core_models as cm, data_pipeline as dp
from xgboost import XGBRegressor
bank='/home/claude/dev_work/synth/synth_data'
reg=dp.load_registry(bank+'/registry.json')
P=dp.load_panel(reg,'Syn0017_D4_n60_mat',{'synth':bank+'/data'},allow_lockbox=False)
cfg=cm.load_config(); ep=dp.build_episodes(P,2)
X=cm.xgb_features(P,ep); truth=P.y[ep.i,ep.q]; folds=P.folds[ep.i]
tr=np.flatnonzero(folds!=0); te=np.flatnonzero(folds==0)
def fit_eval(scale, **kw):
    y=truth*scale; Xs=cm.xgb_features(dataclasses.replace(P,y=P.y*scale),ep)
    params=dict(objective='reg:absoluteerror',max_depth=3,n_estimators=200,learning_rate=0.055,min_child_weight=15,reg_lambda=4.0,n_jobs=1,random_state=1,tree_method='hist'); params.update(kw)
    m=XGBRegressor(**params).fit(Xs[tr],y[tr])
    return float(np.mean(np.abs(m.predict(Xs[te])-y[te]))/scale)
print('scale   MAE(rescaled)   default | base_score=median | y standardised')
for sc in [1000,10,1,0.3,0.1,0.03,0.01,0.001]:
    d=fit_eval(sc)
    bs=fit_eval(sc, base_score=float(np.median(truth*sc)))
    y=truth*sc; mu,sd=np.median(y[tr]),y[tr].std()
    Xs=cm.xgb_features(dataclasses.replace(P,y=P.y*sc),ep)
    m=XGBRegressor(objective='reg:absoluteerror',max_depth=3,n_estimators=200,learning_rate=0.055,min_child_weight=15,reg_lambda=4.0,n_jobs=1,random_state=1,tree_method='hist').fit(Xs[tr],(y[tr]-mu)/sd)
    st=float(np.mean(np.abs(m.predict(Xs[te])*sd+mu-y[te]))/sc)
    print(f'{sc:<8g}{d:10.4f}      {bs:10.4f}        {st:10.4f}')
