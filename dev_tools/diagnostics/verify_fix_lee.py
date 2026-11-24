import sys, time, numpy as np
sys.path.insert(0, '/home/claude/dev_work/kpi_patched')
import data_pipeline as dp, core_models as cm
DATA = '/mnt/user-data/uploads/DH307 Last Final/SER_Code_and_All_Cleaned_Study_Data/SER/data'
reg = dp.load_registry('/home/claude/kpi/configs/panels.json')
for name in ['Lee2026_EGFR']:
    P = dp.load_panel(reg, name, {'archive': DATA}, allow_lockbox=False)
    cfg = cm.load_config(); t=time.time()
    r = cm.run_panel(P, 2, cfg, methods=('SAER-7g','AlignedAddAudit','NestedAligned'), pool=None)
    for m in ('SAER-7g','AlignedAddAudit','NestedAligned'):
        print(name, m, 'predicted', int(np.isfinite(r.preds[m]).sum()), 'failed', len(r.failures[m]), flush=True)
    print('seconds', round(time.time()-t))
