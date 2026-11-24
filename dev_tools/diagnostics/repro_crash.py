"""Reproduce the D=3 crash on a dev panel (Lee, 3 drugs) with the ORIGINAL code, then with the patched copy."""
import sys, traceback, numpy as np
which = sys.argv[1]            # 'orig' or 'patched'
sys.path.insert(0, '/home/claude/kpi' if which == 'orig' else '/home/claude/dev_work/kpi_patched')
import data_pipeline as dp, core_models as cm
DATA = '/mnt/user-data/uploads/DH307 Last Final/SER_Code_and_All_Cleaned_Study_Data/SER/data'
reg = dp.load_registry('/home/claude/kpi/configs/panels.json')
P = dp.load_panel(reg, 'Lee2026_EGFR', {'archive': DATA}, allow_lockbox=False)
cfg = cm.load_config()
bank = cm.ExpertBank(P, np.arange(P.n), cfg)
print(which, 'D =', bank.D, 'use_joint =', bank.use_joint, flush=True)
try:
    bank._joint_tree()
    print(which, 'joint tree built OK', flush=True)
except Exception:
    print(which, 'joint tree FAILED:'); traceback.print_exc(limit=3)
