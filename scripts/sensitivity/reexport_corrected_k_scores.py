import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/root/autodl-tmp/HCR_input/all_patients_ith_results_3.csv")
TR = Path("/root/autodl-tmp/HCR_input/ITH3_TRAIN.csv")
VA = Path("/root/autodl-tmp/HCR_input/ITH3_TEST.csv")
OUT = Path("/root/autodl-tmp/HCR_input/all_patients_ith_K2_K10_corrected.csv")
AUDIT = Path("/root/autodl-tmp/HCR_runs/k_sensitivity_20260819/K3_replacement_audit.csv")


def norm_id(x):
    s = str(x).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("c"):
        s = s[1:]
    return s.lstrip("0")


new = pd.read_csv(SRC, encoding="gb18030")
formal = pd.concat(
    [pd.read_csv(TR).assign(formal_source="ITH3_TRAIN.csv"),
     pd.read_csv(VA).assign(formal_source="ITH3_TEST.csv")],
    ignore_index=True,
).rename(columns={"id": "formal_id", "ith_3clusters": "formal_ith_3clusters"})

new["id_norm"] = new.patient_id.map(norm_id)
formal["id_norm"] = formal.formal_id.map(norm_id)
if new.id_norm.duplicated().any() or formal.id_norm.duplicated().any():
    raise RuntimeError("Duplicate normalized patient IDs detected")

merged = new.merge(formal[["id_norm", "formal_id", "formal_ith_3clusters", "formal_source"]], on="id_norm", how="left", validate="one_to_one")
if len(merged) != 151 or merged.formal_ith_3clusters.isna().any():
    raise RuntimeError(f"Expected 151 complete matches; rows={len(merged)}, unmatched={merged.formal_ith_3clusters.isna().sum()}")

merged["previous_ith_3clusters"] = merged["ith_3clusters"]
merged["replacement_difference"] = merged.formal_ith_3clusters - merged.previous_ith_3clusters
merged["ith_3clusters"] = merged.formal_ith_3clusters

audit = merged[["patient_id", "id_norm", "dataset", "previous_ith_3clusters", "ith_3clusters", "replacement_difference", "formal_source", "formal_id"]].copy()
audit["exact_before_replacement"] = np.isclose(audit.previous_ith_3clusters, audit.ith_3clusters, atol=1e-12, rtol=0)
AUDIT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(AUDIT, index=False)

drop = ["id_norm", "formal_id", "formal_ith_3clusters", "formal_source", "previous_ith_3clusters", "replacement_difference"]
corrected = merged.drop(columns=drop)
corrected.to_csv(OUT, index=False, encoding="gb18030")

check = pd.read_csv(OUT, encoding="gb18030")
check["id_norm"] = check.patient_id.map(norm_id)
verify = check.merge(formal[["id_norm", "formal_ith_3clusters"]], on="id_norm", validate="one_to_one")
if not np.allclose(verify.ith_3clusters, verify.formal_ith_3clusters, atol=1e-12, rtol=0):
    raise RuntimeError("Post-write K=3 verification failed")

print(f"OUTPUT={OUT}")
print(f"N={len(check)}; exact_matches_after=151/151")
print(f"replaced_nonidentical={int((~audit.exact_before_replacement).sum())}; already_identical={int(audit.exact_before_replacement.sum())}")
print(f"MD5={hashlib.md5(OUT.read_bytes()).hexdigest()}")
