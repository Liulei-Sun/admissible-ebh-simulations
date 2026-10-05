import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "closed_eBH_asymmetric_weights_variance010_R1000"
OUT = REPO / "reproductions" / "asymmetric_representative"
sys.path.insert(0, str(ROOT / "code"))
from compare_reproduction import SCIENTIFIC_COLUMNS

OUT.mkdir(parents=True, exist_ok=False)
records = []
started_all = time.perf_counter()
for batch in ("batch1_seed20260803", "batch2_seed20260804"):
    for n in (20, 30):
        for condition in ("clean", "signed_square", "persistent_noise"):
            published = ROOT / "data" / batch / f"n{n}" / condition
            for method, script, result in (
                ("procedure1", "optimize_method1_learned.py", "method1_results.csv"),
                ("procedure2", "run_learned_asymmetric_closed_ebh_2.py", "original_formula_results.csv")):
                stage = OUT / batch / f"n{n}" / condition / method
                stage.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-B", str(ROOT / "code" / "solver" / script),
                    "--input", str(published / "learned_instances.csv"), "--output-dir", str(stage),
                    "--alpha", "0.05", "--replication-start", "1", "--replication-end", "2",
                    "--workers", "2", "--separator-time-limit", "300", "--separator-threads", "1",
                    "--master-time-limit", "300", "--layer-batch", "100", "--fixed-batch", "100",
                    "--layer-window", "10", "--checkpoint-every", "1"]
                begin = time.perf_counter()
                with (stage / "execution.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=ROOT, check=False)
                record = dict(batch=batch, nonnulls=n, condition=condition, procedure=method,
                              replications=[1, 2], elapsed_seconds=time.perf_counter()-begin,
                              returncode=completed.returncode)
                if completed.returncode == 0:
                    fresh = pd.read_csv(stage / result, keep_default_na=False).sort_values("replication")
                    old = pd.read_csv(published / f"{method}_results.csv", keep_default_na=False)
                    old = old.loc[old.replication.isin([1,2])].sort_values("replication")
                    try:
                        pd.testing.assert_frame_equal(fresh[SCIENTIFIC_COLUMNS].reset_index(drop=True),
                            old[SCIENTIFIC_COLUMNS].reset_index(drop=True), check_dtype=False,
                            check_exact=False, rtol=2e-12, atol=2e-10)
                        record["comparison"] = "all scientific fields identical within tolerance"
                    except AssertionError as error:
                        record["comparison"] = "MISMATCH"
                        record["error"] = str(error)
                records.append(record)
                print(json.dumps(record), flush=True)
                (OUT / "replay_audit.json").write_text(json.dumps(dict(stages=records,
                    elapsed_seconds=time.perf_counter()-started_all), indent=2)+"\n", encoding="utf-8")


if len(records) != 24 or any(r["returncode"] != 0 or r.get("comparison") != "all scientific fields identical within tolerance" for r in records):
    raise RuntimeError("Representative replay failed; inspect replay_audit.json and stage logs.")
print("PASS: 48 fresh optimizations across all 24 stages agree with the publication.")
