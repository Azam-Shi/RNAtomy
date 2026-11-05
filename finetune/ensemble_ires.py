"""Soft-vote ensemble of finetune_ires.py runs: averages the members' logits (and probabilities).

    python -m finetune.ensemble_ires run_over run_under1 run_under2 run_under3 --out ensemble.json

Each run directory needs val_predictions.csv and test_predictions.csv; all members must share the same
validation split. The tuned threshold is the Youden-J optimum on the ensemble's validation predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.heads.ires_classification import binary_metrics, youden_threshold

PAPER = {"structRFM": dict(AUPRC=0.694, AUROC=0.713, F1=0.701),
         "structRFM-": dict(AUPRC=0.642, AUROC=0.684),
         "IRESfinder": dict(F1=0.475), "DeepIRES": dict(F1=0.284)}


def load(dirs, name):
    dfs = [pd.read_csv(Path(d) / name).set_index("id") for d in dirs]
    ids = dfs[0].index
    for d, df in zip(dirs, dfs):
        if len(df) != len(ids) or not df.index.equals(ids):
            raise ValueError(f"{d}/{name}: ids differ from the first member "
                             f"(members must use the same --split-seed and test set)")
        if not (df["label"].values == dfs[0]["label"].values).all():
            raise ValueError(f"{d}/{name}: labels differ from the first member")
    y = dfs[0]["label"].values
    return (np.stack([df["logit"].values for df in dfs]),
            np.stack([df["prob"].values for df in dfs]), y)


def fmt(m):
    return (f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} f1={m['f1']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} acc={m['accuracy']:.4f} mcc={m['mcc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    vL, vP, vy = load(a.members, "val_predictions.csv")
    tL, tP, ty = load(a.members, "test_predictions.csv")
    res = {"members": a.members, "individual_test": {}}
    for d, lg in zip(a.members, tL):
        res["individual_test"][d] = binary_metrics(lg, ty)
        print(f"[member] {d}: {fmt(res['individual_test'][d])}")

    ens_t, ens_v = tL.mean(0), vL.mean(0)
    thr = youden_threshold(ens_v, vy)
    res["ensemble_logit_mean_test"] = binary_metrics(ens_t, ty)
    res["ensemble_logit_mean_test_at_val_youden"] = binary_metrics(ens_t, ty, threshold=thr)
    p = tP.mean(0)
    res["ensemble_prob_mean_test"] = binary_metrics(np.log(np.clip(p, 1e-9, 1 - 1e-9) /
                                                           np.clip(1 - p, 1e-9, 1)), ty)
    res["paper"] = PAPER
    print(f"[ENSEMBLE] mean-logit  @0.5          : {fmt(res['ensemble_logit_mean_test'])}")
    print(f"[ENSEMBLE] mean-logit  @val-Youden {thr:.3f}: {fmt(res['ensemble_logit_mean_test_at_val_youden'])}")
    print(f"[ENSEMBLE] mean-prob   @0.5          : {fmt(res['ensemble_prob_mean_test'])}")
    print("[PAPER]    structRFM  AUPRC 0.694  AUROC 0.713  F1 0.701 | structRFM- AUPRC 0.642 AUROC 0.684 "
          "| IRESfinder F1 0.475 | DeepIRES F1 0.284")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"[ENSEMBLE] -> {a.out}")


if __name__ == "__main__":
    main()
