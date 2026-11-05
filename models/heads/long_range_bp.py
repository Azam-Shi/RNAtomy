"""Long-range base-pair head: a ContactMapHead trained on the base-pair matrix parsed from dot-bracket structures.

Loss and metrics can be restricted to pairs with |i - j| >= min_range. Metrics are per-RNA F1 / precision / recall
for long-range pairs (lr_*), all pairs (all_*) and pseudoknot pairs (pk_*), averaged over RNAs with at least one
true pair in the subset. The decision threshold is tuned on the validation split (find_best_threshold).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.heads.contact_map import ContactMapHead


def _crossing_pair_set(pairs: List[Tuple[int, int]]) -> set:
    out = set()
    for a in range(len(pairs)):
        i, j = pairs[a]
        for b in range(len(pairs)):
            if a == b:
                continue
            k, l = pairs[b]
            if (i < k < j < l) or (k < i < l < j):
                out.add((i, j))
                break
    return out


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else (
        0.0 if (tp + fp + fn) > 0 else float("nan"))
    return p, r, f


class LongRangeBPHead(ContactMapHead):
    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        conv_kernel: int = 3,
        input_repr: str = "pair",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        min_seq_sep: int = 3,
        min_range: int = 101,
        loss_range: str = "all",
        eval_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim, proj_dim=proj_dim, hidden_dim=hidden_dim,
            n_layers=n_layers, conv_kernel=conv_kernel, input_repr=input_repr,
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            min_seq_sep=min_seq_sep, long_range_sep=min_range,
        )
        if loss_range not in ("all", "long"):
            raise ValueError("loss_range must be all | long")
        self.min_range = int(min_range)
        self.loss_range = loss_range
        self.eval_threshold = float(eval_threshold)

    def compute_loss(self, predictions, targets, seq_len=None, pair_valid_mask=None, **kwargs):
        _, L, _ = predictions.shape
        idx = torch.arange(L, device=predictions.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        tri = idx.unsqueeze(1) < idx.unsqueeze(0)
        m = (sep >= self.min_seq_sep) & tri
        if self.loss_range == "long":
            m = m & (sep >= self.min_range)
        m = m.unsqueeze(0)
        if pair_valid_mask is not None:
            vm = pair_valid_mask.to(torch.bool, predictions.device)
            m = m & vm & vm.transpose(-2, -1)
        mf = m.to(predictions.dtype)
        per = self._focal_loss(predictions, targets.to(predictions.dtype),
                               alpha=self.focal_alpha, gamma=self.focal_gamma)
        return (per * mf).sum() / mf.sum().clamp(min=1.0)

    @staticmethod
    def decode_matching(prob: np.ndarray, threshold: float) -> np.ndarray:
        """Greedy one-pair-per-nucleotide matching on the upper triangle."""
        L = prob.shape[0]
        iu, ju = np.triu_indices(L, k=1)
        p = prob[iu, ju]
        keep = p >= threshold
        order = np.argsort(-p[keep])
        ii, jj = iu[keep][order], ju[keep][order]
        used = np.zeros(L, dtype=bool)
        out = np.zeros((L, L), dtype=np.float32)
        for i, j in zip(ii, jj):
            if not used[i] and not used[j]:
                out[i, j] = out[j, i] = 1.0
                used[i] = used[j] = True
        return out

    def _rna_counts(self, logit: torch.Tensor, target: torch.Tensor,
                    ss_pairs: Optional[List[Tuple[int, int]]],
                    threshold: float, decode: str) -> Dict[str, Tuple[int, int, int]]:
        L = logit.shape[0]
        prob = torch.sigmoid(logit.detach().float()).cpu().numpy()
        tgt = target.detach().float().cpu().numpy()
        if decode == "matching":
            pred = self.decode_matching(prob, threshold)
        else:
            pred = (prob >= threshold).astype(np.float32)
        iu, ju = np.triu_indices(L, k=1)
        sep = (ju - iu)
        base = sep >= self.min_seq_sep
        pr, tg = pred[iu, ju], (tgt[iu, ju] > 0.5)

        def counts(sub_mask):
            pp, tt = pr[sub_mask] > 0.5, tg[sub_mask]
            tp = int((pp & tt).sum()); fp = int((pp & ~tt).sum()); fn = int((~pp & tt).sum())
            return tp, fp, fn

        out = {"all": counts(base), "lr": counts(base & (sep >= self.min_range))}
        if ss_pairs is not None:
            cross = _crossing_pair_set(ss_pairs)
            pkmask = np.array([(int(i), int(j)) in cross for i, j in zip(iu, ju)], dtype=bool)
            out["pk"] = counts(base & pkmask)
        else:
            out["pk"] = (0, 0, 0)
        return out

    def compute_metrics(self, predictions, targets, metas=None,
                        threshold: Optional[float] = None, decode: str = "threshold",
                        **kwargs) -> Dict[str, float]:
        thr = self.eval_threshold if threshold is None else float(threshold)
        preds = predictions if isinstance(predictions, list) else [predictions[i] for i in range(len(predictions))]
        tgts = targets if isinstance(targets, list) else [targets[i] for i in range(len(targets))]
        metas = metas or [None] * len(preds)

        agg = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ("all", "lr", "pk")}
        per_rna = {s: [] for s in ("all", "lr", "pk")}
        for p, t, meta in zip(preds, tgts, metas):
            p = p.squeeze(0) if p.dim() == 3 else p
            t = t.squeeze(0) if t.dim() == 3 else t
            ss_pairs = None
            if meta is not None and meta.get("ss") is not None:
                from finetune.lrbp_dataset import dotbracket_to_pairs
                ss_pairs = dotbracket_to_pairs(meta["ss"])
            c = self._rna_counts(p, t, ss_pairs, thr, decode)
            for s in ("all", "lr", "pk"):
                tp, fp, fn = c[s]
                agg[s]["tp"] += tp; agg[s]["fp"] += fp; agg[s]["fn"] += fn
                if tp + fn > 0:
                    pr, rc, f1 = _prf(tp, fp, fn)
                    per_rna[s].append((pr, rc, f1))

        out: Dict[str, float] = {}
        for s in ("all", "lr", "pk"):
            rows = per_rna[s]
            for name, k in (("precision", 0), ("recall", 1), ("f1", 2)):
                vals = [r[k] for r in rows if r[k] == r[k]]
                out[f"{s}_{name}"] = float(np.mean(vals)) if vals else float("nan")
            a = agg[s]
            mp, mr, mf = _prf(a["tp"], a["fp"], a["fn"])
            out[f"{s}_f1_micro"] = mf
            out[f"{s}_n_rna"] = float(len(rows))
        out["threshold"] = thr
        return out

    def find_best_threshold(self, predictions, targets, metas=None,
                            grid: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
                            key: str = "lr_f1", decode: str = "threshold") -> Tuple[float, float]:
        best_t, best_v = 0.5, -1.0
        for t in grid:
            v = self.compute_metrics(predictions, targets, metas=metas, threshold=float(t), decode=decode).get(key, float("nan"))
            if v == v and v > best_v:
                best_v, best_t = v, float(t)
        self.eval_threshold = best_t
        return best_t, best_v
