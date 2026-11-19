"""Reference-based accuracy metrics. Area-weighted so results stay correct on geographic grids."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def _edge(m):
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3)), border_value=1)


def segmentation_metrics(pred, ref, valid, area_m2=None, boundary_tol_px=2):
    w = (area_m2 if area_m2 is not None else np.ones(pred.shape)) * valid
    tp, fp, fn = (w * (pred & ref)).sum(), (w * (pred & ~ref)).sum(), (w * (~pred & ref)).sum()
    prec, rec = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
           "iou": tp / max(tp + fp + fn, 1e-9),
           "area_pred_ha": (tp + fp) / 1e4, "area_ref_ha": (tp + fn) / 1e4,
           "area_error_pct": 100 * ((tp + fp) - (tp + fn)) / max(tp + fn, 1e-9)}
    ep, er = _edge(pred & valid), _edge(ref & valid)
    if ep.any() and er.any():
        st = ndi.generate_binary_structure(2, 2)
        bp = (ep & ndi.binary_dilation(er, st, boundary_tol_px)).sum() / ep.sum()
        br = (er & ndi.binary_dilation(ep, st, boundary_tol_px)).sum() / er.sum()
        out["boundary_f1"] = 2 * bp * br / max(bp + br, 1e-9)
    else:
        out["boundary_f1"] = float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def calibration(p, ref, valid, bins=10):
    """Expected calibration error and reliability table of a probability-like score."""
    pv, rv = p[valid & np.isfinite(p)], ref[valid & np.isfinite(p)]
    edges, rows, ece = np.linspace(0, 1, bins + 1), [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (pv >= lo) & (pv < hi if hi < 1 else pv <= hi)
        if s.sum():
            conf, acc = float(pv[s].mean()), float(rv[s].mean())
            ece += s.mean() * abs(conf - acc)
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(s.sum()), "mean_score": round(conf, 3), "observed_burned": round(acc, 3)})
    return {"ece": round(float(ece), 4), "table": rows}


def uncertainty_vs_error(unc, pred, ref, valid, qs=(0.5, 0.8, 0.95)):
    """Does the uncertainty layer actually point at errors? Error rate above / below quantiles."""
    v = valid & np.isfinite(unc)
    err, u = (pred != ref)[v], unc[v]
    out = {"overall_error_rate": round(float(err.mean()), 4)}
    for q in qs:
        t = np.quantile(u, q)
        hi, lo = err[u > t], err[u <= t]
        out[f"q{int(q*100)}"] = {"error_rate_above": round(float(hi.mean()), 4) if hi.size else None,
                                 "error_rate_below": round(float(lo.mean()), 4) if lo.size else None}
    return out


def recall_by_class(pred, strata, valid, area_m2):
    """Share of each reference class (e.g. a damage grade) that the prediction covers."""
    out = {}
    for c in np.unique(strata[valid]):
        sel = (strata == c) & valid
        out[int(c)] = {"area_ha": round(float(area_m2[sel].sum() / 1e4), 1),
                       "predicted_burned_share": round(float(area_m2[sel & pred].sum() / max(area_m2[sel].sum(), 1e-9)), 3)}
    return out
