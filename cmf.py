"""Multispectral adaptation of Contrastive Mask Fidelity (CMF).

Original (Cao et al. 2026): F(M) = P(c | keep view) - P(c | erase view), views composited with a
Gaussian blur, P from a frozen RemoteCLIP softmax over class texts, RGB images at 0.05-1 m GSD.

Here (NEW, not from the paper): the same keep/erase construction is applied to reflectance bands,
and the judge is a frozen physical index instead of an RGB vision-language model, because an RGB
VLM has no defined behaviour on SWIR/red-edge bands or at 20 m GSD. The default judge uses the
Burned Area Index (Chuvieco et al. 2002), which reads only red and NIR, so it shares no SWIR band
with the NBR-family indices that generate the candidate masks. Independence is partial (both read
NIR). The score ranks competing masks; it is not ground truth.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


class BAIJudge:
    """P(burned | view) from pooled per-pixel evidence, two-class softmax (burned vs not burned)."""
    bands = ("red", "nir")

    def __init__(self, tau=8.0, top=0.10):
        self.tau, self.top = tau, top

    def fit(self, bands, valid):
        """Freeze the normalisation on the full scene once; it never sees any mask."""
        bai = self._bai(bands)
        v = bai[valid & np.isfinite(bai)]
        self.med, self.mad = float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)
        return self

    @staticmethod
    def _bai(b):
        return np.log(1.0 / ((0.1 - b["red"]) ** 2 + (0.06 - b["nir"]) ** 2 + 1e-6))

    def evidence(self, bands):
        return 1 / (1 + np.exp(-((self._bai(bands) - self.med) / self.mad - 1.5)))

    def prob(self, bands, valid):
        e = self.evidence(bands)[valid]
        if e.size == 0:
            return 0.5
        k = max(1, int(self.top * e.size))
        eb, eu = np.sort(e)[-k:].mean(), np.sort(1 - e)[-k:].mean()
        return float(1 / (1 + np.exp(-self.tau * (eb - eu))))


def cmf_score(bands, mask, valid, judge, sigma=12.0):
    """F in [-1, 1]: high when burn evidence sits inside the mask (purity) and none is left outside
    (completeness). `bands` must contain the judge's bands on the analysis grid."""
    keep, erase = {}, {}
    for name in judge.bands:
        a = np.nan_to_num(bands[name], nan=float(np.nanmedian(bands[name])))
        blur = ndi.gaussian_filter(a, sigma)
        keep[name], erase[name] = np.where(mask, a, blur), np.where(mask, blur, a)
    pk, pe = judge.prob(keep, valid), judge.prob(erase, valid)
    return {"F": pk - pe, "p_keep": pk, "p_erase": pe}


def _chips(shape, chip):
    for r in range(0, shape[0] - chip + 1, chip):
        for c in range(0, shape[1] - chip + 1, chip):
            yield (slice(r, r + chip), slice(c, c + chip))


def cmf_tiled(bands, mask, valid, judge, chip=64, also=None):
    """Mean chip-level F (the paper scores ~224 px images, not whole scenes). Chips are scored when
    they are mostly valid and `mask` or `also` claims something inside them."""
    fs = []
    for sl in _chips(mask.shape, chip):
        m, v = mask[sl], valid[sl]
        claimed = m.any() or (also is not None and also[sl].any())
        if v.mean() < 0.8 or not claimed:
            continue
        fs.append(cmf_score({k: bands[k][sl] for k in judge.bands}, m, v, judge, chip / 8)["F"])
    return {"F": float(np.mean(fs)) if fs else float("nan"), "n_chips": len(fs), "per_chip": fs}


def arbitrate(bands, mask_a, mask_b, valid, judge, chip=64, margin=0.0):
    """Symmetric comparison of two masks (paper Eq. 6-7), chip by chip. Ties defer to mask_a."""
    fa = cmf_tiled(bands, mask_a, valid, judge, chip, also=mask_b)
    fb = cmf_tiled(bands, mask_b, valid, judge, chip, also=mask_a)
    d = fb["F"] - fa["F"]
    wins = float(np.mean(np.array(fb["per_chip"]) > np.array(fa["per_chip"]))) if fa["n_chips"] else float("nan")
    return {"F_a": round(fa["F"], 4), "F_b": round(fb["F"], 4), "delta": round(d, 4), "n_chips": fa["n_chips"],
            "b_win_rate": round(wins, 3), "preferred": "b" if d > margin else "a"}


def corruption_controls(bands, ref_mask, valid, judge, chip=64, px=3, min_frac=0.1, max_frac=0.9):
    """Paper Sec. 4.2 validity control: does F rank a clean mask above dilated / eroded / shifted copies?"""
    H, W = ref_mask.shape
    st, res = np.ones((3, 3), bool), {"dilate": [], "erode": [], "shift": []}
    for sl in _chips((H, W), chip):
        m, v = ref_mask[sl], valid[sl]
        if v.mean() < 0.9 or not (min_frac <= m[v].mean() <= max_frac):
            continue
        b = {k: bands[k][sl] for k in judge.bands}
        f0 = cmf_score(b, m, v, judge, chip / 8)["F"]
        variants = {"dilate": ndi.binary_dilation(m, st, px), "erode": ndi.binary_erosion(m, st, px),
                    "shift": np.roll(m, (2 * px, 2 * px), (0, 1))}
        for name, mm in variants.items():
            res[name].append(f0 > cmf_score(b, mm, v, judge, chip / 8)["F"])
    return {k: {"detection_rate": round(float(np.mean(v)), 3) if v else None, "n_chips": len(v)} for k, v in res.items()}
