"""Knowledge Space + Solution Space retrieval (lightweight, offline, deterministic).

Task-Aware Retrieval: the retrieval key is the inferred task type, not the raw query (RS-Agent).
DualRAG-style retrieval: one 'global' query plus per-keyword queries whose document quota follows
keyword weights (RS-Agent Eq. 3-6), on a TF-IDF index instead of a graph/dense store.
"""
from __future__ import annotations
import math, re
from collections import Counter
from pathlib import Path
import yaml
from ..config import ROOT

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]+")
DOMAIN_WEIGHTS = {  # task / index / sensor terms outrank generic words (paper's prioritisation order)
    10: "burn burned fire wildfire severity flood deforestation scar".split(),
    8: "nbr dnbr rdnbr rbr ndvi bais2 mirbi cmf sar threshold cloud uncertainty".split(),
    7: "sentinel-2 sentinel-1 landsat swir nir b12 b8a vv vh mtbs effis firecci stac".split(),
}


def tokens(text):
    """Lower-case tokens with a minimal plural stem (thresholds -> threshold)."""
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in TOKEN.findall(text.lower())]


class TfIdf:
    def __init__(self, docs: dict[str, str]):
        self.ids, self.tf = list(docs), [Counter(tokens(t)) for t in docs.values()]
        df = Counter(w for tf in self.tf for w in tf)
        self.idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
        self.norm = [math.sqrt(sum((c * self.idf[w]) ** 2 for w, c in tf.items())) or 1 for tf in self.tf]

    def search(self, query, k=3):
        q = Counter(tokens(query))
        scores = [(sum(c * self.idf.get(w, 0) ** 2 * tf.get(w, 0) for w, c in q.items()) / n, i)
                  for i, (tf, n) in enumerate(zip(self.tf, self.norm))]
        return [(self.ids[i], round(s, 4)) for s, i in sorted(scores, reverse=True)[:k] if s > 0]


class KnowledgeBase:
    def __init__(self, root: Path | None = None):
        root = Path(root or ROOT / "knowledge")
        self.workflows = yaml.safe_load((root / "workflows.yaml").read_text())
        self.chunks = {}
        for md in sorted(root.glob("*.md")):
            for i, part in enumerate(re.split(r"\n(?=## )", md.read_text())):
                if len(part.strip()) > 80:                      # skip bare titles
                    self.chunks[f"{md.stem}#{i}"] = part.strip()
        self.kb_index = TfIdf(self.chunks) if self.chunks else None
        self.wf_index = TfIdf({k: f"{k.replace('_', ' ')} {v['description']} {v.get('keywords', '')}"
                               for k, v in self.workflows.items()})

    def workflow_for(self, task_type: str):
        hits = self.wf_index.search(task_type.replace("_", " "), k=1)
        name = task_type if task_type in self.workflows else (hits[0][0] if hits else None)
        if name is None:
            return None, None
        wf = self.workflows[name]
        if "same_as" in wf:
            wf = {**self.workflows[wf["same_as"]], **{k: v for k, v in wf.items() if k != "same_as"}}
        return name, wf

    @staticmethod
    def keywords(query):
        out = {}
        for w in tokens(query):
            for weight, words in DOMAIN_WEIGHTS.items():
                if w in words:
                    out[w] = max(out.get(w, 0), weight)
        return out or {w: 3 for w in tokens(query)[:5]}

    def dual_retrieve(self, query, n=4):
        if not self.kb_index:
            return []
        kws = self.keywords(query)
        got = dict(self.kb_index.search(" ".join(kws), k=n))            # global path
        total = sum(kws.values())
        quota = {k: int(w / total * n) for k, w in kws.items()}
        for k in sorted(kws, key=kws.get, reverse=True)[: n - sum(quota.values())]:
            quota[k] += 1                                               # distribute the remainder by weight
        for kw, q in quota.items():                                     # keyword path
            for cid, s in self.kb_index.search(kw, k=q):
                got.setdefault(cid, s)
        return [{"id": cid, "score": s, "text": self.chunks[cid]} for cid, s in got.items()]
