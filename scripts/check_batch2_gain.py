"""Does batch 2 improve the tagger the app SHIPS? Fixed recipe (the one frozen by cv_tagger.py), same held-out
pages, probes trained with vs without the extra photos. Run after cv_tagger.py (uses its cached embeddings).

  batch 1 photos: 5 folds by page over batch 1; probe trained on the other batch 1 pages
                  (before) or on those pages + all of batch 2 (after).
  batch 2 photos: 5 folds by page over batch 2; probe trained on batch 1 (before) or on batch 1 + the other
                  batch 2 pages (after). A photo's own page is never in training.

    python scripts/check_batch2_gain.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_tagger as V  # noqa: E402  (loads labels, cached embeddings and zero-shot scores)
from fpt import tagger as T  # noqa: E402

recipe = json.load(open(V.OUT / 'cv_meta.json'))['recipe']
ALL = np.arange(len(V.paths))
B1, B2 = ALL[V.batch == 1], ALL[V.batch == 2]


def predict(train, test):
    """The shipped recipe, with probes trained on photos `train`, applied to photos `test`."""
    sub = lambda p: {h: v[test] for h, v in p.items()}
    yt = {h: V.y[h][train] for h in V.y}
    pa = T.LinearProbe().fit(V.VA[train], yt).predict(V.VA[test], fallback=sub(V.ZS['A']))
    pb = T.LinearProbe().fit(V.VB[train], yt).predict(V.VB[test], fallback=sub(V.ZS['B']))
    A, B = sub(V.ZS['A']), sub(V.ZS['B'])
    parts = {'A1': sub(V.ZS['A1']), 'A': A, 'A+pr': V.avg(A, pa), 'B': B, 'A+B': V.avg(A, B), 'A+B+pr': V.avg(A, B, pa, pb)}
    return {h: parts[recipe[h]][h] for h in V.HEADS}


rows = []
for name, idx, extra in (('batch 1 photos', B1, B2), ('batch 2 photos', B2, None)):
    before = {h: np.zeros((len(V.paths), V.y[h].shape[1])) for h in V.HEADS}
    after = {h: np.zeros_like(before[h]) for h in V.HEADS}
    for a, b in GroupKFold(5).split(idx, groups=V.groups[idx]):
        tr, te = idx[a], idx[b]
        base = tr if extra is not None else B1                      # batch 2 'before' = trained on batch 1 only
        more = np.concatenate([tr, extra]) if extra is not None else np.concatenate([B1, tr])
        for out, train in ((before, base), (after, more)):
            p = predict(train, te)
            for h in V.HEADS:
                out[h][te] = p[h]
    sb, sa = V.scores(idx, before), V.scores(idx, after)
    for h in sb:
        rows.append({'photos': name, 'n': len(idx), 'attribute': h, 'recipe': recipe[h],
                     'right_before': sb[h][0], 'right_after': sa[h][0], 'top2_before': sb[h][1], 'top2_after': sa[h][1]})
res = pd.DataFrame(rows)
res.to_csv(V.OUT / 'batch2_gain_fixed_recipe.csv', index=False)
show = res.copy()
for c in ('right_before', 'right_after', 'top2_before', 'top2_after'):
    show[c] = (show[c] * 100).round(1)
pd.set_option('display.width', 160)
print(show.to_string(index=False))
