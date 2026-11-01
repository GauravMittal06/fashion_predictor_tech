"""Layer 1, final evaluation: nested 5-fold cross-validation by Instagram page on ALL labelled photos
(batch 1: the team's 195 screenshots; batch 2: 100 influencer photos from 10 more pages).

Outer loop: the pages are split into 5 groups; each group is held out once as test photos.
Inner loop: on the remaining pages only, every candidate recipe is scored (4 inner folds) and the best
one PER ATTRIBUTE is chosen; that choice is then scored on the held-out pages it never saw.
So every photo gets an honest score, and the recipe choice itself is tested too.

Candidates (fixed before running):
  A1      Marqo-FashionSigLIP zero-shot, one description per class (the earlier tagger)
  A       Marqo zero-shot, several descriptions per class + mirrored second look
  A+pr    A averaged with a linear probe trained on the training photos
  B       FashionCLIP 2.0 zero-shot, several descriptions + mirrored look
  A+B     average of A and B
  A+B+pr  average of A, B and linear probes on both models' features

Did batch 2 help? Scored on the SAME photos before and after (batch_comparison.csv):
  batch 1 photos  before = nested CV on batch 1 only (the earlier result, reproduced exactly)
                  after  = nested CV on both batches (training folds now include batch 2 pages)
  batch 2 photos  before = the earlier frozen tagger (chosen and trained on batch 1 only): it never saw them
                  after  = nested CV on both batches (a photo's own page is always held out)

Finally the recipe is chosen on all photos the same way and refit on all of them for the app.

    python scripts/cv_tagger.py
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402
from fpt import tagger as T  # noqa: E402
from train_tagger import load_table, get_labels  # noqa: E402

OUT = C.OUT / 'tagger'
CACHE = OUT / 'cv_embeddings.npz'

df = load_table()
if 'wrong_post' in df:
    df = df[df.wrong_post.fillna(0) != 1].reset_index(drop=True)
cues = [c for c in C.CUES if df[c].notna().any() and df[c].fillna(0).sum() >= 5]
HEADS = ['style', 'fabric', 'colour'] + cues
y = get_labels(df)
groups = df['group'].to_numpy()
batch = df['batch'].to_numpy()
paths = list(df['path'])
print(f'{len(df)} photos ({(batch == 1).sum()} batch 1, {(batch == 2).sum()} batch 2), {len(set(groups))} pages; '
      f'heads: {HEADS}')

# ---------- embeddings (cached per photo; only new photos are embedded) ----------
old = {}
if CACHE.exists():
    z = np.load(CACHE)
    old = {p: i for i, p in enumerate(z['paths'])}
new = [p for p in paths if p not in old]
if new:
    print(f'embedding {len(new)} new photos')
    ea = T.FashionEmbedder()
    nA1, nA = ea.images(new), ea.images(new, flip=True)
    del ea
    eb = T.FashionCLIPEmbedder()
    nB = eb.images(new, flip=True)
    del eb
    keep = list(old)
    stack = lambda k, n: np.concatenate([z[k][[old[p] for p in keep]], n]) if keep else n
    allp = keep + new
    np.savez(CACHE, paths=np.array(allp), A1=stack('A1', nA1), A=stack('A', nA), B=stack('B', nB))
z = np.load(CACHE)
pos = {p: i for i, p in enumerate(z['paths'])}
order = [pos[p] for p in paths]
VA1, VA, VB = z['A1'][order], z['A'][order], z['B'][order]

ea, eb = T.FashionEmbedder(), T.FashionCLIPEmbedder()
ZS = {'A1': T.zero_shot(ea, VA1), 'A': T.zero_shot_ens(ea, VA), 'B': T.zero_shot_ens(eb, VB)}


def avg(*ps):
    return {h: np.mean([p[h] for p in ps], 0) for h in HEADS}


def candidates(tr, te):
    """Predictions on photos te for every candidate, training only on photos tr."""
    sub = lambda p, idx: {h: v[idx] for h, v in p.items()}
    ya = {h: y[h][tr] for h in y}
    pa = T.LinearProbe().fit(VA[tr], ya).predict(VA[te], fallback=sub(ZS['A'], te))
    pb = T.LinearProbe().fit(VB[tr], ya).predict(VB[te], fallback=sub(ZS['B'], te))
    A, B, A1 = sub(ZS['A'], te), sub(ZS['B'], te), sub(ZS['A1'], te)
    return {'A1': A1, 'A': A, 'A+pr': avg(A, pa), 'B': B, 'A+B': avg(A, B), 'A+B+pr': avg(A, B, pa, pb)}


def metric(h, yt, p):
    """The number used to CHOOSE: style accuracy; fabric/colour 'main guess is one of the true ones';
    details balanced accuracy."""
    if h == 'style':
        return np.mean(p.argmax(1) == yt.argmax(1))
    if h in ('fabric', 'colour'):
        has = yt.sum(1) > 0
        return np.mean(yt[np.arange(len(yt)), p.argmax(1)][has])
    return balanced_accuracy_score(yt.ravel().astype(int), (p.ravel() > 0.5).astype(int))


def report(h, yt, p, ytrain):
    """Plain-language scores for one attribute on held-out photos, plus the most-common-answer guess."""
    if h == 'style':
        maj = ytrain.sum(0).argmax()
        return {'right': np.mean(p.argmax(1) == yt.argmax(1)), 'top2': np.mean([yt[i, np.argsort(-p[i])[:2]].max() for i in range(len(yt))]),
                'guess': np.mean(yt.argmax(1) == maj), 'macroF1': f1_score(yt.argmax(1), p.argmax(1), average='macro',
                                                                            labels=np.unique(yt.argmax(1)), zero_division=0)}
    if h in ('fabric', 'colour'):
        has = yt.sum(1) > 0
        top2 = np.array([yt[i, np.argsort(-p[i])[:2]].max() for i in range(len(yt))])
        maj = ytrain.sum(0).argmax()
        return {'right': np.mean(yt[np.arange(len(yt)), p.argmax(1)][has]), 'top2': top2[has].mean(),
                'guess': yt[has][:, maj].mean(), 'macroF1': np.nan}
    t, q = yt.ravel().astype(int), (p.ravel() > 0.5).astype(int)
    return {'right': np.mean(t == q), 'top2': np.nan, 'guess': max(np.mean(t), 1 - np.mean(t)),
            'macroF1': f1_score(t, q, zero_division=0)}


def choose(idx):
    """Pick the best candidate per attribute using 4 inner folds over the pages in idx."""
    scores = {}
    for itr, ite in GroupKFold(4).split(idx, groups=groups[idx]):
        c = candidates(idx[itr], idx[ite])
        for name, p in c.items():
            for h in HEADS:
                scores.setdefault((name, h), []).append(metric(h, y[h][idx[ite]], p[h]))
    return {h: max({n for n, _ in scores}, key=lambda n: np.mean(scores[(n, h)])) for h in HEADS}


def nested(idx, label):
    """Nested CV over the photos idx. Returns per-fold rows, per-candidate rows, picks and out-of-fold predictions."""
    rows, plain, picks = [], [], []
    oof = {h: np.zeros((len(paths), y[h].shape[1])) for h in HEADS}
    for k, (a, b) in enumerate(GroupKFold(5).split(idx, groups=groups[idx])):
        tr, te = idx[a], idx[b]
        pick = choose(tr)
        picks.append(pick)
        c = candidates(tr, te)
        for h in HEADS:
            oof[h][te] = c[pick[h]][h]
            rows.append({'fold': k, 'attribute': h, 'photos': len(te), **report(h, y[h][te], c[pick[h]][h], y[h][tr])})
            for name, p in c.items():
                plain.append({'fold': k, 'attribute': h, 'candidate': name, **report(h, y[h][te], p[h], y[h][tr])})
        print(f'[{label}] fold {k + 1}: {len(te)} test photos from {len(set(groups[te]))} pages; chosen {pick}')
    return pd.DataFrame(rows), pd.DataFrame(plain), picks, oof


def scores(idx, pred):
    """right / top-2 for each attribute over the photos idx (fabric and colour: photos with a label only)."""
    out = {}
    for h in ('style', 'fabric', 'colour'):
        r = report(h, y[h][idx], pred[h][idx], y[h][idx])
        out[h] = (r['right'], r['top2'])
    return out


if __name__ == '__main__':
    ALL = np.arange(len(paths))
    B1, B2 = ALL[batch == 1], ALL[batch == 2]

    # ---------- official numbers: every photo, both batches ----------
    res, plain, picks, oof_after = nested(ALL, 'both batches')
    summary = res.groupby('attribute', sort=False).agg(
        right=('right', 'mean'), right_min=('right', 'min'), right_max=('right', 'max'),
        top2=('top2', 'mean'), guess=('guess', 'mean'), macroF1=('macroF1', 'mean'))
    summary.to_csv(OUT / 'cv_summary.csv')
    res.to_csv(OUT / 'cv_folds.csv', index=False)
    cand = plain.groupby(['attribute', 'candidate'], sort=False)[['right', 'top2']].mean().unstack(0)
    cand.to_csv(OUT / 'cv_candidates.csv')

    # ---------- did batch 2 help? same photos, before and after ----------
    _, _, _, oof_before = nested(B1, 'batch 1 only')
    rec1 = choose(B1)                                     # the earlier frozen tagger, rebuilt from batch 1 only
    pa = T.LinearProbe().fit(VA[B1], {h: y[h][B1] for h in y}).predict(VA, fallback=ZS['A'])
    pb = T.LinearProbe().fit(VB[B1], {h: y[h][B1] for h in y}).predict(VB, fallback=ZS['B'])
    parts = {'A1': ZS['A1'], 'A': ZS['A'], 'A+pr': avg(ZS['A'], pa), 'B': ZS['B'], 'A+B': avg(ZS['A'], ZS['B']),
             'A+B+pr': avg(ZS['A'], ZS['B'], pa, pb)}
    frozen1 = {h: parts[rec1[h]][h] for h in HEADS}
    comp = []
    for name, idx, before, after in (('batch 1 photos (Southern Europe screenshots)', B1, oof_before, oof_after),
                                     ('batch 2 photos (influencer pages)', B2, frozen1, oof_after)):
        sb, sa = scores(idx, before), scores(idx, after)
        for h in sb:
            comp.append({'photos': name, 'n': len(idx), 'attribute': h,
                         'right_before': sb[h][0], 'right_after': sa[h][0],
                         'top2_before': sb[h][1], 'top2_after': sa[h][1]})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / 'batch_comparison.csv', index=False)

    pd.set_option('display.width', 180)
    print('\n=== NESTED CROSS-VALIDATION, all photos (every photo scored by a model that never saw its page) ===')
    print((summary * 100).round(1).to_string())
    print('\nEach candidate on its own (for comparison, not used to choose):')
    print((cand['right'] * 100).round(1).to_string())
    print('\n=== SAME PHOTOS, BEFORE vs AFTER adding batch 2 ===')
    show = comp.copy()
    for c in ('right_before', 'right_after', 'top2_before', 'top2_after'):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))

    # ---------- final recipe: chosen on all photos, refit on all photos ----------
    final = choose(ALL)
    joblib.dump({'recipe': final, 'probe_A': T.LinearProbe().fit(VA, y), 'probe_B': T.LinearProbe().fit(VB, y),
                 'heads': HEADS}, OUT / 'tagger_final.joblib')
    json.dump({'recipe': final, 'photos': len(paths), 'pages': int(len(set(groups))),
               'batches': {'1': int((batch == 1).sum()), '2': int((batch == 2).sum())},
               'labels': [C.LABEL_FILE.name, C.BATCH2_LABELS.name], 'picks_per_fold': picks,
               'recipe_batch1_only': rec1}, open(OUT / 'cv_meta.json', 'w'), indent=2)
    print('\nFinal recipe per attribute (frozen for the app):', final)
