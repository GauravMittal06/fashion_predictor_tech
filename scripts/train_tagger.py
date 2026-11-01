"""Layer 1 on the team's ~150 labelled photos: Plan A zero-shot vs Plan A linear probe vs the
majority-class guesser, scored on the SAME validation photos (split by influencer group).
The better option is kept per attribute; the test photos are used only with --test.

Needs images/ and labels.csv (image, group, style, fabric, colour, button, zipper, motif;
multi-values joined by |) in the project folder.

    python scripts/train_tagger.py [--test]
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402
from fpt.tagger import FashionEmbedder, LinearProbe, zero_shot  # noqa: E402

OUT = C.OUT / 'tagger'
OUT.mkdir(parents=True, exist_ok=True)
SIGMOID_HEADS = ['fabric', 'colour'] + C.CUES
HEADS = ['style'] + SIGMOID_HEADS
GRID = np.round(np.arange(0.20, 0.75, 0.05), 2)


# ---------- split and labels (from the team's data.py, without TensorFlow) ----------
def load_table(batches=(1, 2)):
    """Batch 1: the team's screenshots (labels.csv); batch 2: the influencer photos (labels_batch2.csv)."""
    parts = []
    if 1 in batches:
        a = pd.read_csv(C.LABEL_FILE, dtype={'row': str}).assign(batch=1)
        a['path'] = [str(C.IMAGE_DIR / i) for i in a['image']]
        parts.append(a)
    if 2 in batches and C.BATCH2_LABELS.exists():
        b = pd.read_csv(C.BATCH2_LABELS, dtype={'row': str}).assign(batch=2)
        b['path'] = [str(C.BATCH2_DIR / i) for i in b['image']]
        parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    df = df[[Path(p).exists() for p in df['path']]].reset_index(drop=True)
    for col in ['fabric', 'colour']:
        df[col] = df[col].fillna('').astype(str).apply(lambda s: [v for v in s.split('|') if v])
    return df


def group_split(df, frac, tries=200):
    best = None
    for seed in range(tries):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
                    .split(df, groups=df['group']))
        gap = abs(len(b) / len(df) - frac)
        if best is None or gap < best[0]:
            best = (gap, a, b)
    return df.iloc[best[1]], df.iloc[best[2]]


def split_data(df):
    trainval, test = group_split(df, 0.15)
    train, val = group_split(trainval, 0.15 / 0.85)
    sets = {'train': train, 'val': val, 'test': test}
    for x in sets:
        for y in sets:
            if x < y:
                assert not set(sets[x]['group']) & set(sets[y]['group']), f'group leak {x}/{y}'
    return [p.reset_index(drop=True) for p in (train, val, test)]


_fab = MultiLabelBinarizer(classes=C.FABRICS).fit([C.FABRICS])
_col = MultiLabelBinarizer(classes=C.COLOURS).fit([C.COLOURS])


def get_labels(df):
    style = np.zeros((len(df), len(C.STYLES)), 'float32')
    style[np.arange(len(df)), df['style'].map(C.STYLES.index).values] = 1.0
    y = {'style': style, 'fabric': _fab.transform(df['fabric']).astype('float32'),
         'colour': _col.transform(df['colour']).astype('float32')}
    for cue in C.CUES:                       # an unlabelled cue column becomes zeros and is not scored
        y[cue] = df[[cue]].fillna(0).values.astype('float32')
    return y


# ---------- scores (from the team's evaluate.py) ----------
def head_f1(y, p, head, t=0.5):
    if head == 'style':
        true = y.argmax(1)
        return f1_score(true, p.argmax(1), average='macro', labels=np.unique(true), zero_division=0)
    avg = 'macro' if head in ('fabric', 'colour') else 'binary'
    return f1_score(y.astype(int), (p > t).astype(int), average=avg, zero_division=0)


def tune_thresholds(y, pred):
    """ONE threshold per head: the validation set (~22 photos) is too small for more."""
    return {h: float(max(GRID, key=lambda t: head_f1(y[h], pred[h], h, t))) for h in SIGMOID_HEADS}


def score_table(y, options):
    table, thr = {}, {}
    for name, pred in options.items():
        thr[name] = tune_thresholds(y, pred) if name != 'majority guesser' else {h: 0.5 for h in SIGMOID_HEADS}
        table[name] = {h: head_f1(y[h], pred[h], h, thr[name].get(h, 0.5)) for h in HEADS}
    return pd.DataFrame(table).T, thr


if __name__ == '__main__':
    if not C.LABEL_FILE.exists():
        sys.exit(f'Put labels.csv in {C.ROOT} and the photos in {C.IMAGE_DIR}')
    df = load_table()
    labelled = [c for c in C.CUES if df[c].notna().any()]
    HEADS[:] = ['style', 'fabric', 'colour'] + labelled
    SIGMOID_HEADS[:] = ['fabric', 'colour'] + labelled
    print('heads scored:', HEADS, '| no labels for:', [c for c in C.CUES if c not in labelled])
    train, val, test = split_data(df)
    print(f'train {len(train)} / val {len(val)} / test {len(test)} photos')
    emb = FashionEmbedder()
    X = {n: emb.images(list(p['path'])) for n, p in (('train', train), ('val', val), ('test', test))}
    y = {n: get_labels(p) for n, p in (('train', train), ('val', val), ('test', test))}
    probe = LinearProbe().fit(X['train'], y['train'])

    def options(split):
        zs = zero_shot(emb, X[split])
        return {'majority guesser': {h: np.tile(y['train'][h].mean(0), (len(X[split]), 1)) for h in HEADS},
                'A zero-shot': zs,
                'A linear probe': probe.predict(X[split], fallback=zs),
                'A zero-shot + probe': {h: (zs[h] + probe.predict(X[split], fallback=zs)[h]) / 2 for h in HEADS}}

    table, thr = score_table(y['val'], options('val'))
    print('\n=== VALIDATION macro-F1 per attribute ===\n', table.round(3).to_string())
    choice = {h: table.drop('majority guesser')[h].idxmax() for h in HEADS}
    print('\nChosen per attribute:', choice)

    joblib.dump(probe, OUT / 'probe.joblib')
    table.to_csv(OUT / 'validation.csv')
    json.dump({'choice': choice, 'thresholds': {n: thr[n] for n in thr},
               'sizes': {'train': len(train), 'val': len(val), 'test': len(test)}},
              open(OUT / 'choice.json', 'w'), indent=2)
    np.savez(OUT / 'embeddings.npz', vecs=np.concatenate([X['train'], X['val'], X['test']]),
             paths=np.array(list(train['path']) + list(val['path']) + list(test['path'])))

    if '--test' in sys.argv:                              # once, after every choice is final
        opts = options('test')
        rows = {}
        for name in ('majority guesser', 'chosen'):
            pred = {h: opts[choice[h] if name == 'chosen' else name][h] for h in HEADS}
            th = {h: thr[choice[h]].get(h, 0.5) if name == 'chosen' else 0.5 for h in SIGMOID_HEADS}
            rows[name] = {h: head_f1(y['test'][h], pred[h], h, th.get(h, 0.5)) for h in HEADS}
        t = pd.DataFrame(rows).T
        print('\n=== TEST (used once) ===\n', t.round(3).to_string())
        t.to_csv(OUT / 'test.csv')
