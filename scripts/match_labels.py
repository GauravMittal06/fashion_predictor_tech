"""Match the team's screenshots (dataset_images_1/, in screenshot order) to the rows of
SOUTHERN EUROPE DATA.xlsx (IMAGE NO.1..101, same order). A post can have 0-4 screenshots.

Monotonic alignment by dynamic programming: each screenshot goes to one row, rows never go
backwards, and the score rewards (a) the tagger's image-text similarity between the screenshot and
the row's colour/fabric/style description, (b) keeping near-identical consecutive screenshots on the
same row, with a small penalty for skipping rows. Writes data/screenshot_matches.csv for review.

    python scripts/match_labels.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402
from fpt.tagger import FashionEmbedder  # noqa: E402

LAMBDA, SKIP = 2.0, 0.6

rows = pd.read_excel(C.TEAM_DIR / 'SOUTHERN EUROPE DATA.xlsx')
rows['row'] = np.arange(1, len(rows) + 1)
z = np.load(C.OUT / 'tagger' / 'team_embeddings.npz')
paths, V = [str(p) for p in z['paths']], z['vecs']


def describe(r):
    c = str(r.COLOUR).lower().replace(',', ' and ').strip(' -')
    f = str(r.FABRIC).lower().replace(',', ' and ').replace('nan', '').strip(' -')
    s = str(r.STYLE).lower().replace('nan', '').strip(' -')
    return f'a photo of a person wearing {c} {f} clothes, {s}'.replace('  ', ' ')


emb = FashionEmbedder()
T = emb.texts([describe(r) for r in rows.itertuples()])
S = V @ T.T                                              # (images, rows)
S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
cont = np.r_[0, np.sum(V[1:] * V[:-1], 1)]              # similarity to the previous screenshot
cont = (cont - np.median(cont)) / cont.std()

n, m = S.shape
dp = np.full((n, m), -np.inf)
back = np.zeros((n, m), int)
dp[0] = S[0] - SKIP * np.arange(m)
for i in range(1, n):
    best_prev = np.maximum.accumulate(dp[i - 1] + SKIP * np.arange(m))   # max over r' < r of dp - skip cost
    arg_prev = np.zeros(m, int)
    cur, k = -np.inf, 0
    for r in range(m):
        if r > 0 and dp[i - 1, r - 1] + SKIP * (r - 1) > cur:
            cur, k = dp[i - 1, r - 1] + SKIP * (r - 1), r - 1
        arg_prev[r] = k
    switch = np.r_[-np.inf, [dp[i - 1, arg_prev[r]] - SKIP * (r - arg_prev[r] - 1) for r in range(1, m)]]
    stay = dp[i - 1] + LAMBDA * cont[i]
    take_stay = stay >= switch
    dp[i] = S[i] + np.where(take_stay, stay, switch)
    back[i] = np.where(take_stay, np.arange(m), arg_prev)
assign = np.zeros(n, int)
assign[-1] = int(np.argmax(dp[-1] - SKIP * (m - 1 - np.arange(m))))
for i in range(n - 1, 0, -1):
    assign[i - 1] = back[i, assign[i]]

out = pd.DataFrame({'image': [Path(p).name for p in paths], 'row': assign + 1,
                    'fit': S[np.arange(n), assign].round(2)})
out = out.merge(rows[['row', 'INSTAGRAM PAGE', 'COLOUR', 'FABRIC', 'STYLE']], on='row')
out.to_csv(C.ROOT / 'data' / 'screenshot_matches.csv', index=False)
print(f'{n} screenshots -> {out.row.nunique()} of {m} rows; rows with none:',
      sorted(set(range(1, m + 1)) - set(out.row)))
