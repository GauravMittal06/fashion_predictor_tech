"""Layer 3, final evaluation: generate concepts from the frozen forecast and verify them with REAL photos.

  1. Calibrate the realism and near-copy thresholds on the team's 195 real outfit photos.
  2. Spring 2027 (Spain): brief from the frozen forecaster's calls -> SDXL-Turbo concepts -> checks.
     Optional (USE_REF=1): the same brief is also generated with the closest real photo as a style
     reference through IP-Adapter (synopsis Plan A + C), and both are compared. Not evaluated yet.
  3. 2025 generation replay: a brief from the forecast made in Jan 2025 (data up to Dec 2024 only),
     and a control prompt with no forecast. Do the forecast concepts look more like the team's REAL
     2025 photos than their 2024 photos, compared with the control?

    python scripts/build_designer_eval.py [n_per_set=12]
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C, design as D, forecast as F, verify as V  # noqa: E402
from fpt.tagger import FashionEmbedder  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = C.OUT / 'designer'
(OUT / 'images').mkdir(parents=True, exist_ok=True)
RP = C.OUT / 'replay'

# ---------- real photos ----------
z = np.load(C.OUT / 'tagger' / 'team_embeddings.npz')
real, real_paths = z['vecs'], [Path(str(p)).name for p in z['paths']]
labels = pd.read_csv(C.LABEL_FILE)
year = labels.set_index('image')['date'].pipe(pd.to_datetime, errors='coerce').dt.year
real_year = np.array([year.get(p, np.nan) for p in real_paths])
cal = V.calibrate(real)
print(f"real photos: {len(real)}; nearest-neighbour similarity median {cal['median']:.3f}; "
      f"realism threshold (5th pct) {cal['realism']:.3f}; near-copy threshold (99th pct) {cal['near_copy']:.3f}")

# ---------- frozen forecaster calls at any start month ----------
frozen = json.loads((RP / 'frozen.json').read_text())['chosen']
pz = np.load(RP / 'preds_ES.npz')
wide = pd.read_csv(RP / 'search_ES.csv', index_col=0, parse_dates=True)
terms = pd.read_csv(RP / 'terms_ES.csv', index_col=0)
Y = wide.to_numpy().T / pz['scale'][:, None]


def calls_at(month):
    t = int((wide.index < pd.Timestamp(month)).sum())
    ci = list(pz['cutoffs']).index(t)
    rows = []
    for s, term in enumerate(wide.columns):
        r = F.call(Y[s, :t], pz[frozen][ci, s])
        rows.append({'term': term, 'english': terms.loc[term, 'english'], 'attribute': terms.loc[term, 'attribute'],
                     'call': r['call'], 'change': r['change']})
    return pd.DataFrame(rows)


emb = FashionEmbedder()
results = {'calibration': {k: v for k, v in cal.items() if k != 'nn'}, 'frozen_forecaster': frozen}


def run(tag, prompt, brief, seed, reference=None):
    emb.to('cpu')
    imgs = D.generate(prompt, n=N, seed=seed) if reference is None else D.generate_ref(prompt, reference, n=N, seed=seed)
    D.load_generator().maybe_free_model_hooks()
    emb.to(F.DEVICE)
    vecs = emb.images(imgs)
    chk = V.check(vecs, brief, emb, real, cal) if brief else None
    for i, im in enumerate(imgs):
        im.save(OUT / 'images' / f'{tag}_{i}.jpg', quality=90)
    return imgs, vecs, chk


def summary(chk):
    return {'concepts': len(chk), 'kept': sum(c['kept'] for c in chk),
            'match_brief': sum(c['matches_brief'] for c in chk), 'look_real': sum(c['looks_real'] for c in chk),
            'near_copies': sum(c['near_copy'] for c in chk),
            'median_realism_percentile': float(np.median([c['realism_percentile'] for c in chk]))}


# ---------- 2. Spring 2027 ----------
calls27 = pd.read_csv(RP / 'calls_ES.csv')
b27 = D.make_search_brief(calls27, 'Spain', 'spring 2027')
p27 = D.search_prompt(b27)
print('\nSpring 2027 brief:', {k: b27[k] for k in ('style', 'fabrics', 'colours', 'motif', 'evidence')})
print('prompt:', p27)
_, _, chk = run('spring2027', p27, b27, seed=0)
results['spring2027'] = {'brief': b27, 'prompt': p27, 'text_only': summary(chk), 'checks': chk}
print('text only:', summary(chk))

if os.environ.get('USE_REF') == '1' and D.use_reference():   # optional IP-Adapter step: off unless USE_REF=1
    ref_idx = int(np.argmax(real @ emb.texts([p27])[0]))
    from PIL import Image
    from fpt.tagger import crop_person
    ref = crop_person(Image.open(C.TEAM_DIR / real_paths[ref_idx]).convert('RGB'))[0]
    _, _, chk_r = run('spring2027_ref', p27, b27, seed=0, reference=ref)
    results['spring2027'].update({'with_reference': summary(chk_r), 'checks_ref': chk_r, 'reference': real_paths[ref_idx]})
    print('with real-photo reference:', summary(chk_r))

# ---------- 3. 2025 generation replay ----------
calls25 = calls_at('2025-01-01')
b25 = D.make_search_brief(calls25, 'Southern Europe', 'spring and summer 2025')
p25 = D.search_prompt(b25)
control = ('full-body street style photo of a woman in Southern Europe wearing an outfit, '
           'natural daylight, editorial photography, sharp focus')
_, v25, chk25 = run('replay2025', p25, b25, seed=100)
_, vctl, _ = run('replay2025_control', control, None, seed=100)
in25, in24 = real_year == 2025, real_year == 2024


def lean(v):
    """How much closer to real 2025 outfits than to real 2024 outfits (mean cosine similarity)."""
    return float((v @ real[in25].T).mean() - (v @ real[in24].T).mean())


rep = {'brief': b25, 'prompt': p25, 'control_prompt': control, 'photos_2025': int(in25.sum()),
       'photos_2024': int(in24.sum()), 'forecast_lean_to_2025': lean(v25), 'control_lean_to_2025': lean(vctl),
       'forecast_concepts_closer_to_2025': int(sum(lean(v[None]) > 0 for v in v25)),
       'control_concepts_closer_to_2025': int(sum(lean(v[None]) > 0 for v in vctl)), 'checks': summary(chk25)}
results['replay2025'] = rep
print('\n2025 replay brief:', {k: b25[k] for k in ('style', 'fabrics', 'colours', 'motif', 'evidence')})
print(f"forecast concepts lean to 2025 by {rep['forecast_lean_to_2025']:+.4f} "
      f"({rep['forecast_concepts_closer_to_2025']}/{N} closer to 2025); control {rep['control_lean_to_2025']:+.4f} "
      f"({rep['control_concepts_closer_to_2025']}/{N})")
(OUT / 'results.json').write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print('saved', OUT)
