"""Check our zero-shot tagger on real Barcelona street photos: how often does Marqo-FashionSigLIP
(no training) give the same answer as GeoStyle's CNN, which was trained on 27k labelled people?

First screens every photo: only clear photos of a clothed person are kept (no statues, objects,
artworks, nudity or underwear), plus a manual block list in config.EXCLUDE_PHOTOS.
Also embeds the kept photos once, for the concept board and the near-copy check.

The repository holds only the 28 photos that passed this screen. Listed photos that are missing are downloaded
again from their Flickr URL (same 640 px resize as fetch_barcelona_photos.py), so the screen runs on the same 72.

    python scripts/check_tagger_barcelona.py
"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402
from fpt.tagger import FashionEmbedder, crop_person, trend_attributes, _softmax  # noqa: E402

OUT = C.OUT / 'tagger'
OUT.mkdir(parents=True, exist_ok=True)
PHOTOS = C.ROOT / 'data' / 'barcelona_photos'
df = pd.read_csv(C.ROOT / 'data' / 'barcelona_photos.csv')
missing = [r for r in df.itertuples() if not (PHOTOS / r.file).exists()]
if missing:
    print(f'downloading {len(missing)} listed photos that are not in the repository...')
    for r in missing:
        try:
            im = Image.open(io.BytesIO(urllib.request.urlopen(r.url, timeout=180).read())).convert('RGB')
            im.thumbnail((640, 640))
            im.save(PHOTOS / r.file, quality=88)
            time.sleep(1)                                     # polite to Flickr
        except Exception as e:
            print(f'  could not download {r.file}: {type(e).__name__}')
df = df[[(PHOTOS / f).exists() for f in df['file']]].reset_index(drop=True)
paths = [str(PHOTOS / f) for f in df['file']]
attrs = json.loads((C.OUT / 'trends' / 'meta.json').read_text())['attributes']

emb = FashionEmbedder()

# 1. Screen: the largest person must fill a fair part of the photo, and the tagger must see
#    a clothed person rather than anything else
SCREEN = ['a street style photo of a fully clothed person', 'a nude or topless person',
          'a person in underwear', 'a statue or sculpture', 'a painting, drawing or poster',
          'an object or landscape with no people']
crops, person_share = [], []
for p in paths:
    im = Image.open(p).convert('RGB')
    crop, box = crop_person(im)
    crops.append(crop)
    person_share.append(0 if box is None else (box[2] - box[0]) * (box[3] - box[1]) / (im.width * im.height))
whole = emb.images(paths, crop=False)
screen = _softmax(emb.scale() * whole @ emb.texts(SCREEN).T)
df['person_share'] = np.round(person_share, 3)
df['p_clothed'] = screen[:, 0].round(3)
df['p_unsuitable'] = (screen[:, 1] + screen[:, 2]).round(3)
df['suitable'] = ((df.person_share >= 0.04) & (screen.argmax(1) == 0) & (df.p_unsuitable < 0.05)
                  & ~df.file.isin(C.EXCLUDE_PHOTOS))
df.to_csv(OUT / 'barcelona_screen.csv', index=False)
print(f'{df.suitable.sum()} of {len(df)} photos kept after screening')

keep = df.suitable.to_numpy()
df = df[keep].reset_index(drop=True)
vecs = emb.images([c for c, k in zip(crops, keep) if k], crop=False)
np.savez(OUT / 'barcelona_embeddings.npz', vecs=vecs,
         paths=np.array([Path(p).name for p, k in zip(paths, keep) if k]))   # file names: portable

# 2. Agreement with GeoStyle's CNN on the kept photos
probs = trend_attributes(emb, vecs, attrs)
rows = []
for group in ['Colour', 'Pattern', 'Garment', 'Sleeves', 'Hat', 'Glasses', 'Jacket', 'Scarf']:
    truth = df[group].astype(str)
    if group in probs:                                   # yes/no attribute
        pred = np.where(probs[group] >= 0.5, 'Yes', 'No')
    else:
        keys = [k for k in probs if k.startswith(group + ': ')]
        pred = np.array([k.split(': ')[1] for k in keys])[np.argmax([probs[k] for k in keys], 0)]
        known = truth.isin([k.split(': ')[1] for k in keys])
        truth, pred = truth[known], pred[known.values]
    rows.append({'attribute': group, 'photos': len(truth),
                 'agreement': float(np.mean(pred == truth.values)),
                 'majority_guess': float(truth.value_counts(normalize=True).iloc[0])})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'barcelona_agreement.csv', index=False)
print(res.round(3).to_string(index=False))
