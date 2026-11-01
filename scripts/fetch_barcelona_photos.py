"""Download a small, dated sample of the public Barcelona Flickr photos listed in GeoStyle.

Only photos with exactly one detected person are taken, so GeoStyle's labels for that person are
unambiguous. Photos are resized to 640 px and kept locally (never republished), as in the synopsis.

    python scripts/fetch_barcelona_photos.py [per_month=8] [months=24] [--offline]
"""
import hashlib
import io
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C, geodata  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
PER_MONTH = int(ARGS[0]) if len(ARGS) > 0 else 8
MONTHS = int(ARGS[1]) if len(ARGS) > 1 else 24
OUT = C.ROOT / 'data' / 'barcelona_photos'
OUT.mkdir(parents=True, exist_ok=True)

meta = geodata._load(C.DATA_RAW / 'barcaflickr_metadata.pkl')
attrs, cats = meta['attributes'], meta['categories']
rows = []
for key, preds in meta['classifications'][C.CITY].items():
    urls = [str(u) for u in meta['urls'][C.CITY][key]]
    counts = pd.Series(urls).value_counts()
    for u, p in zip(urls, preds):
        if counts[u] == 1:
            rows.append({'url': u, 'month': geodata._week_to_month(key),
                         **{C.GEO_ATTR_NAMES[a]: cats[j][int(p[j])] for j, a in enumerate(attrs)}})
df = pd.DataFrame(rows)
months = sorted(df.month.unique())[-MONTHS:]
random.seed(C.SEED)
pick = pd.concat([g.sample(min(len(g), PER_MONTH + 3), random_state=C.SEED)
                  for m, g in df[df.month.isin(months)].groupby('month')])


def fetch(r):
    """Polite download: Flickr answers 429 (too many requests) when hit fast, so wait and retry."""
    path = OUT / f"{r.month:%Y-%m}_{hashlib.md5(r.url.encode()).hexdigest()[:8]}.jpg"
    if path.exists():
        return str(path.name)
    if '--offline' in sys.argv:                   # only list photos already downloaded
        return None
    for attempt in range(6):
        try:
            data = urllib.request.urlopen(r.url, timeout=180).read()
            im = Image.open(io.BytesIO(data)).convert('RGB')
            im.thumbnail((640, 640))
            im.save(path, quality=88)
            time.sleep(1)
            return str(path.name)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return None
            time.sleep(20 * (attempt + 1))
        except Exception:
            return None
    return None


with ThreadPoolExecutor(2) as ex:
    pick['file'] = list(ex.map(fetch, pick.itertuples()))
pick = pick.dropna(subset=['file'])
pick = pick.groupby('month').head(PER_MONTH)            # spares cover dead links
pick.to_csv(C.ROOT / 'data' / 'barcelona_photos.csv', index=False)
print(f'{len(pick)} photos from {len(months)} months -> {OUT}')
