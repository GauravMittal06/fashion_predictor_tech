"""Download the public research data the prototype uses (run once; about 40 MB).

  GeoStyle (Cornell, ICCV 2019)        : barcaflickr_metadata.pkl - 96,965 dated Barcelona street photos,
                                         each person tagged with 12 clothing attributes
  From Paris to Berlin (CVPR 2020)     : trends of 50 styles in 44 cities - used to pre-train the LSTM

Files already present are skipped.

    python scripts/download_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402

FILES = {
    'barcaflickr_metadata.pkl': 'http://geostyle.cs.cornell.edu/static/data/barcaflickr_metadata.pkl',
    'geostyle_readme.txt': 'http://geostyle.cs.cornell.edu/static/data/Readme.txt',
    'trends_seasonal.zip': 'https://www.cs.utexas.edu/~ziad/data/city_influence/trends_seasonal.zip',
    'trends_deseasonalized.zip': 'https://www.cs.utexas.edu/~ziad/data/city_influence/trends_deseasonalized.zip',
}
C.DATA_RAW.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    path = C.DATA_RAW / name
    if path.exists():
        print(f'have  {name}')
        continue
    print(f'get   {name} ...', flush=True)
    urllib.request.urlretrieve(url, path)
for name, folder in (('trends_seasonal.zip', 'seasonal'), ('trends_deseasonalized.zip', 'deseasonal')):
    if not (C.DATA_RAW / folder).exists():
        zipfile.ZipFile(C.DATA_RAW / name).extractall(C.DATA_RAW / folder)
print('research data ready in', C.DATA_RAW)
