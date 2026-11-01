"""Layer 2, final evaluation: freeze the forecaster on data up to Dec 2023, then REPLAY 2024-2026.

Leakage rules
  1. Development = Spain's Google Trends up to Dec 2023 (plus the 44-city street-style trends).
     The LSTM is retrained on that, and the best forecaster is chosen on forecasts that END by Dec 2023.
  2. The choice is written to outputs/replay/frozen.json BEFORE any later month is scored.
  3. Replays: from every month Jan 2024 - May 2026 the frozen forecaster predicts the next 6 months
     using only earlier data; we then compare with what really happened.
  4. Italy is never used for any choice: its replay is the cleanest test of all.
Each search term is divided by its own development-period average, so terms can be scored together.

    python scripts/build_replays.py
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C, forecast as F, geodata  # noqa: E402

SRC = C.ROOT / 'data' / 'google_trends'
OUT = C.OUT / 'replay'
OUT.mkdir(parents=True, exist_ok=True)
DEV_END = pd.Timestamp('2023-12-01')
MIN_DEV_MEAN = 3.0          # terms averaging below 3 (out of 100) before 2024 are too sparse to score
COUNTRIES = {'ES': ('Spain', sorted(glob.glob(str(SRC / 'ES_*.csv'))), SRC / 'terms.csv'),
             'IT': ('Italy', sorted(glob.glob(str(SRC / 'IT' / 'IT_*.csv'))), SRC / 'IT' / 'terms.csv')}


def load(files, terms_file):
    frames = [pd.read_csv(f, dtype=str).set_index('month').replace('<1', '0.5').astype(float) for f in files]
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    wide.index = pd.to_datetime(wide.index)
    terms = pd.read_csv(terms_file).set_index('term')
    wide = wide[[t for t in terms.index if t in wide.columns]]
    dev_mean = wide[wide.index <= DEV_END].mean()
    keep = dev_mean[dev_mean >= MIN_DEV_MEAN].index
    return wide[keep], terms.loc[keep], dev_mean[keep].to_numpy()


data = {}
for cc, (name, files, tf) in COUNTRIES.items():
    if files:
        wide, terms, scale = load(files, tf)
        data[cc] = {'name': name, 'wide': wide, 'terms': terms, 'Y': wide.to_numpy().T / scale[:, None], 'scale': scale}
        print(f'{name}: {wide.shape[1]} terms, {wide.index[0]:%b %Y} - {wide.index[-1]:%b %Y}')

es = data['ES']
months = es['wide'].index
dev_n = int((months <= DEV_END).sum())                 # months up to Dec 2023

# ---------------- 1. Development (Spain up to Dec 2023 only) ----------------
city, _ = geodata.city_style_series()
print('retraining the LSTM on 44-city street-style trends + Spain searches up to Dec 2023')
lstm_mix, hist, info = F.train_lstm(city + list(es['Y'][:, :dev_n]), log=lambda s: None)
torch.save(lstm_mix.state_dict(), OUT / 'lstm_mix.pt')
hist.to_csv(OUT / 'lstm_mix_history.csv', index=False)
lstm_geo = F.QuantileLSTM().to(F.DEVICE)
lstm_geo.load_state_dict(torch.load(C.OUT / 'trends' / 'lstm.pt', map_location=F.DEVICE))
lstm_geo.eval()

base = {'snaive': F.snaive, 'holt': F.holt, 'cycle': F.cycle, 'chronos2': F.chronos2,
        'lstm': F.lstm_forecaster(lstm_geo), 'lstm_mix': F.lstm_forecaster(lstm_mix)}
COMBOS = {'chronos2+holt': ('chronos2', 'holt'), 'chronos2+lstm_mix': ('chronos2', 'lstm_mix'),
          'chronos2+snaive': ('chronos2', 'snaive')}

dev_cut = list(range(C.MIN_HISTORY, dev_n - C.HORIZON + 1))   # forecasts that end by Dec 2023
P = F.forecast_all_cutoffs(es['Y'][:, :dev_n], base, dev_cut, log=lambda s: None)
for k, (a, b) in COMBOS.items():
    P[k] = (P[a] + P[b]) / 2
dev = F.summarise(F.score(es['Y'][:, :dev_n], P, dev_cut))
chosen = dev.index[0]
frozen = {'chosen': chosen, 'development_until': DEV_END.strftime('%Y-%m'), 'development_start_months': len(dev_cut),
          'lstm_mix': {**info, 'epochs': len(hist)}, 'decided_before_replays': True}
(OUT / 'frozen.json').write_text(json.dumps(frozen, indent=1))
dev.to_csv(OUT / 'development.csv')
print(f'\nDEVELOPMENT (Spain, {len(dev_cut)} start months, forecasts ending by Dec 2023):')
cols = ['mae', 'skill_vs_last_year', 'direction_ok', 'coverage', 'calls_made', 'calls_correct']
print(dev[cols].round(3).to_string())
print(f'FROZEN: {chosen}')

# ---------------- 2. Replays 2024-2026 with the frozen setup ----------------
used = sorted({chosen, 'snaive', 'chronos2', 'lstm_mix', 'holt'} | set(COMBOS.get(chosen, ())))
result = {}
for cc, d in data.items():
    Y, n = d['Y'], d['Y'].shape[1]
    cut = list(range(dev_n, n + 1))                     # Jan 2024 ... one past the data (final forecast)
    preds = F.forecast_all_cutoffs(Y, {k: base[k] for k in used if k in base}, cut, log=lambda s: None)
    for k, (a, b) in COMBOS.items():
        if a in preds and b in preds:
            preds[k] = (preds[a] + preds[b]) / 2
    detail = F.score(Y, preds, cut)
    detail['term'] = [d['wide'].columns[s] for s in detail['series']]
    summ = F.summarise(detail)
    summ.to_csv(OUT / f'replay_{cc}.csv')
    detail.to_csv(OUT / f'replay_{cc}_detail.csv', index=False)
    np.savez_compressed(OUT / f'preds_{cc}.npz', cutoffs=np.array(cut), names=np.array(list(preds)),
                        scale=d['scale'], **preds)
    d['wide'].to_csv(OUT / f'search_{cc}.csv')
    d['terms'].to_csv(OUT / f'terms_{cc}.csv')
    # final calls: 6 months after the data, by the frozen forecaster
    rows = []
    for s, t in enumerate(d['wide'].columns):
        r = F.call(Y[s], preds[chosen][-1, s])
        rows.append({'term': t, 'english': d['terms'].loc[t, 'english'], 'attribute': d['terms'].loc[t, 'attribute'],
                     **{k: r[k] for k in ('call', 'change', 'low', 'mid', 'high', 'last_year')}})
    pd.DataFrame(rows).to_csv(OUT / f'calls_{cc}.csv', index=False)
    result[cc] = summ
    print(f'\nREPLAY {d["name"]} ({detail.cutoff.nunique()} start months, Jan 2024 - '
          f'{months[0] + pd.DateOffset(months=int(detail.cutoff.max())):%b %Y}):')
    print(summ[cols].round(3).to_string())

meta = {'countries': {cc: d['name'] for cc, d in data.items()}, 'months': [m.strftime('%Y-%m') for m in months],
        'dev_n': dev_n, 'frozen': frozen}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
