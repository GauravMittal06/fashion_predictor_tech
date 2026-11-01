"""Layer 2 end to end: build Barcelona series, train the LSTM, forecast from every past month,
backtest, and make the final rising / stable / fading calls. Saves everything to outputs/trends/.

    python scripts/build_trends.py            (about 2-4 minutes on the laptop GPU)
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C, forecast as F, geodata  # noqa: E402

OUT = C.OUT / 'trends'
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# 1. Monthly series
wide = geodata.barcelona_series()
attrs = [c for c in wide.columns if c != 'n_photos']
Y = wide[attrs].to_numpy(float).T                     # (n_series, n_months)
wide.to_csv(OUT / 'barcelona_monthly.csv')
print(f'{len(attrs)} attribute series, {Y.shape[1]} months '
      f'({wide.index[0]:%b %Y} to {wide.index[-1]:%b %Y}), {int(wide.n_photos.sum()):,} photos')

# 2. Plan C: train the LSTM on 44 other cities (Barcelona is never shown to it)
city_series, _ = geodata.city_style_series()
print(f'training LSTM on {len(city_series)} city-style series')
model, hist, info = F.train_lstm(city_series)
torch.save(model.state_dict(), OUT / 'lstm.pt')
hist.to_csv(OUT / 'lstm_history.csv', index=False)
print(f'LSTM trained: {len(hist)} epochs, best val loss {hist.val_loss.min():.4f}')

# 3. Forecast from every start month that has two years of history behind it
cutoffs = list(range(C.MIN_HISTORY, Y.shape[1] + 1))  # the last one forecasts beyond the data
base = {'snaive': F.snaive, 'holt': F.holt, 'cycle': F.cycle,
        'chronos2': F.chronos2, 'lstm': F.lstm_forecaster(model)}
preds = F.forecast_all_cutoffs(Y, base, cutoffs)
preds['chronos2+holt'] = (preds['chronos2'] + preds['holt']) / 2          # A + B
preds['chronos2+lstm'] = (preds['chronos2'] + preds['lstm']) / 2          # A + C
np.savez_compressed(OUT / 'preds.npz', cutoffs=np.array(cutoffs), names=np.array(list(preds)),
                    **{k: v for k, v in preds.items()})

# 4. Backtest: every start month with a full 6-month truth, and the most recent 12 of them
detail = F.score(Y, preds, cutoffs)
detail['attribute'] = [attrs[s] for s in detail['series']]
detail.to_csv(OUT / 'backtest_detail.csv', index=False)
full = sorted(detail.cutoff.unique())
recent = full[-C.BACKTEST_CUTOFFS:]
summ_all = F.summarise(detail)
summ_recent = F.summarise(detail[detail.cutoff.isin(recent)])
summ_all.to_csv(OUT / 'backtest_all.csv')
summ_recent.to_csv(OUT / 'backtest_recent.csv')
pd.set_option('display.width', 160)
print('\nBacktest, all', len(full), 'start months:\n', summ_all.round(4))
print('\nBacktest, most recent', len(recent), 'start months:\n', summ_recent.round(4))

# 5. Final calls for the 6 months after the data ends: Chronos-2 calls, the LSTM confirms
last = len(cutoffs) - 1
rows = []
for s, a in enumerate(attrs):
    ca = F.call(Y[s], preds['chronos2'][last, s])
    cc = F.call(Y[s], preds['lstm'][last, s])
    cb = F.call(Y[s], preds['chronos2+lstm'][last, s])
    rows.append({'attribute': a, 'verdict': F.combine_models(ca['call'], cc['call']),
                 'chronos2': ca['call'], 'lstm': cc['call'], 'change': cb['change'],
                 'low': cb['low'], 'mid': cb['mid'], 'high': cb['high'], 'last_year': cb['last_year']})
calls = pd.DataFrame(rows)
calls.to_csv(OUT / 'calls.csv', index=False)
print('\nCalls for', (wide.index[-1] + pd.offsets.MonthBegin()).strftime('%b %Y'), '+ 5 months:')
print(calls[calls.verdict != 'stable'][['attribute', 'verdict', 'chronos2', 'lstm', 'change']].to_string())

meta = {'city': C.CITY, 'country': C.COUNTRY_NAME, 'attributes': attrs,
        'months': [d.strftime('%Y-%m') for d in wide.index], 'cutoffs': cutoffs,
        'recent_cutoffs': [int(c) for c in recent], 'photos': int(wide.n_photos.sum()),
        'lstm': {**info, 'epochs': len(hist), 'val_loss': float(hist.val_loss.min()),
                 'series': len(city_series)},
        'seconds': round(time.time() - t0, 1)}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
print(f'\ndone in {meta["seconds"]} s -> {OUT}')
