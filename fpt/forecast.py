"""Layer 2, step 2: forecast each monthly series 6 months ahead, test on the past,
and turn forecasts into rising / stable / fading calls.

Every forecaster takes Y of shape (n_series, n_months) and returns (n_series, horizon, 3):
low, middle, high (10%, 50%, 90%).
  Plan A  chronos2  : Chronos-2 (Amazon, 120M-parameter transformer), pretrained, zero-shot
  Plan B  snaive    : same month last year
          holt      : Holt-Winters exponential smoothing
          cycle     : straight-line trend + yearly sine wave (inspired by GeoStyle)
  Plan C  lstm      : one quantile LSTM trained by us across 2,200 city-style series

Ported from the team's forecast.py; the LSTM is written in PyTorch so it trains on the laptop GPU.
"""
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import config as C

Z90 = 1.2816                                   # normal quantile for the 10% / 90% band
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _band(mid, sd):
    mid = np.asarray(mid, float)
    return np.stack([mid - Z90 * sd, mid, mid + Z90 * sd], 1)


def _per_series(f):
    """Lift a one-series forecaster f(y, h) -> (h, 3) to the batch interface."""
    return lambda Y, h: np.stack([f(np.asarray(y, float), h) for y in Y])


# ---------------- Plan B: seasonal baselines ----------------
def _snaive(y, h):
    mid = np.array([y[len(y) - C.SEASON + (i % C.SEASON)] for i in range(h)])
    sd = np.std(y[C.SEASON:] - y[:-C.SEASON])
    return _band(mid, sd)


def _holt(y, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ExponentialSmoothing(y, trend='add', damped_trend=True, seasonal='add',
                                   seasonal_periods=C.SEASON).fit()
    return _band(fit.forecast(h), np.std(fit.resid))


def _cycle(y, h):
    """Least-squares fit of a + b*t + c*sin(2*pi*t/12) + d*cos(2*pi*t/12)."""
    t = np.arange(len(y) + h)
    w = 2 * np.pi * t / C.SEASON
    X = np.stack([np.ones_like(t), t, np.sin(w), np.cos(w)], 1).astype(float)
    coef, *_ = np.linalg.lstsq(X[:len(y)], y, rcond=None)
    resid = y - X[:len(y)] @ coef
    return _band(X[len(y):] @ coef, np.std(resid))


snaive, holt, cycle = _per_series(_snaive), _per_series(_holt), _per_series(_cycle)


# ---------------- Plan A: Chronos-2 (zero-shot) ----------------
_chronos2 = None


def chronos2(Y, h):
    """All series forecast in one batch on the GPU; no training at all."""
    global _chronos2
    from chronos import Chronos2Pipeline
    if _chronos2 is None:
        _chronos2 = Chronos2Pipeline.from_pretrained(C.CHRONOS2_MODEL, device_map=DEVICE)
    Y = np.asarray(Y, float)
    stamps = pd.date_range('2000-01-01', periods=Y.shape[1], freq='MS')
    df = pd.concat([pd.DataFrame({'item_id': i, 'timestamp': stamps, 'target': y})
                    for i, y in enumerate(Y)], ignore_index=True)
    out = _chronos2.predict_df(df, prediction_length=h, quantile_levels=C.QUANTILES, freq='MS')
    qcols = [str(q) for q in C.QUANTILES]
    out = out.sort_values(['item_id', 'timestamp'])
    q = out[qcols].to_numpy().reshape(len(Y), h, 3)
    return np.sort(q, axis=2)                     # low <= mid <= high


# ---------------- Plan C: quantile LSTM trained across many series ----------------
class QuantileLSTM(nn.Module):
    """12 months in -> LSTM(64) -> Dense(18) -> 6 months x (low, middle, high)."""

    def __init__(self, units=C.LSTM_UNITS, n_out=C.LSTM_OUT):
        super().__init__()
        self.lstm = nn.LSTM(1, units, batch_first=True)
        self.head = nn.Linear(units, n_out * 3)
        self.n_out = n_out

    def forward(self, x):                          # x: (batch, 12, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).view(-1, self.n_out, 3)


def lstm_windows(series_list, n_in=C.LSTM_IN, n_out=C.LSTM_OUT):
    """Cut every series into (12 in, 6 out) windows; each window divided by its input mean."""
    X, Y = [], []
    for y in series_list:
        y = np.asarray(y, float)
        y = y[~np.isnan(y)]
        for i in range(len(y) - n_in - n_out + 1):
            x = y[i:i + n_in]
            scale = x.mean() if x.mean() > 1e-9 else 1.0
            X.append(x / scale)
            Y.append(y[i + n_in:i + n_in + n_out] / scale)
    return np.array(X, 'float32')[..., None], np.array(Y, 'float32')


def pinball(pred, target, qs=torch.tensor(C.QUANTILES)):
    """Quantile (pinball) loss for the 10%, 50% and 90% levels."""
    err = target.unsqueeze(-1) - pred
    qs = qs.to(pred.device)
    return torch.maximum(qs * err, (qs - 1) * err).mean()


def train_lstm(series_list, epochs=C.LSTM_EPOCHS, patience=C.LSTM_PATIENCE, log=print):
    """One LSTM for all series, early-stopped on 10% held-out windows. Returns (model, history)."""
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    X, Y = lstm_windows(series_list)
    idx = rng.permutation(len(X))
    n_val = len(X) // 10
    tr, va = idx[n_val:], idx[:n_val]
    Xt, Yt = torch.tensor(X[tr]).to(DEVICE), torch.tensor(Y[tr]).to(DEVICE)
    Xv, Yv = torch.tensor(X[va]).to(DEVICE), torch.tensor(Y[va]).to(DEVICE)
    model = QuantileLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, wait, history = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(len(Xt), device=DEVICE).split(256):
            opt.zero_grad()
            loss = pinball(model(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tl = pinball(model(Xt), Yt).item()
            vl = pinball(model(Xv), Yv).item()
        history.append({'epoch': ep + 1, 'train_loss': tl, 'val_loss': vl})
        log(f'epoch {ep + 1:3d}  train {tl:.4f}  val {vl:.4f}')
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history), {'windows_train': len(tr), 'windows_val': len(va)}


def lstm_forecaster(model):
    def f(Y, h):
        Y = np.asarray(Y, float)
        x = Y[:, -C.LSTM_IN:]
        scale = x.mean(1, keepdims=True)
        scale[scale < 1e-9] = 1.0
        with torch.no_grad():
            q = model(torch.tensor(x / scale, dtype=torch.float32, device=DEVICE)[..., None]).cpu().numpy()
        q = q * scale[:, :, None]
        return np.sort(q, axis=2)[:, :h]           # keep low <= middle <= high
    return f


def ensemble(*fs):
    """Average the bands of several forecasters (e.g. Plan A + Plan B)."""
    return lambda Y, h: np.mean([f(Y, h) for f in fs], axis=0)


# ---------------- Calls ----------------
def direction(change, threshold=C.CALL_THRESHOLD):
    """Three-way rule versus the same months last year."""
    return 'rising' if change >= threshold else 'fading' if change <= -threshold else 'stable'


def call(history, band, threshold=C.CALL_THRESHOLD):
    """Forecast band for the next h months compared with the same months last year.
    Rising needs the middle 10% above last year AND even the low forecast above it."""
    h = len(band)
    ly = float(np.mean(history[len(history) - C.SEASON:len(history) - C.SEASON + h]))
    lo, mid, hi = band.mean(0)
    change = (mid - ly) / ly if ly else np.nan
    if mid >= ly * (1 + threshold) and lo > ly:
        verdict = 'rising'
    elif mid <= ly * (1 - threshold) and hi < ly:
        verdict = 'fading'
    else:
        verdict = 'stable'
    return {'call': verdict, 'change': change, 'low': lo, 'mid': mid, 'high': hi, 'last_year': ly}


def combine_models(call_a, call_c):
    """Plan A (Chronos-2) makes the call: in the backtest its rising/fading calls were the most
    reliable. 'confirmed' when the Plan C LSTM makes the same call; an LSTM-only call is 'watch'."""
    if call_a != 'stable':
        return ('confirmed ' if call_c == call_a else '') + call_a
    if call_c != 'stable':
        return 'watch'
    return 'stable'


# ---------------- Backtest ----------------
def forecast_all_cutoffs(Y, forecasters, cutoffs, h=C.HORIZON, log=print):
    """For every forecast start month t in cutoffs, forecast from Y[:, :t] only.
    Returns {name: array (n_cutoffs, n_series, h, 3)}."""
    out = {}
    for name, f in forecasters.items():
        log(f'forecasting with {name} from {len(cutoffs)} start months')
        out[name] = np.stack([f(Y[:, :t], h) for t in cutoffs])
    return out


def score(Y, preds, cutoffs, h=C.HORIZON):
    """Backtest scores per forecaster and series, only for start months with a full truth."""
    rows = []
    for name, P in preds.items():
        for ci, t in enumerate(cutoffs):
            if t + h > Y.shape[1]:
                continue
            for s in range(Y.shape[0]):
                q, actual = P[ci, s], Y[s, t:t + h]
                ly = Y[s, t - C.SEASON:t - C.SEASON + h].mean()
                rows.append({
                    'forecaster': name, 'cutoff': t, 'series': s,
                    'mae': np.mean(np.abs(q[:, 1] - actual)),
                    'mape': np.mean(np.abs(q[:, 1] - actual) / np.maximum(np.abs(actual), 1e-9)) * 100,
                    'direction_ok': direction(q[:, 1].mean() / ly - 1) == direction(actual.mean() / ly - 1),
                    'coverage': np.mean((actual >= q[:, 0]) & (actual <= q[:, 2])),
                    'pred_call': call(Y[s, :t], q)['call'],
                    'true_dir': direction(actual.mean() / ly - 1),
                })
    return pd.DataFrame(rows)


def summarise(detail):
    """One row per forecaster: average errors, direction accuracy, range coverage, and how many
    rising/fading calls came true (precision of the calls a brand would act on)."""
    g = detail.groupby('forecaster')
    summ = g[['mae', 'mape', 'direction_ok', 'coverage']].mean()
    made = detail[detail['pred_call'] != 'stable']
    summ['calls_made'] = made.groupby('forecaster').size().reindex(summ.index).fillna(0).astype(int)
    hits = made[made['pred_call'] == made['true_dir']].groupby('forecaster').size()
    summ['calls_correct'] = hits.reindex(summ.index).fillna(0).astype(int)
    base = summ.loc['snaive', 'mae'] if 'snaive' in summ.index else np.nan
    summ['skill_vs_last_year'] = 1 - summ['mae'] / base
    return summ.sort_values('mae')
