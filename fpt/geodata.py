"""Layer 2, step 1: turn the public GeoStyle research data into monthly trend series.

  Barcelona (Spain)  : 97k dated Flickr photos, 2013-2018, each tagged by GeoStyle's CNN with
                       12 clothing attributes -> monthly share of people wearing each attribute.
  44 world cities    : weekly popularity of 50 styles (From Paris to Berlin, CVPR 2020)
                       -> monthly series used only to train the Plan C LSTM.

Sources: http://geostyle.cs.cornell.edu/ and https://www.cs.utexas.edu/~ziad/fashion_influence.html
"""
import datetime as dt
import importlib
import pickle
import numpy as np
import pandas as pd
from . import config as C


class _NumpyOnly(pickle.Unpickler):
    """The research files are pickles; only numpy arrays are allowed to load, nothing executable."""

    def find_class(self, module, name):
        if module.startswith('numpy') and name in ('_reconstruct', 'ndarray', 'dtype'):
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def _load(path):
    with open(path, 'rb') as f:
        return _NumpyOnly(f, encoding='latin1').load()


def _week_to_month(key):
    """'2013_27' -> first day of the month holding that week's Thursday (ISO convention)."""
    year, week = map(int, key.split('_'))
    thursday = dt.date.fromisocalendar(year, min(max(week, 1), 52), 4)
    return pd.Timestamp(thursday.year, thursday.month, 1)


def barcelona_series():
    """Wide monthly table: one column per attribute ('Colour: Black', 'Hat', ...), plus n_photos."""
    meta = _load(C.DATA_RAW / 'barcaflickr_metadata.pkl')
    attrs, cats = meta['attributes'], meta['categories']
    weeks = meta['classifications'][C.CITY]
    frames = []
    for key, preds in weeks.items():                  # preds: (n_people, 12) class indices
        if len(preds):
            frames.append(pd.DataFrame(preds, columns=attrs).assign(month=_week_to_month(key)))
    people = pd.concat(frames, ignore_index=True)

    cols = {}
    for a, classes in zip(attrs, cats):
        name = C.GEO_ATTR_NAMES[a]
        if classes == ['No', 'Yes']:                  # yes/no attribute: share wearing it
            cols[name] = people[a] == 1
        else:
            for i, cls in enumerate(classes):
                cols[f'{name}: {cls}'] = people[a] == i
    onehot = pd.DataFrame(cols).assign(month=people['month'])
    g = onehot.groupby('month')
    wide = g.mean()
    wide['n_photos'] = g.size()
    wide = wide[wide['n_photos'] >= C.MIN_PHOTOS_PER_MONTH].asfreq('MS')
    wide = wide.interpolate(limit=2)                  # fill at most 2 missing months in a row
    keep = [c for c in wide.columns if c == 'n_photos' or wide[c].mean() >= C.MIN_MEAN_SHARE]
    return wide[keep]


def city_style_series():
    """List of monthly arrays, one per (city, style): 44 cities x 50 styles, mid-2013 to 2016."""
    t = _load(C.DATA_RAW / 'seasonal' / 'trends_styles50_cities44.pkl')
    months = pd.DatetimeIndex([_week_to_month(k) for k in t['time_range']])
    out, names = [], []
    for city, traj in zip(t['units_list'], t['units_traj']):
        monthly = pd.DataFrame(traj, index=months).groupby(level=0).mean()
        for s in monthly.columns:
            out.append(monthly[s].to_numpy(float))
            names.append(f'{city} / style {s}')
    return out, names
