"""Layer 4: one self-contained HTML trend report for a brand (country + season)."""
import base64
import html
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from . import config as C  # noqa: E402

VERDICT_COLOUR = {'confirmed rising': '#1a7f4b', 'rising': '#1a7f4b', 'watch': '#b7791f', 'stable': '#5a6472',
                  'fading': '#b42318', 'confirmed fading': '#b42318'}


def _png(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _jpg(im):
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def chart(history, band, title):
    """history: monthly pd.Series; band: (h, 3) forecast for the months after it."""
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.plot(history.index, history.values * 100, color='#2f5d8a', lw=1.6, label='share of people')
    future = pd.date_range(history.index[-1] + pd.offsets.MonthBegin(), periods=len(band), freq='MS')
    ax.fill_between(future, band[:, 0] * 100, band[:, 2] * 100, color='#e3a33b', alpha=0.3, label='low to high')
    ax.plot(future, band[:, 1] * 100, color='#d08415', lw=1.8, label='forecast')
    ax.set_title(title, fontsize=9, loc='left')
    ax.set_ylabel('% of people', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=3)
    return _png(fig)


def build_report(season, calls, charts, backtest, brief=None, prompt='', concepts=(), board=None,
                 method_note=''):
    order = {'confirmed rising': 0, 'rising': 1, 'watch': 2, 'stable': 3, 'fading': 4, 'confirmed fading': 5}
    calls = calls.assign(_o=calls.verdict.map(order), _a=-calls.change.abs()).sort_values(['_o', '_a'])
    trs = ''.join(
        f"<tr><td>{html.escape(r.attribute)}</td>"
        f"<td><b style='color:{VERDICT_COLOUR[r.verdict]}'>{r.verdict}</b>"
        f"{(' <small>leaning ' + ('up' if r.change > 0 else 'down') + '</small>') if r.verdict == 'stable' and abs(r.change) >= 0.05 else ''}</td>"
        f"<td>{r.chronos2}</td><td>{r.lstm}</td><td>{r.change * 100:+.0f}%</td>"
        f"<td>{r.last_year * 100:.1f}%</td><td>{r.low * 100:.1f} – {r.high * 100:.1f}%</td></tr>"
        for r in calls.itertuples())
    bt = backtest.copy()
    bt_rows = ''.join(
        f"<tr><td>{html.escape(n)}</td><td>{r.mae * 100:.2f}</td><td>{r.skill_vs_last_year * 100:+.1f}%</td>"
        f"<td>{r.direction_ok * 100:.0f}%</td><td>{r.coverage * 100:.0f}%</td>"
        f"<td>{int(r.calls_correct)}/{int(r.calls_made)}</td></tr>" for n, r in bt.iterrows())
    imgs = ''.join(f'<img src="data:image/png;base64,{c}">' for c in charts)
    con = ''
    if brief:
        con += (f"<p><b>Design brief:</b> {html.escape(', '.join(brief['attributes']) or 'none')}"
                f"{' · avoid ' + html.escape(', '.join(brief['avoid'])) if brief['avoid'] else ''}</p>"
                f"<p><b>Prompt:</b> <i>{html.escape(prompt)}</i></p>")
    if concepts:
        con += '<div class="grid">' + ''.join(
            f'<figure><img src="data:image/jpeg;base64,{_jpg(im)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
            for im, cap in concepts) + '</div>'
    if board is not None:
        con += f'<p><b>Concept board</b> (closest real photos: our dataset and Barcelona):</p><img class="board" src="data:image/jpeg;base64,{_jpg(board)}">'
    first = ''
    if start:
        rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in r) + '</tr>' for r in start['rows'])
        first = (f'<h2>Starting photo</h2><div class="start"><img src="data:image/jpeg;base64,{_jpg(start["photo"])}">'
                 f'<table><tr><th>Attribute</th><th>In the photo</th><th>Trend</th><th>In the design</th></tr>{rows}</table></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Trend report {C.COUNTRY_NAME} {html.escape(season)}</title>
<style>body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:960px;margin:28px auto;padding:0 16px;color:#1c2532;line-height:1.45}}
h1{{margin-bottom:4px}}.sub{{color:#5a6472;margin-top:0}}table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0 18px}}
td,th{{border-bottom:1px solid #d5dde6;padding:6px 8px;text-align:left}}th{{background:#eaf1f8}}
img{{max-width:49%}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.grid img{{max-width:100%}}
figure{{margin:0}}figcaption{{font-size:11.5px;color:#5a6472}}img.board{{max-width:60%}}
.start{{display:flex;gap:16px;align-items:flex-start}}.start img{{max-width:220px}}
.note{{background:#f5f7fa;border-left:4px solid #2f5d8a;padding:10px 14px;font-size:13.5px}}</style></head><body>
<h1>Fashion trend report: {C.COUNTRY_NAME} ({C.CITY}), {html.escape(season)}</h1>
<p class="sub">Fashion Prediction Technology · prototype · Gaurav Chandra Mittal, Hari Krishna Tiwari · Guide: Dr. D.P. Sharma</p>
<div class="note">Each attribute's share of people wearing it is forecast for the next {C.HORIZON} months and compared with
the same months last year. The call is made by <b>Chronos-2</b> (the most reliable forecaster in the backtest);
<b>confirmed</b> = our LSTM makes the same call; <b>watch</b> = only the LSTM sees a move. Rising needs the middle forecast at least {int(C.CALL_THRESHOLD * 100)}% above last year and even
the low forecast above it.</div>
<h2>1. Trend calls</h2>
<table><tr><th>Attribute</th><th>Verdict</th><th>Chronos-2</th><th>LSTM</th><th>Change vs last year</th><th>Last year</th><th>Forecast range</th></tr>{trs}</table>
<h2>2. Evidence</h2>{imgs}
<h2>3. How accurate is this? (backtest on hidden past months)</h2>
<table><tr><th>Forecaster</th><th>Error (MAE, % points)</th><th>Better than "same as last year"</th><th>Direction right</th><th>Truth inside range</th><th>Rising/fading calls right</th></tr>{bt_rows}</table>
<h2>4. Design concepts</h2>{con or '<p>No concepts generated yet.</p>'}
<h2>5. Method and limits</h2><p>{html.escape(method_note)}</p></body></html>"""


def search_chart(history, band, title):
    """history: monthly search interest; band: (h, 3) forecast in the same units."""
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.plot(history.index, history.values, color='#2f5d8a', lw=1.6, label='Google search interest')
    future = pd.date_range(history.index[-1] + pd.offsets.MonthBegin(), periods=len(band), freq='MS')
    ax.fill_between(future, band[:, 0], band[:, 2], color='#e3a33b', alpha=0.3, label='low to high')
    ax.plot(future, band[:, 1], color='#d08415', lw=1.8, label='forecast')
    ax.set_title(title, fontsize=9, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=3)
    return _png(fig)


def build_brand_report(place, season, calls, charts, trust_rows, brief=None, prompt='', concepts=(), board=None,
                       method_note='', start=None):
    """Brand report from the frozen search forecaster. calls: term, english, call, change, low_chg, high_chg.
    trust_rows: list of (what, result) pairs shown as 'How far to trust this'.
    start: optional {'photo': PIL image, 'rows': [(attribute, photo, trend, design)]} for the pipeline's starting photo."""
    order = {'rising': 0, 'stable': 1, 'fading': 2}
    calls = calls.assign(_o=calls.call.map(order), _a=-calls.change).sort_values(['_o', '_a'])
    trs = ''.join(
        f"<tr><td>{html.escape(r.english)} <small>({html.escape(r.term)})</small></td>"
        f"<td><b style='color:{VERDICT_COLOUR.get(r.call, '#5a6472')}'>{r.call}</b>"
        f"{(' <small>leaning ' + ('up' if r.change > 0 else 'down') + '</small>') if r.call == 'stable' and abs(r.change) >= 0.05 else ''}</td>"
        f"<td>{r.change * 100:+.0f}%</td><td>{r.low_chg * 100:+.0f}% to {r.high_chg * 100:+.0f}%</td></tr>"
        for r in calls.itertuples())
    trust = ''.join(f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>' for a, b in trust_rows)
    imgs = ''.join(f'<img src="data:image/png;base64,{c}">' for c in charts)
    con = ''
    if brief:
        con += (f"<p><b>Design brief</b> (from the rising searches: {html.escape(', '.join(brief['evidence']))})</p>"
                f"<p><b>Prompt:</b> <i>{html.escape(prompt)}</i></p>")
    if concepts:
        con += '<div class="grid">' + ''.join(
            f'<figure><img src="data:image/jpeg;base64,{_jpg(im)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
            for im, cap in concepts) + '</div>'
    if board is not None:
        con += f'<p><b>Concept board</b> (closest real outfit photos from our dataset):</p><img class="board" src="data:image/jpeg;base64,{_jpg(board)}">'
    first = ''
    if start:
        rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in r) + '</tr>' for r in start['rows'])
        first = (f'<h2>Starting photo</h2><div class="start"><img src="data:image/jpeg;base64,{_jpg(start["photo"])}">'
                 f'<table><tr><th>Attribute</th><th>In the photo</th><th>Trend</th><th>In the design</th></tr>{rows}</table></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Trend report {html.escape(place)} {html.escape(season)}</title>
<style>body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:960px;margin:28px auto;padding:0 16px;color:#1c2532;line-height:1.45}}
h1{{margin-bottom:4px}}.sub{{color:#5a6472;margin-top:0}}table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0 18px}}
td,th{{border-bottom:1px solid #d5dde6;padding:6px 8px;text-align:left}}th{{background:#eaf1f8}}
img{{max-width:49%}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.grid img{{max-width:100%}}
figure{{margin:0}}figcaption{{font-size:11.5px;color:#5a6472}}img.board{{max-width:60%}}
.start{{display:flex;gap:16px;align-items:flex-start}}.start img{{max-width:220px}}
.note{{background:#f5f7fa;border-left:4px solid #2f5d8a;padding:10px 14px;font-size:13.5px}}</style></head><body>
<h1>Fashion trend report: {html.escape(place)}, {html.escape(season)}</h1>
<p class="sub">Fashion Prediction Technology · MVP · Gaurav Chandra Mittal, Hari Krishna Tiwari · Guide: Dr. D.P. Sharma</p>
<div class="note">Each search's interest over the next {C.HORIZON} months is forecast by the frozen forecaster and compared
with the same months last year. <b>Rising</b> needs the middle forecast at least {int(C.CALL_THRESHOLD * 100)}% above
last year and even the low forecast above it (fading mirrored); <b>leaning</b> means a move of 5% or more that is not
yet certain.</div>
{first}
<h2>1. Trend calls</h2>
<table><tr><th>Search</th><th>Call</th><th>Change vs same months last year</th><th>Range</th></tr>{trs}</table>
<h2>2. How far to trust this</h2><table>{trust}</table>
<h2>3. Evidence</h2>{imgs}
<h2>4. Design concepts</h2>{con or '<p>No concepts generated yet.</p>'}
<h2>5. Method and limits</h2><p>{html.escape(method_note)}</p></body></html>"""
