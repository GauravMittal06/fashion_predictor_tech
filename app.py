"""Fashion Prediction Technology — MVP (Southern Europe: Spain, with Italy as an untouched test).

    streamlit run app.py

Pipeline page: one photo flows through every layer.
  1 Tagger      Faster R-CNN person crop + Marqo-FashionSigLIP / FashionCLIP + linear probe (frozen recipe)
  2 Forecaster  the photo's tags -> matching Google searches -> Chronos-2 + LSTM forecast (frozen on data to 2023)
  3 Brief       keep what is trending, swap what is fading
  4 Designer    SDXL-Turbo concepts, checked against 195 real photos
  5 Report      self-contained HTML trend report
Evidence page: how each layer was tested (replays 2024-26, cross-validation, Barcelona street photos).
"""
import hashlib
import io
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from fpt import config as C, forecast as F

st.set_page_config(page_title='Fashion Prediction Technology', page_icon='👗', layout='wide')
st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:1250px}
div[data-testid="stMetricValue"]{font-size:1.35rem}
.step{padding:6px 10px;border-radius:8px;font-size:.88rem;text-align:center;border:1px solid rgba(128,128,128,.25)}
.step.done{background:rgba(46,160,67,.12);border-color:rgba(46,160,67,.45)}
.step.wait{opacity:.55}
.tag{padding:8px 12px;border-radius:10px;border:1px solid rgba(128,128,128,.25)}
.tag .t{font-size:.8rem;margin-top:6px;padding-top:6px;border-top:1px solid rgba(128,128,128,.2)}
.tag .k{font-size:.78rem;opacity:.7}.tag .v{font-size:1.25rem;font-weight:600}.tag .s{font-size:.78rem;opacity:.7}
</style>""", unsafe_allow_html=True)

TR, TG, RP, DS = C.OUT / 'trends', C.OUT / 'tagger', C.OUT / 'replay', C.OUT / 'designer'
NAMES = {'chronos2': 'Chronos-2 (Plan A, pretrained transformer)', 'lstm': 'LSTM (Plan C, trained by us)',
         'snaive': 'Same month last year (Plan B)', 'holt': 'Holt-Winters (Plan B)',
         'cycle': 'Trend + yearly wave (Plan B)', 'chronos2+holt': 'Chronos-2 + Holt-Winters (A+B)',
         'chronos2+lstm': 'Chronos-2 + LSTM (A+C)', 'lstm_mix': 'LSTM retrained on search data (Plan C)',
         'chronos2+lstm_mix': 'Chronos-2 + retrained LSTM (A+C, frozen)',
         'chronos2+snaive': 'Chronos-2 + same month last year (A+B)'}
COLOURS = {'chronos2': '#d08415', 'lstm': '#7b3fb5', 'snaive': '#6b7785', 'holt': '#2a9d8f', 'cycle': '#8d99ae',
           'chronos2+holt': '#e76f51', 'chronos2+lstm': '#c2185b', 'lstm_mix': '#7b3fb5', 'chronos2+lstm_mix': '#c2185b',
           'chronos2+snaive': '#e76f51'}
STATUS = {'rising': '🟢 rising', 'leaning up': '↗️ leaning up', 'stable': '⚪ stable', 'leaning down': '↘️ leaning down',
          'fading': '🔴 fading', 'no data': '— no search data', 'not checked': '—'}
VERDICT_ICON = {'confirmed rising': '🟢 confirmed rising', 'rising': '🟢 rising', 'watch': '🟡 watch',
                'stable': '⚪ stable', 'fading': '🔴 fading', 'confirmed fading': '🔴 confirmed fading'}
KIND_LABEL = {'style': 'Style', 'fabric': 'Fabric', 'colour': 'Colour', 'motif': 'Print', 'print': 'Print'}


# ---------------- data and models (cached) ----------------
def _name(p):
    """File name from a stored path, whichever system wrote it."""
    return re.split(r'[\\/]', str(p))[-1]


def _stamp(*files):
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


@st.cache_data
def load_json(path, stamp):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


@st.cache_data
def load_replay(stamp):
    meta = json.loads((RP / 'meta.json').read_text())
    out = {}
    for cc in meta['countries']:
        z = np.load(RP / f'preds_{cc}.npz')
        out[cc] = {'wide': pd.read_csv(RP / f'search_{cc}.csv', index_col=0, parse_dates=True),
                   'terms': pd.read_csv(RP / f'terms_{cc}.csv', index_col=0),
                   'preds': {str(n): z[str(n)] for n in z['names']}, 'cutoffs': [int(c) for c in z['cutoffs']],
                   'scale': z['scale'], 'summary': pd.read_csv(RP / f'replay_{cc}.csv', index_col=0),
                   'calls': pd.read_csv(RP / f'calls_{cc}.csv')}
    return meta, out


@st.cache_data
def load_team(stamp):
    e = TG / 'team_embeddings.npz'
    if not e.exists():
        return [], None
    z = np.load(e)
    return [str(C.TEAM_DIR / _name(x)) for x in z['paths']], z['vecs']


@st.cache_data
def load_photos(stamp):
    f, e = C.ROOT / 'data' / 'barcelona_photos.csv', TG / 'barcelona_embeddings.npz'
    if not f.exists():
        return None, None
    df = pd.read_csv(f)
    df['path'] = [str(C.ROOT / 'data' / 'barcelona_photos' / x) for x in df['file']]
    vecs = None
    if e.exists():
        z = np.load(e)
        order = {_name(p): i for i, p in enumerate(z['paths'])}
        df = df[df.file.isin(order)].reset_index(drop=True)
        vecs = z['vecs'][[order[f] for f in df.file]]
    return df, vecs


@st.cache_data
def load_trends():
    meta = json.loads((TR / 'meta.json').read_text())
    wide = pd.read_csv(TR / 'barcelona_monthly.csv', index_col=0, parse_dates=True)
    z = np.load(TR / 'preds.npz')
    preds = {str(n): z[str(n)] for n in z['names']}
    return (meta, wide, preds, [int(c) for c in z['cutoffs']],
            pd.read_csv(TR / 'backtest_recent.csv', index_col=0), pd.read_csv(TR / 'backtest_all.csv', index_col=0),
            pd.read_csv(TR / 'lstm_history.csv'), pd.read_csv(TR / 'backtest_detail.csv'))


@st.cache_resource
def embedder():
    from fpt.tagger import FashionEmbedder
    return FashionEmbedder()


@st.cache_resource
def embedder_b():
    from fpt.tagger import FashionCLIPEmbedder
    return FashionCLIPEmbedder()


@st.cache_resource
def sdxl():
    from fpt.design import load_generator
    return load_generator()


@st.cache_resource
def gpu_lock():
    """One GPU job at a time across all tabs. SDXL fits in 6 GB by moving its parts on and off the GPU with hooks
    that every run removes and re-adds; two runs at once (two tabs, or a new photo while drawing) break each other."""
    import threading
    return threading.Lock()


def draw(prompt, n, seed):
    from fpt import design as D
    pipe = sdxl()
    try:
        return D.generate(prompt, n=n, seed=seed, pipe=pipe)
    except AttributeError:              # hooks left half-removed by an interrupted run: re-attach them, try once more
        pipe.enable_model_cpu_offload()
        return D.generate(prompt, n=n, seed=seed, pipe=pipe)


@st.cache_resource
def chronos_cpu():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(C.CHRONOS2_MODEL, device_map='cpu')


rmeta, R = load_replay(_stamp(RP / 'meta.json'))
FROZEN = rmeta['frozen']['chosen']
cv_meta = load_json(TG / 'cv_meta.json', _stamp(TG / 'cv_meta.json'))
dres = load_json(DS / 'results.json', _stamp(DS / 'results.json'))
team_paths, team_vecs = load_team(_stamp(TG / 'team_embeddings.npz'))
batch2 = pd.read_csv(C.BATCH2_LABELS, dtype=str).fillna('—') if C.BATCH2_LABELS.exists() else None
photos, photo_vecs = load_photos(_stamp(C.ROOT / 'data' / 'barcelona_photos.csv', TG / 'barcelona_embeddings.npz'))
CVS = (pd.read_csv(TG / 'cv_summary.csv', index_col=0) * 100) if (TG / 'cv_summary.csv').exists() else None
FIRST = R['ES']['wide'].index[-1] + pd.offsets.MonthBegin()           # first forecast month (Nov 2026)
SEASON = f"{FIRST:%b %Y} – {FIRST + pd.DateOffset(months=C.HORIZON - 1):%b %Y}"
SEASON_WORD = f'spring {FIRST.year + (FIRST.month > 6)}'


def scorecard(cc):
    s = R[cc]['summary']
    f, b = s.loc[FROZEN], s.loc['snaive']
    return {'skill': f.skill_vs_last_year, 'dir': f.direction_ok, 'dir_base': b.direction_ok, 'cover': f.coverage,
            'calls_right': int(f.calls_correct), 'calls': int(f.calls_made)}


def calls_table(cc):
    c = R[cc]['calls'].copy()
    c['low_chg'] = c['low'] / c['last_year'] - 1
    c['high_chg'] = c['high'] / c['last_year'] - 1
    c['kind'] = c.attribute.str.split(':').str[0]
    c['value'] = c.attribute.str.split(':').str[1]
    return c


def verdict_text(v, ch):
    if v == 'stable' and abs(ch) >= 0.05:
        return STATUS['leaning up' if ch > 0 else 'leaning down']
    return VERDICT_ICON.get(v, v)


def pct(x):
    return f'{x * 100:.1f}%'


@st.cache_data
def track_record(cc, stamp):
    """Per search: in the 2024-26 replays, rising/fading calls made, calls right, direction right."""
    D_ = R[cc]
    w = D_['wide']
    Y = w.to_numpy(float).T / D_['scale'][:, None]
    P = D_['preds'][FROZEN]
    out = {}
    for s, term in enumerate(w.columns):
        made = right = ok = n = 0
        for ci, t in enumerate(D_['cutoffs']):
            if t + C.HORIZON > Y.shape[1]:
                continue
            q, ly = P[ci, s], Y[s, t - 12:t - 12 + C.HORIZON].mean()
            truth = F.direction(Y[s, t:t + C.HORIZON].mean() / ly - 1)
            n += 1
            ok += F.direction(q[:, 1].mean() / ly - 1) == truth
            c = F.call(Y[s, :t], q)['call']
            if c != 'stable':
                made += 1
                right += c == truth
        out[term] = {'made': made, 'right': right, 'dir': ok / max(n, 1)}
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def tag_photo(data):
    """Layer 1 on one photo with the frozen recipe chosen by nested cross-validation."""
    import joblib
    from fpt.tagger import crop_person, zero_shot_ens
    t0 = time.time()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    emb = embedder()
    with gpu_lock():                    # the designer moves this embedder to the CPU while it draws
        crop, box = crop_person(img)
        vec = emb.images([crop], crop=False, flip=True)
        zs = zero_shot_ens(emb, vec)
    pred = dict(zs)
    fin = TG / 'tagger_final.joblib'
    if fin.exists():
        final = joblib.load(fin)
        pa = final['probe_A'].predict(vec, fallback=zs)
        pb = zsb = None
        if any('B' in r for r in final['recipe'].values()):
            vb = embedder_b().images([crop], crop=False, flip=True)
            zsb = zero_shot_ens(embedder_b(), vb)
            pb = final['probe_B'].predict(vb, fallback=zsb)
        parts = {'A': zs, 'pr': pa, 'B': zsb, 'prB': pb}
        for h, r in final['recipe'].items():
            use = {'A': ['A'], 'A+pr': ['A', 'pr'], 'B': ['B'], 'A+B': ['A', 'B'],
                   'A+B+pr': ['A', 'B', 'pr', 'prB'], 'A1': ['A']}[r]
            pred[h] = np.mean([parts[u][h] for u in use], 0)
    return {'style': np.asarray(pred['style'][0]), 'fabric': np.asarray(pred['fabric'][0]),
            'colour': np.asarray(pred['colour'][0]), 'print': float(zs['motif'][0, 0]),
            'box': tuple(float(v) for v in box) if box else None, 'vec': vec[0], 'secs': time.time() - t0}


def term_fig(cc, term, title):
    w = R[cc]['wide']
    s = list(w.columns).index(term)
    q = R[cc]['preds'][FROZEN][-1, s] * R[cc]['scale'][s]
    fut = pd.date_range(FIRST, periods=C.HORIZON, freq='MS')
    hist = w[term].iloc[-36:]
    fig = go.Figure([
        go.Scatter(x=hist.index, y=hist, line=dict(color='#2f5d8a', width=1.8), name='searches'),
        go.Scatter(x=list(fut) + list(fut[::-1]), y=list(q[:, 2]) + list(q[::-1, 0]), fill='toself',
                   fillcolor='#d08415', opacity=0.2, line=dict(width=0), hoverinfo='skip'),
        go.Scatter(x=fut, y=q[:, 1], line=dict(color='#d08415', width=2.4), name='forecast')])
    fig.update_layout(height=190, margin=dict(l=4, r=4, t=30, b=4), showlegend=False,
                      title=dict(text=title, font=dict(size=13)), yaxis=dict(showticklabels=False))
    return fig


def tag_card(k, v, sub, trend=''):
    return (f"<div class='tag'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{sub}</div>"
            f"<div class='t'>{trend}</div></div>")


def trend_line(ct, kind, value, country):
    """'🟢 rising in Spain · linen shirt +33%' for one tag (value None = every search of that kind)."""
    from fpt.design import status
    sub = ct[(ct.kind == kind) & ((ct.value == value) if value else True)]
    if not len(sub):
        return f'no {country} search to forecast'
    top = sub.loc[sub.change.abs().idxmax()]
    return f"{STATUS[status(sub)]} in {country} · {top.english} {top.change * 100:+.0f}%"


def stepper(done):
    names = ['① Tag', '② Forecast', '③ Brief', '④ Design', '⑤ Report']
    cols = st.columns(5)
    for col, name, ok in zip(cols, names, done):
        col.markdown(f"<div class='step {'done' if ok else 'wait'}'>{'✅' if ok else '⏳'} {name}</div>",
                     unsafe_allow_html=True)


# ======================================================================================
# Pipeline page
# ======================================================================================
def pipeline():
    from fpt import design as D, verify as V
    a, b = st.columns([4, 1.3], vertical_alignment='bottom')
    a.title('👗 Fashion Prediction Technology')
    a.caption('One photo in → what it is → where it is heading → a design for next season → checked against real photos '
              '→ report.')
    cc = b.segmented_control('Market', list(R), default='ES', format_func=lambda k: rmeta['countries'][k]) or 'ES'
    country, ct = rmeta['countries'][cc], calls_table(cc)
    top = st.empty()

    # ---------- ① Photo → tags ----------
    with st.container(border=True):
        st.markdown('#### ① Photo → Tagger')
        left, right = st.columns([1, 2.6], gap='large')
        with left:
            srcs = ['Upload'] + (['Team photo'] if team_paths else []) + (['Influencer photo'] if batch2 is not None else []) + \
                (['Barcelona street'] if photos is not None else [])
            src = st.segmented_control('Source', srcs, default='Upload', label_visibility='collapsed') or 'Upload'
            data, note = None, ''
            if src == 'Upload':
                up = st.file_uploader('Outfit photo', type=['jpg', 'jpeg', 'png', 'webp'], label_visibility='collapsed')
                if up:
                    data, note = up.getvalue(), 'Your photo: never seen by any model. This is the honest test.'
            elif src == 'Team photo':
                i = int(st.number_input('Team photo number', 0, len(team_paths) - 1, 0))
                data = Path(team_paths[i]).read_bytes()
                lab = pd.read_csv(C.LABEL_FILE).fillna('—').set_index('image')
                name = Path(team_paths[i]).name
                note = 'Team photo: the probe was trained on these, so this is a demo, not a test.'
                if name in lab.index:
                    r = lab.loc[name]
                    note = f"Our label: **{r['style']} · {r['fabric']} · {r['colour']}**. " + note
            elif src == 'Influencer photo':
                i = int(st.number_input('Influencer photo number', 0, len(batch2) - 1, 0))
                r = batch2.iloc[i]
                data = (C.BATCH2_DIR / r['image']).read_bytes()
                note = (f"Batch 2, photo {r['row']} ({r['group']}). Our label: **{r['style']} · {r['fabric']} · "
                        f"{r['colour']}**. The probe was trained on these too, so this is a demo, not a test.")
            else:
                i = int(st.number_input('Barcelona photo number', 0, len(photos) - 1, 7))
                data = Path(photos.iloc[i].path).read_bytes()
                note = f'Barcelona, {pd.Timestamp(photos.iloc[i].month):%B %Y} · public Flickr photo, kept locally.'
        if data is None:
            with right:
                st.info('Upload an outfit photo, or pick a team / Barcelona photo. Everything below then runs from it.')
            with top.container():
                stepper([False] * 5)
            return
        h = hashlib.md5(data).hexdigest()[:10]
        img = Image.open(io.BytesIO(data)).convert('RGB')
        with st.spinner('Tagging...'):
            tg = tag_photo(data)
        with left:
            shown = img.copy()
            if tg['box']:
                ImageDraw.Draw(shown).rectangle(tg['box'], outline='#ff2d55', width=max(3, img.width // 150))
            st.image(shown, width=280)
            st.caption(note + (' Red box: person found by Faster R-CNN.' if tg['box'] else ' No person found: whole photo used.'))
        with right:
            heads = {'style': C.STYLES, 'fabric': C.FABRICS, 'colour': C.COLOURS}
            m = st.columns(4)
            tags = {}
            for col, (k, names) in zip(m, heads.items()):
                p = tg[k]
                o = np.argsort(-p)
                tags[k] = st.session_state.get(f'{k[0]}{h}', names[o[0]])     # a corrected tag wins
                sure = f'{p[o[0]]:.0%} sure · next: {names[o[1]]}' if tags[k] == names[o[0]] else 'corrected by you'
                col.markdown(tag_card(k.title(), tags[k], sure, trend_line(ct, k, tags[k], country)),
                             unsafe_allow_html=True)
            tags['print'] = st.session_state.get(f'p{h}', tg['print'] >= 0.5)
            m[3].markdown(tag_card('Print / pattern', 'yes' if tags['print'] else 'plain',
                                   f"{max(tg['print'], 1 - tg['print']):.0%} sure",
                                   trend_line(ct, 'motif', None, country) if tags['print'] else 'plain: nothing to forecast'),
                          unsafe_allow_html=True)
            st.write('')
            with st.popover('✎ Correct a tag'):
                st.caption('The tagger is right about 2 times in 3. Fix a tag and every step below follows.')
                tags['style'] = st.selectbox('Style', C.STYLES, C.STYLES.index(tags['style']), key=f's{h}')
                tags['fabric'] = st.selectbox('Fabric', C.FABRICS, C.FABRICS.index(tags['fabric']), key=f'f{h}')
                tags['colour'] = st.selectbox('Colour', C.COLOURS, C.COLOURS.index(tags['colour']), key=f'c{h}')
                tags['print'] = st.toggle('Has a print', tags['print'], key=f'p{h}')
            if CVS is not None:
                st.caption(f"Tagged in {tg['secs']:.1f} s · Faster R-CNN → Marqo-FashionSigLIP + FashionCLIP + our probe · "
                           f"on unseen photos the true answer is in the top 2: style {CVS.loc['style', 'top2']:.0f}%, "
                           f"fabric {CVS.loc['fabric', 'top2']:.0f}%, colour {CVS.loc['colour', 'top2']:.0f}%")
            brief = D.photo_brief(tags, ct, country, SEASON_WORD)
            name = lambda x: f"{KIND_LABEL[x['kind']]} {x['photo']}"
            ups = [name(x) for x in brief['steps'] if x['trend'] in ('rising', 'leaning up')]
            downs = [x for x in brief['steps'] if x['trend'] in ('fading', 'leaning down')]
            if downs:
                st.warning(f"**Partly off trend in {country}.** " + ' · '.join(
                    f"{name(x)} is {x['trend']}" + (f" → the design swaps it for **{x['design']}** ({x['swap_trend']})"
                                                     if x['action'] == 'swap' else '') for x in downs) +
                    (f" · on trend: {', '.join(ups)}" if ups else ''))
            elif ups:
                st.success(f"**On trend in {country}:** {', '.join(ups)}. The design keeps this look and builds on it.")
            else:
                st.info(f'**Holding steady in {country}:** no tag is clearly rising or fading. The design keeps the look.')

    # ---------- ② Tags → forecast ----------
    with st.container(border=True):
        st.markdown(f'#### ② Forecast: where are these tags heading in {country}? · {SEASON}')
        tr = track_record(cc, _stamp(RP / 'meta.json'))
        want = [('style', tags['style']), ('fabric', tags['fabric']), ('colour', tags['colour'])]
        linked = pd.concat([ct[(ct.kind == k) & (ct.value == v)] for k, v in want] +
                           ([ct[ct.kind == 'motif']] if tags['print'] else []))
        rows = []
        for k, v in want + ([('motif', 'print')] if tags['print'] else []):
            sub = ct[(ct.kind == k) & (ct.value == v)] if k != 'motif' else ct[ct.kind == 'motif']
            if not len(sub):
                rows.append({'tag': f'{KIND_LABEL[k]}: {v}', 'search': '—', 'forecast': STATUS['no data'],
                             'vs last year': '', 'range': '', '2024–26 track record': ''})
            for r in sub.sort_values('change', ascending=False).itertuples():
                t = tr[r.term]
                rows.append({'tag': f'{KIND_LABEL[k]}: {v}', 'search': f'{r.english} ({r.term})',
                             'forecast': verdict_text(r.call, r.change), 'vs last year': f'{r.change * 100:+.0f}%',
                             'range': f'{r.low_chg * 100:+.0f}% to {r.high_chg * 100:+.0f}%',
                             '2024–26 track record': (f"calls {t['right']}/{t['made']} right · " if t['made'] else '') +
                                                     f"direction {t['dir']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        show = linked.reindex(linked.change.abs().sort_values(ascending=False).index).head(4)
        if len(show):
            cols = st.columns(4)
            for col, r in zip(cols, show.itertuples()):
                col.plotly_chart(term_fig(cc, r.term, f'{r.english} · {r.change * 100:+.0f}%'), width='stretch',
                                 config={'displayModeBar': False}, key=f'tf{cc}{r.term}')
        sc = scorecard(cc)
        st.caption(f"Google search interest in {country}; blue = last 3 years, orange = forecast with 80% range. "
                   f"{NAMES[FROZEN]}, frozen on data to Dec 2023. In 2024–26 replays: {sc['skill'] * 100:.0f}% less error "
                   f"than 'same as last year', calls right {sc['calls_right']}/{sc['calls']}. Rising = +10% and even the "
                   f"low forecast above last year; leaning = a 5%+ move that is not yet certain.")

    # ---------- ③ Brief ----------
    with st.container(border=True):
        st.markdown('#### ③ Design brief: keep what is trending, swap what is fading')
        bcols = st.columns(4)
        for col, s in zip(bcols, brief['steps']):
            arrow = f"{s['photo']} → **{s['design']}**" if s['action'] == 'swap' else f"**{s['design']}**"
            why = (f"{s['photo']} is {s['trend']}, {s['design']} is {s['swap_trend']}" if s['action'] == 'swap'
                   else STATUS.get(s['trend'], s['trend']))
            col.markdown(f"{'🔁 swap' if s['action'] == 'swap' else '✔ keep'} · {KIND_LABEL[s['kind']]}  \n{arrow}  \n"
                         f"<small>{why}</small>", unsafe_allow_html=True)
        default = D.photo_prompt(brief)
        prompt = st.text_area('Prompt for SDXL-Turbo (editable)', default, height=75,
                              key='prompt_' + hashlib.md5(default.encode()).hexdigest()[:8])
        st.caption(f"Hero piece = the fastest-moving {country} search in the design's fabric ({brief['pieces'][0]}). "
                   f"Strongest rising searches in {country}: {', '.join(brief['evidence'][:4]) or 'none'}.")

    # ---------- ④ Designer ----------
    with st.container(border=True):
        st.markdown('#### ④ Designer: SDXL-Turbo, then real-photo checks')
        b, c, d, go_col = st.columns([1, 1, 1.4, 2], vertical_alignment='bottom')
        n = b.selectbox('Concepts', [2, 4, 6, 8], index=1)
        seed = int(c.number_input('Seed', 0, 9999, 0))
        auto = d.toggle('Auto-run on new photo', True)
        go_btn = go_col.button('🎨 Generate concepts', type='primary')
        key = (h, prompt, n, seed, cc, tuple(sorted(tags.items())))
        gen = st.session_state.get('gen')
        cal = V.calibrate(team_vecs) if team_vecs is not None else None
        if cal is not None and (go_btn or (auto and (gen is None or gen['key'] != key))):
            emb = embedder()
            with st.status('Designer running', expanded=True) as box:
                lock = gpu_lock()
                if not lock.acquire(blocking=False):
                    box.write('Another design run is still drawing (another tab, or the previous photo). Waiting for it...')
                    lock.acquire()
                try:
                    box.write('Drawing with SDXL-Turbo (the first run loads the 7 GB model, about a minute)...')
                    emb.to('cpu')
                    t0 = time.time()
                    images = draw(prompt, n, seed)
                    secs = time.time() - t0
                    emb.to(F.DEVICE)
                    box.write(f'Checking each concept against our {len(team_vecs)} real outfit photos...')
                    vecs = emb.images(images)
                    results = V.check(vecs, brief, emb, team_vecs, cal)
                finally:
                    emb.to(F.DEVICE)
                    lock.release()
                gen = {'key': key, 'images': images, 'results': results, 'sims': vecs @ tg['vec'], 'secs': secs,
                       'brief': brief, 'prompt': prompt}
                st.session_state['gen'] = gen
                box.update(label=f"Designer done: {sum(r['kept'] for r in results)} of {n} concepts passed every check",
                           state='complete', expanded=False)
        fresh = gen is not None and gen['key'] == key
        if gen is not None and not fresh:
            st.warning('These concepts are from an earlier photo or prompt. Press Generate to update them.')
        if gen is not None:
            res = gen['results']
            st.markdown(f"**{sum(r['kept'] for r in res)} of {len(res)} kept** · drawn in {gen['secs']:.0f} s · "
                        f"{sum(r['looks_real'] for r in res)} look real · {sum(r['near_copy'] for r in res)} copies of real photos")
            cols = st.columns(4)
            for i, (im, r, s) in enumerate(zip(gen['images'], res, gen['sims'])):
                with cols[i % 4]:
                    st.image(im, width='stretch')
                    st.markdown(f"{'✅ kept' if r['kept'] else '❌ rejected'} · {r['matched']}/{r['needed']} attributes")
                    st.caption(' · '.join(f"{'✓' if v else '✗'} {k.split(' ', 1)[1]}" for k, v in r['detail'].items()) +
                               f" · {'✓' if r['looks_real'] else '✗'} looks real ({r['nearest_real']:.2f})"
                               f" · {s:.2f} similar to your photo")
            if cal is not None:
                st.caption(f"Kept = the requested style, fabric and colour are each in the tagger's top 2 · the closest real photo "
                           f"is at least {cal['realism']:.2f} similar (95% of real photos reach that) · and below "
                           f"{cal['near_copy']:.2f}, the level of duplicate screenshots (not a copy).")

    # ---------- ⑤ Report ----------
    with st.container(border=True):
        st.markdown('#### ⑤ Report')
        rkey = (key, fresh)
        if st.session_state.get('report_key') != rkey:
            from fpt.report import build_brand_report, search_chart
            w = R[cc]['wide']
            charts = []
            for r in show.itertuples():
                s_ = list(w.columns).index(r.term)
                charts.append(search_chart(w[r.term].iloc[-36:], R[cc]['preds'][FROZEN][-1, s_] * R[cc]['scale'][s_],
                                           f'{r.english} ({r.term}): {r.call}'))
            trust = [(f'Forecast, {country} replays 2024–26 (frozen on data to Dec 2023)',
                      f"{sc['skill'] * 100:.0f}% less error than 'same as last year'; calls came true "
                      f"{sc['calls_right']}/{sc['calls']}; truth inside range {sc['cover'] * 100:.0f}%")]
            if CVS is not None:
                trust.append((f"Tagger, cross-validated on {cv_meta['photos']} photos", f"top-2 right: style {CVS.loc['style', 'top2']:.0f}%, "
                              f"fabric {CVS.loc['fabric', 'top2']:.0f}%, colour {CVS.loc['colour', 'top2']:.0f}%"))
            concepts = [(im, f"{'kept' if r['kept'] else 'rejected'} · {r['matched']}/{r['needed']} attributes · "
                             f"{'looks real' if r['looks_real'] else 'not realistic enough'}")
                        for im, r in zip(gen['images'], gen['results'])] if fresh else ()
            start = {'photo': img, 'rows': [(KIND_LABEL[s['kind']], s['photo'], s['trend'], s['design'])
                                            for s in brief['steps']]}
            note = (f'Forecasts: Google Trends search interest in {country}, Nov 2019 – Oct 2026; {NAMES[FROZEN]}, frozen on '
                    'data up to Dec 2023. Search interest is a proxy for demand, not sales. Concepts: SDXL-Turbo '
                    '(non-commercial licence), checked by our tagger against 195 real outfit photos.')
            page = build_brand_report(country, SEASON, ct, charts, trust, brief, prompt, concepts, None, note, start)
            out = C.OUT / f'report_{cc}_{FIRST:%Y%m}.html'
            out.write_text(page, encoding='utf-8')
            st.session_state['report'], st.session_state['report_key'] = (page, out), rkey
        page, out = st.session_state['report']
        a, b = st.columns([1, 3], vertical_alignment='center')
        a.download_button('⬇ Download HTML report', page, file_name=out.name, mime='text/html', type='primary')
        b.caption(f'Saved to {out.relative_to(C.ROOT)} · photo, tags, forecasts, brief' +
                  (', concepts and checks.' if fresh else '. Generate concepts to include them.'))
        with st.expander('Preview'):
            st.components.v1.html(page, height=800, scrolling=True)

    with top.container():
        stepper([True, True, True, fresh, True])


# ======================================================================================
# Evidence page
# ======================================================================================
def evidence():
    st.title('📊 Evidence: how each step was tested')
    es, it = scorecard('ES'), scorecard('IT')
    c = st.columns(4)
    if CVS is not None:
        c[0].metric('Tagger: colour in top 2', f"{CVS.loc['colour', 'top2']:.0f}%",
                    f"style {CVS.loc['style', 'top2']:.0f}% · fabric {CVS.loc['fabric', 'top2']:.0f}%", delta_color='off')
    c[1].metric('Forecast error vs "same as last year"', f"{es['skill'] * 100:.0f}% less (Spain)",
                f"{it['skill'] * 100:.0f}% less in untouched Italy", delta_color='off')
    c[2].metric('Rising/fading calls that came true', f"{es['calls_right'] + it['calls_right']} of {es['calls'] + it['calls']}",
                f"{(es['calls_right'] + it['calls_right']) / max(1, es['calls'] + it['calls']) * 100:.0f}% · 2024–26 replays",
                delta_color='off')
    if dres:
        s = dres['spring2027']['text_only']
        c[3].metric('Generated concepts kept', f"{s['kept']} of {s['concepts']}", f"{s['near_copies']} copies of real photos",
                    delta_color='off')
    st.markdown("""
| Step | Deep learning model | Tested on data it never saw by |
|---|---|---|
| ① Tagger | Faster R-CNN (CNN) → **Marqo-FashionSigLIP**, **FashionCLIP** (vision transformers) + our linear probe | 5-fold cross-validation split by Instagram page; recipe chosen inside each fold |
| ② Forecaster | **Chronos-2** (time-series transformer) + **our quantile LSTM** | frozen on data to Dec 2023, replayed Jan 2024 – May 2026; Italy never used to build anything |
| ④ Designer | **SDXL-Turbo** (diffusion) | thresholds from real photos only; 2025 replay against real 2025 photos |
""")
    view = st.segmented_control('Show', ['Forecast replays 2024–26', 'All 2027 calls', 'Tagger accuracy', 'Designer test',
                                         'Barcelona street-photo proof'], default='Forecast replays 2024–26') or \
        'Forecast replays 2024–26'
    st.divider()

    if view == 'Forecast replays 2024–26':
        fz = rmeta['frozen']
        st.caption(f"Forecaster chosen and LSTM retrained on Spain's searches up to {pd.Timestamp(fz['development_until']):%b %Y} "
                   f"only, then saved. From every month Jan 2024 – May 2026 it forecast the next {C.HORIZON} months from "
                   f"earlier data and was scored against what happened.")
        st.dataframe(pd.DataFrame([{
            'country': rmeta['countries'][k], 'searches': R[k]['wide'].shape[1],
            'less error than "same as last year"': f"{s['skill'] * 100:.1f}%",
            'direction right': f"{s['dir'] * 100:.0f}% (baseline {s['dir_base'] * 100:.0f}%)",
            'truth inside 80% range': f"{s['cover'] * 100:.0f}%",
            'calls that came true': f"{s['calls_right']} of {s['calls']}"} for k, s in ((k, scorecard(k)) for k in R)]),
            hide_index=True, width='stretch')
        a, b, c = st.columns([1, 1.6, 1.6])
        cc = a.segmented_control('Country', list(R), default='ES', format_func=lambda k: rmeta['countries'][k], key='rcc') or 'ES'
        D_ = R[cc]
        w, terms = D_['wide'], D_['terms']
        term = b.selectbox('Search', list(w.columns), format_func=lambda t: f"{terms.loc[t, 'english']} ({t})", key='rterm',
                           index=list(w.columns).index('pantalon de lino') if 'pantalon de lino' in w.columns else 0)
        cut = D_['cutoffs']
        valid = [i for i, t in enumerate(cut) if t + C.HORIZON <= len(w)]
        ci = c.select_slider('Pretend today is', options=valid, value=valid[12] if len(valid) > 12 else valid[0],
                             format_func=lambda i: (w.index[0] + pd.DateOffset(months=cut[i])).strftime('%b %Y'), key='rci')
        s_idx, t = list(w.columns).index(term), cut[ci]
        fut = w.index[t:t + C.HORIZON]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w.index[:t], y=w[term].iloc[:t], name='known history', line=dict(color='#2f5d8a', width=2)))
        fig.add_trace(go.Scatter(x=w.index[t - 1:], y=w[term].iloc[t - 1:], name='what really happened (hidden)',
                                 line=dict(color='#9aa5b1', width=2, dash='dot')))
        for m in (FROZEN, 'snaive'):
            q = D_['preds'][m][ci, s_idx] * D_['scale'][s_idx]
            fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=list(q[:, 2]) + list(q[::-1, 0]), fill='toself',
                                     fillcolor=COLOURS[m], opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=fut, y=q[:, 1], name=NAMES[m], line=dict(color=COLOURS[m], width=2.5)))
        fig.add_vline(x=w.index[t], line_dash='dash', line_color='#c0c6cc')
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title='Google search interest (0–100)',
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, width='stretch')
        Ysc = w.to_numpy().T / D_['scale'][:, None]
        r = F.call(Ysc[s_idx, :t], D_['preds'][FROZEN][ci, s_idx])
        ly = Ysc[s_idx, t - 12:t - 12 + C.HORIZON].mean()
        actual = Ysc[s_idx, t:t + C.HORIZON].mean() / ly - 1
        err = np.mean(np.abs(D_['preds'][FROZEN][ci, s_idx][:, 1] - Ysc[s_idx, t:t + C.HORIZON]))
        err0 = np.mean(np.abs(D_['preds']['snaive'][ci, s_idx][:, 1] - Ysc[s_idx, t:t + C.HORIZON]))
        st.info(f"Call: **{r['call']}** ({r['change'] * 100:+.0f}%) · what happened: **{F.direction(actual)}** "
                f"({actual * 100:+.0f}%) · error {err / max(err0, 1e-9) * 100:.0f}% of the 'same as last year' error.")

    elif view == 'All 2027 calls':
        st.caption(f'Every search, {SEASON}, frozen forecaster on data up to Oct 2026.')
        cols = st.columns(len(R))
        for col, cc in zip(cols, R):
            ct = calls_table(cc).sort_values('change', ascending=False)
            col.markdown(f"**{rmeta['countries'][cc]}**")
            col.dataframe(pd.DataFrame({'search': ct.english, 'call': [verdict_text(v, ch) for v, ch in zip(ct.call, ct.change)],
                                        'vs last year': (ct.change * 100).round(0).astype(int).astype(str) + '%',
                                        'range': [f'{a * 100:+.0f}% to {b * 100:+.0f}%' for a, b in zip(ct.low_chg, ct.high_chg)]}),
                          hide_index=True, width='stretch', height=600)

    elif view == 'Tagger accuracy':
        if CVS is not None:
            st.caption(f"Nested 5-fold cross-validation on all {cv_meta['photos']} labelled photos from {cv_meta['pages']} "
                       f"Instagram pages: each photo tagged by a model that never saw its page.")
            st.dataframe(pd.DataFrame({
                'attribute': CVS.index, 'main guess right': [f'{v:.0f}%' for v in CVS.right],
                'range across folds': [f'{a:.0f}–{b:.0f}%' for a, b in zip(CVS.right_min, CVS.right_max)],
                'true answer in top 2': [f'{v:.0f}%' for v in CVS.top2],
                'always guessing the most common': [f'{v:.0f}%' for v in CVS.guess]}), hide_index=True, width='stretch')
            bc = TG / 'batch_comparison.csv'
            if bc.exists():
                b = pd.read_csv(bc)
                b['batch'] = b.photos.str.split(' photos').str[0]
                st.markdown('**By batch** (same nested cross-validation)')
                st.dataframe(b.pivot_table(index='attribute', columns='batch', values=['right_after', 'top2_after'], sort=False)
                             .mul(100).round(0).astype(int).astype(str).add('%')
                             .rename(columns={'right_after': 'main guess right', 'top2_after': 'in top 2'}), width='stretch')
                st.caption('Batch 1 = the team\'s 195 Southern Europe screenshots. Batch 2 = 100 influencer photos (10 pages) '
                           'with clearer, studio-like photos and more labels per photo, so higher scores there are partly '
                           'easier photos, not a better model. Compare the same photos below.')
            gain = TG / 'batch2_gain_fixed_recipe.csv'
            if gain.exists():
                g = pd.read_csv(gain)
                st.markdown('**Did the 100 new photos help?** The shipped recipe, same held-out pages, trained without vs with them')
                st.dataframe(pd.DataFrame({
                    'photos': g.photos, 'attribute': g.attribute,
                    'main guess: before → after': [f'{a * 100:.0f}% → {b * 100:.0f}%' for a, b in zip(g.right_before, g.right_after)],
                    'top 2: before → after': [f'{a * 100:.0f}% → {b * 100:.0f}%' for a, b in zip(g.top2_before, g.top2_after)]}),
                    hide_index=True, width='stretch')
                st.caption('Fabric top 2 rose on both batches; style rose on the new pages (Tailored/Suiting 8 → 10 of 12: '
                           'the old set had no Tailored examples). Colour uses zero-shot only, so it is unchanged. The small '
                           'drops are about one photo each.')
            cand = pd.read_csv(TG / 'cv_candidates.csv', header=[0, 1], index_col=0)
            with st.expander('Every candidate recipe'):
                st.dataframe((cand['right'] * 100).round(1), width='stretch')
                st.caption('A1 = one description per class · A = Marqo, several descriptions + mirrored view · '
                           'pr = linear probe on our photos · B = FashionCLIP 2.0')
        ag = TG / 'barcelona_agreement.csv'
        if ag.exists():
            with st.expander("Agreement with GeoStyle's CNN on 28 Barcelona photos (indicative)"):
                d = pd.read_csv(ag)
                d['agreement'], d['majority_guess'] = d.agreement.map(pct), d.majority_guess.map(pct)
                st.dataframe(d, hide_index=True)

    elif view == 'Designer test' and dres:
        s = dres['spring2027']['text_only']
        found = {}
        for x in dres['spring2027']['checks']:
            for k, v in x['detail'].items():
                found[k] = found.get(k, 0) + int(v)
        st.markdown(f"**Spring 2027 brief** · *{dres['spring2027']['prompt']}*  \n"
                    f"{s['concepts']} concepts: **{s['kept']} kept**, {s['look_real']} look real, {s['near_copies']} copies · "
                    + ' · '.join(f'{k} {v}/{s["concepts"]}' for k, v in found.items()))
        ims = sorted((DS / 'images').glob('spring2027_*.jpg'), key=lambda p: int(p.stem.split('_')[-1]))
        if ims:
            st.image([str(p) for p in ims[:8]], width=150)
        rp = dres['replay2025']
        st.markdown(f"**2025 replay** · brief from the Jan 2025 forecast. Compared with our real 2025 vs 2024 photos: "
                    f"**{rp['forecast_concepts_closer_to_2025']}/{s['concepts']}** forecast concepts look more like 2025, "
                    f"against **{rp['control_concepts_closer_to_2025']}/{s['concepts']}** for a prompt with no forecast.")
        r1 = sorted((DS / 'images').glob('replay2025_[0-9]*.jpg'), key=lambda p: int(p.stem.split('_')[-1]))[:4]
        r2 = sorted((DS / 'images').glob('replay2025_control_*.jpg'), key=lambda p: int(p.stem.split('_')[-1]))[:4]
        if r1:
            st.image([str(p) for p in r1 + r2], width=150, caption=['forecast'] * len(r1) + ['control'] * len(r2))
        st.caption('Honest note: the prompt wording was revised four times while watching these checks; the checks never '
                   'changed. The print check is shown but cannot reject a concept (no print labels to measure it).')

    elif view == 'Barcelona street-photo proof':
        meta, wide, P, cutoffs, bt_recent, bt_all, lstm_hist, detail = load_trends()
        attrs = meta['attributes']
        Y = wide[attrs].to_numpy(float).T
        months = wide.index
        lab = lambda t: (months[0] + pd.DateOffset(months=t)).strftime('%b %Y')
        st.caption('Before using search data, the method was proven on 96,965 dated Barcelona street photos (GeoStyle, '
                   '2013–2018): the share of people wearing each attribute, forecast 6 months ahead.')
        a, b, c = st.columns([1.2, 1.6, 1.6])
        attr = a.selectbox('Attribute', attrs, index=attrs.index('Garment: Outerwear') if 'Garment: Outerwear' in attrs else 0)
        ci = b.select_slider('Forecast made at', options=list(range(len(cutoffs))), value=len(cutoffs) - 13,
                             format_func=lambda i: lab(cutoffs[i]))
        models = c.multiselect('Forecasters', [m for m in NAMES if m in P], default=['chronos2', 'lstm', 'snaive'],
                               format_func=lambda k: NAMES[k])
        s, t = attrs.index(attr), cutoffs[ci]
        fut = pd.date_range(months[0] + pd.DateOffset(months=t), periods=C.HORIZON, freq='MS')
        fig = go.Figure([go.Scatter(x=months[:t], y=Y[s, :t] * 100, name='known history', line=dict(color='#2f5d8a', width=2))])
        if t < len(months):
            fig.add_trace(go.Scatter(x=months[t - 1:], y=Y[s, t - 1:] * 100, name='what really happened (hidden)',
                                     line=dict(color='#9aa5b1', width=2, dash='dot')))
        for m in models:
            q = P[m][ci, s] * 100
            fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=list(q[:, 2]) + list(q[::-1, 0]), fill='toself',
                                     fillcolor=COLOURS[m], opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=fut, y=q[:, 1], name=NAMES[m], line=dict(color=COLOURS[m], width=2.5)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title='% of people wearing it',
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, width='stretch')
        if t + C.HORIZON <= len(months):
            ly = Y[s, t - 12:t - 12 + C.HORIZON].mean()
            ch = Y[s, t:t + C.HORIZON].mean() / ly - 1
            st.info(' · '.join(f"{NAMES[m].split(' (')[0]}: **{F.call(Y[s, :t], P[m][ci, s])['call']}**" for m in models) +
                    f" · what happened: **{F.direction(ch)}** ({ch * 100:+.0f}%)")
        bt = bt_recent
        st.dataframe(pd.DataFrame({
            'forecaster': [NAMES.get(i, i) for i in bt.index],
            'less error than "same as last year"': (bt.skill_vs_last_year * 100).round(1).astype(str) + '%',
            'direction right': (bt.direction_ok * 100).round(0).astype(int).astype(str) + '%',
            'truth inside range': (bt.coverage * 100).round(0).astype(int).astype(str) + '%',
            'calls right': [f'{int(a)}/{int(b)}' for a, b in zip(bt.calls_correct, bt.calls_made)]}),
            hide_index=True, width='stretch')
        st.caption(f"Most recent {len(meta['recent_cutoffs'])} start months; every forecast uses only earlier months.")
        with st.expander('Run Chronos-2 live on this series (proves the model really runs)'):
            if st.button('Forecast with Chronos-2'):
                t0 = time.time()
                df = pd.DataFrame({'item_id': 0, 'timestamp': months[:t], 'target': Y[s, :t]})
                out = chronos_cpu().predict_df(df, prediction_length=C.HORIZON, quantile_levels=C.QUANTILES, freq='MS')
                show = out[['timestamp'] + [str(q) for q in C.QUANTILES]].copy()
                show.columns = ['month', 'low (10%)', 'middle (50%)', 'high (90%)']
                for k in show.columns[1:]:
                    show[k] = (show[k] * 100).round(2)
                st.write(f'Forecast in {time.time() - t0:.1f} s on the CPU from {t} months, without training:')
                st.dataframe(show, hide_index=True)
        with st.expander('LSTM training curve (trained by us)'):
            lf = go.Figure([go.Scatter(y=lstm_hist.train_loss, name='training'), go.Scatter(y=lstm_hist.val_loss, name='held out')])
            lf.update_layout(height=280, xaxis_title='epoch', yaxis_title='pinball loss', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(lf, width='stretch')

    st.caption('Team: Gaurav Chandra Mittal, Hari Krishna Tiwari · Guide: Dr. D.P. Sharma')


st.navigation([st.Page(pipeline, title='Pipeline', icon='▶️', default=True),
               st.Page(evidence, title='Evidence', icon='📊')], position='top').run()
