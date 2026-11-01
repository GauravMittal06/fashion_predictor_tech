"""Layer 3: trend calls -> design brief -> generated concepts -> checks.

  Plan A  SDXL-Turbo : text-to-image diffusion model, prompted from the brief (fits a 6 GB GPU)
  Plan C  board      : concept board of the real photos that best match the brief

Adapted from the team's design.py: the brief is built from the Barcelona trend attributes, and
Plan A uses SDXL-Turbo (4 steps) instead of SDXL base (30 steps) so it runs on a laptop.
"""
import numpy as np
import torch
from PIL import Image
from . import config as C
from .tagger import trend_attributes

ORDER = {'confirmed rising': 0, 'rising': 0, 'watch': 1}
LEAN = 0.05                   # forecast at least 5% above last year, but not a strict call


def make_brief(calls, season):
    """calls: DataFrame with attribute, verdict, change. Strongest confirmed rises first;
    'watch' and then 'leaning up' attributes fill a slot only if nothing stronger is there for it."""
    up = calls[((calls.verdict.isin(ORDER)) & (calls.change > 0)) | (calls.change >= LEAN)]
    up = up[up.last_year >= 0.03].copy()             # skip attributes almost nobody wears
    up['rank'] = up.verdict.map(ORDER).fillna(2)
    up = up.sort_values(['rank', 'change'], ascending=[True, False])
    pick = {}
    for a in up.attribute:
        group = a.split(': ')[0]
        if group not in pick and group != 'Layers':     # one value per group (one colour, one pattern...)
            pick[group] = a
    fading = calls[(calls.verdict.isin(['confirmed fading', 'fading'])) |
                   ((calls.verdict == 'stable') & (calls.change <= -2 * LEAN))].attribute.tolist()
    return {'country': C.COUNTRY_NAME, 'city': C.CITY, 'season': season,
            'attributes': list(pick.values()), 'avoid': fading}


def _words(attr):
    group, _, value = attr.partition(': ')
    return C.GEO_TEXT[group](value or None)


PATTERN_ADJ = {'Solid': 'plain', 'Striped': 'striped', 'Graphics': 'graphic-print', 'Floral': 'floral',
               'Plaid': 'plaid', 'Spotted': 'polka-dot'}
GARMENT_NOUN = {'Dress': 'dress', 'Outerwear': 'coat', 'T-shirt': 't-shirt', 'Suit': 'suit', 'Shirt': 'shirt',
                'Sweater': 'sweater', 'Tank top': 'tank top'}
EXTRAS = {'Jacket': 'a jacket', 'Hat': 'a hat', 'Glasses': 'glasses', 'Scarf': 'a scarf', 'Necktie': 'a necktie'}


def outfit_phrase(attributes):
    """['Colour: White', 'Pattern: Striped', 'Garment: Shirt', 'Hat'] -> 'a white striped shirt, with a hat'."""
    v = {a.partition(': ')[0]: a.partition(': ')[2] or 'Yes' for a in attributes}
    adj = []
    if v.get('Sleeves') == 'No sleeve':
        adj.append('sleeveless')
    if 'Colour' in v:
        adj.append('multicoloured' if v['Colour'] == 'More than 1 color' else v['Colour'].lower())
    if 'Pattern' in v:
        adj.append(PATTERN_ADJ[v['Pattern']])
    if v.get('Neckline') == 'V-shape':
        adj.append('v-neck')
    if 'Collar' in v:
        adj.append('collared')
    noun = GARMENT_NOUN.get(v.get('Garment'), 'outfit')
    words = ' '.join(adj + [noun])
    phrase = ('an ' if words[0] in 'aeiou' else 'a ') + words
    if v.get('Sleeves') in ('Short sleeve', 'Long sleeve'):
        phrase += f" with {v['Sleeves'].lower()}s"
    extras = [EXTRAS[k] for k in EXTRAS if k in v]
    if v.get('Layers') == 'Multiple layers':
        extras.append('layered pieces')
    if extras:
        phrase += ', with ' + ' and '.join(extras)
    return phrase


def brief_to_prompt(b):
    prompt = (f'full-body street style fashion photo of a model in {b["city"]} wearing '
              f'{outfit_phrase(b["attributes"])}, new season designer collection, natural daylight, '
              f'editorial photography, sharp focus')
    negative = ', '.join([_words(a) for a in b['avoid']] + ['logo', 'text', 'watermark', 'deformed'])
    return prompt, negative


# ---------------- Plan A: SDXL-Turbo ----------------
_pipe = None


def load_generator():
    global _pipe
    if _pipe is None:
        from diffusers import AutoPipelineForText2Image
        _pipe = AutoPipelineForText2Image.from_pretrained(
            C.GEN_MODEL, torch_dtype=torch.float16, variant='fp16')
        _pipe.enable_model_cpu_offload()              # 6 GB GPU: one part on the GPU at a time
    return _pipe


def generate(prompt, n=C.CANDIDATES_PER_BRIEF, seed=0, batch=2, pipe=None):
    """n concepts, made a few at a time. SDXL-Turbo uses 4 steps and no guidance."""
    pipe = pipe or load_generator()
    images = []
    for start in range(0, n, batch):
        g = torch.Generator('cpu').manual_seed(seed + start)
        out = pipe(prompt, num_inference_steps=C.GEN_STEPS, guidance_scale=0.0,
                   width=C.GEN_SIZE, height=C.GEN_SIZE,
                   num_images_per_prompt=min(batch, n - start), generator=g)
        images += out.images
    return images


# ---------------- Checks ----------------
def check_concepts(images, brief, embedder, all_attrs, collection_vecs=None):
    """Keep concepts that show every requested attribute and are not near-copies of real photos."""
    vecs = embedder.images(images)
    probs = trend_attributes(embedder, vecs, all_attrs)
    results = []
    for i in range(len(images)):
        detail = {}
        for a in brief['attributes']:
            group = a.split(': ')[0]
            if ': ' in a:                              # must be the top class within its group
                rivals = [k for k in probs if k.startswith(group + ': ')]
                ok = probs[a][i] >= max(probs[k][i] for k in rivals) - 1e-9
            else:
                ok = probs[a][i] >= C.MATCH_THRESHOLD
            detail[a] = (bool(ok), float(probs[a][i]))
        nearest = float((collection_vecs @ vecs[i]).max()) if collection_vecs is not None and len(collection_vecs) else 0.0
        matched = sum(ok for ok, _ in detail.values())
        results.append({'index': i, 'detail': detail, 'matched': matched, 'needed': len(detail),
                        'matches_brief': matched == len(detail),
                        'nearest_real_photo_similarity': round(nearest, 3),
                        'near_copy': nearest >= C.NEAR_COPY_SIMILARITY})
    return results, vecs


# ---------------- Plan C: concept board ----------------
def concept_board(prompt, embedder, collection_vecs, images, k=9, thumb=256):
    """Retrieve the k real photos closest to the brief's text and tile them in a grid."""
    t = embedder.texts([prompt])[0]
    top = np.argsort(-(collection_vecs @ t))[:k]
    side = int(np.ceil(np.sqrt(k)))
    board = Image.new('RGB', (side * thumb, side * thumb), 'white')
    for n, idx in enumerate(top):
        im = images[idx]
        im = (Image.open(im) if not isinstance(im, Image.Image) else im).convert('RGB')
        im.thumbnail((thumb, thumb))
        board.paste(im, ((n % side) * thumb + (thumb - im.width) // 2,
                         (n // side) * thumb + (thumb - im.height) // 2))
    return board, top


# ---------------- Briefs from the Google Trends forecast (project attributes) ----------------
STYLE_PROMPT = {'Evening/Party': 'an elegant evening look', 'Casual': 'a relaxed casual look',
                'Street': 'a street style look', 'Resort/Swim': 'a summer resort look',
                'Tailored/Suiting': 'a tailored look', 'Winter/Outerwear': 'a warm winter look'}


def make_search_brief(calls, place, season, lean=LEAN):
    """calls: columns term, english, attribute ('fabric:Cotton/Linen'), call, change.
    Rising calls first, then terms forecast at least 5% above last year ('leaning up').
    The concept centres on one hero piece: the strongest rising fabric, with the strongest rising colour and
    print. Other rising pieces stay in the brief's evidence for the designer."""
    c = calls.copy()
    c['kind'] = c.attribute.str.split(':').str[0]
    c['value'] = c.attribute.str.split(':').str[1]
    up = c[(c.call == 'rising') | (c.change >= lean)].assign(r=lambda d: (d.call != 'rising').astype(int))
    up = up.sort_values(['r', 'change'], ascending=[True, False])
    pick = lambda kind, k: up[up.kind == kind].drop_duplicates('value').head(k)
    fab, col, mot = pick('fabric', 2), pick('colour', 1), pick('motif', 1)
    sty = up[(up.kind == 'style') & (up.call == 'rising')].head(1)   # a style only on a real rising call
    return {'place': place, 'season': season,
            'style': sty.value.iloc[0] if len(sty) else None,
            'fabrics': list(fab.value[:1]), 'colours': list(col.value), 'motif': mot.value.iloc[0] if len(mot) else None,
            'styling_fabrics': list(fab.value[1:]),          # second piece: in the prompt, not a required check
            'pieces': list(fab.english), 'colour_words': [w.replace(' dress', '') for w in col.english],
            'evidence': list(up.english.head(6)), 'avoid': list(c[c.call == 'fading'].english)}


MOTIF_ADJ = {'striped': 'boldly striped', 'floral': 'floral print', 'polka dot': 'polka-dot', 'print': 'printed'}


def search_prompt(b):
    pieces = list(b['pieces']) or ['outfit']
    if b['motif']:                                   # the pattern belongs to the main garment
        pieces[0] = f"{MOTIF_ADJ.get(b['motif'], b['motif'])} {pieces[0]}"
    if b['colour_words']:
        pieces[0] = f"{b['colour_words'][0]} {pieces[0]}"
    art = lambda w: w if w.endswith(('trousers', 's')) else ('an ' if w[0] in 'aeiou' else 'a ') + w
    wear = art(pieces[0]) + ' with plain light trousers'   # one hero piece; no second fabric to compete with it
    motif = ', loose matte crumpled linen fabric, not silk' if 'Cotton/Linen' in b['fabrics'] else ''
    return (f"full-body street style photo of a woman in {b['place']} wearing {wear}{motif}, "
            f"{STYLE_PROMPT[b['style']] + ', ' if b['style'] else ''}{b['season']}, natural daylight, "
            f"editorial photography, sharp focus")


# ---------------- Plan A + C: a real photo as style reference (IP-Adapter) ----------------
_ip_loaded = False


def use_reference(scale=0.4):
    """Load IP-Adapter (ViT-H image encoder) into the SDXL-Turbo pipeline; returns False if unavailable."""
    global _ip_loaded
    if _ip_loaded:
        return True
    try:
        pipe = load_generator()
        pipe.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models', weight_name='ip-adapter_sdxl_vit-h.safetensors',
                             image_encoder_folder='models/image_encoder')
        pipe.set_ip_adapter_scale(scale)
        _ip_loaded = True
    except Exception as e:                            # not downloaded / does not fit: text-only generation
        print('IP-Adapter unavailable:', type(e).__name__, str(e)[:120])
    return _ip_loaded


def generate_ref(prompt, reference, n=4, seed=0, batch=2):
    """Like generate(), steered by a real photo (needs use_reference() first)."""
    pipe = load_generator()
    images = []
    for start in range(0, n, batch):
        g = torch.Generator('cpu').manual_seed(seed + start)
        out = pipe(prompt, ip_adapter_image=reference, num_inference_steps=C.GEN_STEPS, guidance_scale=0.0,
                   width=C.GEN_SIZE, height=C.GEN_SIZE, num_images_per_prompt=min(batch, n - start), generator=g)
        images += out.images
    return images


# ---------------- Pipeline: brief seeded by an uploaded photo's tags ----------------
FABRIC_ADJ = {'Denim': 'denim', 'Leather': 'leather', 'Knit': 'knit', 'Satin/Silk': 'satin', 'Cotton/Linen': 'linen',
              'Wool/Tweed': 'wool', 'Sequin/Metallic': 'sequin', 'Lace/Mesh/Sheer': 'lace', 'Fur/Shearling': 'shearling',
              'Jersey/Stretch': 'jersey'}
TEXTURE = {'Cotton/Linen': 'loose matte crumpled linen fabric, not silk', 'Leather': 'smooth real leather',
           'Knit': 'chunky knitted texture', 'Denim': 'washed denim texture', 'Satin/Silk': 'glossy satin fabric',
           'Wool/Tweed': 'thick wool fabric', 'Fur/Shearling': 'fluffy shearling texture', 'Lace/Mesh/Sheer': 'delicate lace',
           'Sequin/Metallic': 'sparkling sequins'}
STYLE_PIECES = {'Evening/Party': ('dress',), 'Resort/Swim': ('dress', 'shirt'), 'Tailored/Suiting': ('suit', 'trousers', 'jacket'),
                'Winter/Outerwear': ('coat', 'jacket', 'jumper')}          # garments that fit the style; others: any
STYLE_NOUN = {'Evening/Party': 'dress', 'Winter/Outerwear': 'coat', 'Tailored/Suiting': 'suit', 'Resort/Swim': 'summer dress',
              'Casual': 'top', 'Street': 'jacket'}


def _calls_kv(calls):
    c = calls.copy()
    c['kind'] = c.attribute.str.split(':').str[0]
    c['value'] = c.attribute.str.split(':').str[1]
    return c


def status(rows, lean=LEAN):
    """One trend status for an attribute from all its searches: rising / leaning up / stable / leaning down / fading."""
    if not len(rows):
        return 'no data'
    if (rows.call == 'rising').any():
        return 'rising'
    top = rows.change.max()
    if top >= lean:
        return 'leaning up'
    if (rows.call == 'fading').any() or top <= -lean:
        return 'fading' if (rows.call == 'fading').any() else 'leaning down'
    return 'stable'


def strongest_up(c, kind, lean=LEAN):
    """The value of this kind with the strongest rising (then leaning-up) search, or None."""
    up = c[(c.kind == kind) & ((c.call == 'rising') | (c.change >= lean))]
    if not len(up):
        return None
    up = up.assign(r=(up.call != 'rising').astype(int)).sort_values(['r', 'change'], ascending=[True, False])
    return up.value.iloc[0]


def photo_brief(tags, calls, place, season, lean=LEAN):
    """tags: {'style', 'fabric', 'colour', 'print'} from the tagger. Keep what the photo has unless the forecast says it
    is fading; then swap it for the strongest rising value of the same kind. A style is swapped only for a real
    rising call. Returns the brief (same keys as make_search_brief, so verify.check works) plus 'steps'."""
    c = _calls_kv(calls)
    steps, pick = [], {}
    for kind in ('style', 'fabric', 'colour'):
        have = tags[kind]
        st = status(c[(c.kind == kind) & (c.value == have)], lean)
        new = have
        if st in ('fading', 'leaning down'):
            alt = strongest_up(c, kind, lean) if kind != 'style' else (
                c[(c.kind == 'style') & (c.call == 'rising')].sort_values('change').value.iloc[-1]
                if ((c.kind == 'style') & (c.call == 'rising')).any() else None)
            new = alt or have
        pick[kind] = new
        steps.append({'kind': kind, 'photo': have, 'trend': st, 'design': new,
                      'action': 'swap' if new != have else 'keep',
                      'swap_trend': status(c[(c.kind == kind) & (c.value == new)], lean) if new != have else st})
    motif = (strongest_up(c, 'motif', lean) or 'print') if tags['print'] else None   # a print stays a print
    steps.append({'kind': 'print', 'photo': 'print' if tags['print'] else 'plain', 'design': motif or 'plain',
                  'trend': status(c[c.kind == 'motif'], lean) if tags['print'] else 'not checked',
                  'action': 'keep', 'swap_trend': ''})
    # hero piece: the fastest-moving search in the chosen fabric that suits the style (an evening look gets the linen
    # dress, not the linen shirt), else the fabric + a garment for the style
    fab = c[(c.kind == 'fabric') & (c.value == pick['fabric'])].sort_values('change', ascending=False)
    fits = STYLE_PIECES.get(pick['style'])
    if fits:
        fab = fab[fab.english.str.endswith(fits)]
    piece = fab.english.iloc[0] if len(fab) else f"{FABRIC_ADJ[pick['fabric']]} {STYLE_NOUN[pick['style']]}"
    col = c[(c.kind == 'colour') & (c.value == pick['colour'])]
    colour_word = col.english.iloc[0].replace(' dress', '') if len(col) else pick['colour'].split('/')[0]
    up = c[(c.call == 'rising') | (c.change >= lean)].sort_values('change', ascending=False)
    return {'place': place, 'season': season, 'style': pick['style'], 'fabrics': [pick['fabric']],
            'colours': [pick['colour']], 'motif': motif, 'pieces': [piece], 'colour_words': [colour_word],
            'styling_fabrics': [], 'evidence': list(up.english.head(6)), 'avoid': list(c[c.call == 'fading'].english),
            'steps': steps}


def photo_prompt(b):
    piece = b['pieces'][0]
    if b['motif']:
        piece = f"{MOTIF_ADJ.get(b['motif'], b['motif'])} {piece}"
    piece = f"{b['colour_words'][0]} {piece}"
    art = piece if piece.endswith(('trousers', 'jeans', 'shorts')) else ('an ' if piece[0] in 'aeiou' else 'a ') + piece
    if piece.endswith(('dress', 'suit')):
        wear = art
    elif piece.endswith('trousers'):
        wear = art + ' with a plain white top'
    else:
        wear = art + ' with plain light trousers'
    texture = f", {TEXTURE[b['fabrics'][0]]}" if b['fabrics'][0] in TEXTURE else ''
    scene = {'Resort/Swim': 'fashion photo of a woman on a seaside promenade',
             'Evening/Party': 'fashion photo of a woman at an evening party'}.get(b['style'], 'street style photo of a woman')
    light = 'warm evening light' if b['style'] == 'Evening/Party' else 'natural daylight'
    return (f"full-body {scene} in {b['place']} wearing {wear}{texture}, "
            f"{STYLE_PROMPT[b['style']]}, {b['season']}, {light}, editorial photography, sharp focus")
