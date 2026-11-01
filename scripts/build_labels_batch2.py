"""Batch 2 labels: the team's free-text CSV for 100 influencer photos -> the tagger's classes (labels_batch2.csv).

The CSV describes each outfit in words ("Smooth black leather", "Monochromatic Jet Black"). Rules:
  - every class whose keyword appears is kept (fabric and colour are multi-label, as in labels.csv); for style the
    first keyword wins;
  - accessories (bags, boots, gloves, tights, hardware...) are dropped before matching, so a leather clutch does not
    make a dress "leather";
  - colours with no class of ours (yellow, orange, purple, multicolour) are left out;
  - row 1.7 has colour and material in each other's columns, and is swapped back;
  - REVIEWED holds the corrections made by looking at every photo next to its draft labels, before any model was run
    on these photos. Each says why.

    python scripts/build_labels_batch2.py
"""
import re
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402

SRC = C.ROOT / 'dataset_images_2' / 'fashion trends and forecasting(Influencers).csv'
ACCESSORY = re.compile(r'glove|boot|bag|clutch|tights|pouch|heel|shoe|sandal|platform|hardware|buckle|stud|headscarf|'
                       r'scarf top|neckline', re.I)
KW = {
    'style': {'Evening/Party': ['evening', 'gown', 'party', 'cocktail', 'glam', 'red carpet', 'couture', 'nocturnal'],
              'Resort/Swim': ['resort', 'swim', 'bikini', 'beach', 'vacation', 'yacht', 'tropical', 'island'],
              'Tailored/Suiting': ['tailor', 'suit', 'tuxedo', 'blazer', 'prep'],
              'Winter/Outerwear': ['winter', 'outerwear', 'trench', 'coat', 'apres-ski', 'puffer', 'poncho'],
              'Street': ['street', 'biker', 'utility', 'double denim', 'darkwear'],
              'Casual': ['casual', 'lounge', 'athleisure', 'activewear', 'everyday', 'co-ord', 'chic']},
    'fabric': {'Denim': ['denim'], 'Leather': ['leather', 'suede', 'vinyl', 'patent'],
               'Knit': ['knit', 'crochet', 'bouclé', 'boucle'], 'Satin/Silk': ['satin', 'silk', 'charmeuse', 'taffeta'],
               'Cotton/Linen': ['linen', 'cotton', 'poplin', 'canvas', 'twill', 'corduroy'],
               'Wool/Tweed': ['wool', 'tweed', 'cashmere', 'houndstooth', 'gabardine', 'alpaca'],
               'Sequin/Metallic': ['sequin', 'metallic', 'lurex', 'chainmail', 'tinsel', 'foil', 'brocade'],
               'Lace/Mesh/Sheer': ['lace', 'mesh', 'tulle', 'sheer', 'organza', 'chiffon', 'fishnet'],
               'Fur/Shearling': ['fur', 'shearling', 'teddy'],
               'Jersey/Stretch': ['jersey', 'stretch', 'spandex', 'elastane', 'nylon', 'rayon', 'viscose', 'polyamide',
                                  'scuba', 'ponte', 'crepe']},
    'colour': {'black': ['black', 'onyx'], 'white/cream': ['white', 'cream', 'ivory'],
               'beige': ['beige', 'oat', 'sand', 'camel', 'tan', 'taupe', 'nude', 'khaki', 'champagne'],
               'red': ['red', 'crimson', 'burgundy', 'wine', 'cherry', 'oxblood', 'terracotta', 'coral'],
               'blue': ['blue', 'navy', 'indigo', 'turquoise', 'aqua'], 'pink': ['pink', 'blush', 'magenta', 'lavender'],
               'green': ['green', 'olive', 'sage', 'lime', 'chartreuse', 'teal'],
               'brown': ['brown', 'chocolate', 'espresso', 'mocha', 'rust', 'caramel', 'chestnut', 'chestunt'],
               'grey': ['grey', 'gray', 'charcoal', 'slate'], 'metallic': ['gold', 'silver', 'metallic', 'bronze']}}

# Corrections after looking at every photo next to its draft labels (done before any model saw these photos).
EVE, CAS = 'Evening/Party', 'Casual'
REVIEWED = {
    # style: the description names the garment, not a style -> chosen from the photo
    '1.2': {'style': 'Street'}, '1.4': {'style': 'Tailored/Suiting'}, '7.1': {'style': EVE, 'colour': 'blue'},
    '7.10': {'style': EVE}, '8.1': {'style': EVE}, '8.3': {'style': EVE}, '8.9': {'style': CAS}, '8.10': {'style': EVE},
    '9.1': {'style': EVE}, '9.4': {'style': CAS}, '9.10': {'style': EVE}, '10.1': {'style': EVE, 'colour': 'black'},
    '10.2': {'style': CAS}, '10.4': {'style': EVE}, '10.5': {'style': EVE, 'colour': 'black|white/cream'},
    '10.6': {'style': EVE, 'colour': 'red'}, '10.7': {'style': CAS, 'colour': 'green'}, '10.8': {'style': EVE},
    '10.9': {'style': EVE, 'colour': 'white/cream'}, '10.10': {'style': EVE},
    # style: "street style" there means photographed in the street, not streetwear; the outfit is everyday wear
    '2.1': {'style': CAS}, '2.2': {'style': CAS}, '2.6': {'style': CAS}, '4.8': {'style': CAS},
    # style: the first keyword was not the main look in the photo
    '3.4': {'style': EVE},                       # sheer chiffon blouse + python skirt, no tailoring visible
    '4.1': {'style': 'Winter/Outerwear'},        # the camel coat dominates
    # colour: an accessory or a colour we have no class for (purple, amber, yellow) was picked up
    '1.8': {'colour': 'white/cream'},            # purple trousers are not pink
    '5.5': {'colour': ''},                       # amber and violet: no class of ours
    '10.3': {'colour': 'black'},                 # green came from the shoes
}


def matches(text, table):
    """Classes found in the text, in order of first mention, after dropping accessory phrases."""
    parts = re.split(r',|&|/| with | and |\(|\)', str(text))
    keep = ' , '.join(p for p in parts if not ACCESSORY.search(p)).lower()
    hits = []
    for cls, words in table.items():
        pos = [m.start() for w in words for m in re.finditer(r'(?<![a-z])' + re.escape(w), keep)]
        if pos:
            hits.append((min(pos), cls))
    return [c for _, c in sorted(hits)]


def build():
    d = pd.read_csv(SRC, encoding='cp1252', dtype=str).fillna('')
    d['name'] = d['name'].str.strip().str.rstrip('.')
    swap = d['name'] == '1.7'
    d.loc[swap, ['color', 'material']] = d.loc[swap, ['material', 'color']].values
    rows = []
    for _, r in d.iterrows():
        styles = matches(r['category (style)'], KW['style'])
        row = {'image': r['name'] + '.jpeg', 'group': r['influencer name'].strip(), 'row': r['name'],
               'style': styles[0] if styles else '', 'fabric': '|'.join(matches(r['material'], KW['fabric'])),
               'colour': '|'.join(matches(r['color'], KW['colour'])), 'button': '', 'zipper': '', 'motif': '',
               'region': r['region'].strip(), 'date': pd.to_datetime(r['date posted'], format='%d-%b-%y', errors='coerce'),
               'batch': 2}
        row.update({k: v for k, v in REVIEWED.get(row['row'], {}).items() if k in ('style', 'fabric', 'colour')})
        rows.append(row)
    out = pd.DataFrame(rows)
    assert all((C.BATCH2_DIR / i).exists() for i in out['image']), 'photo missing'
    assert out['style'].isin(C.STYLES).all(), out[~out['style'].isin(C.STYLES)][['row', 'style']]
    for h, names in (('fabric', C.FABRICS), ('colour', C.COLOURS)):
        bad = {v for s in out[h] for v in s.split('|') if v and v not in names}
        assert not bad, (h, bad)
    return out


if __name__ == '__main__':
    out = build()
    out.to_csv(C.BATCH2_LABELS, index=False)
    print(f'{len(out)} photos -> {C.BATCH2_LABELS}')
    for h in ('style', 'fabric', 'colour'):
        print(h, pd.Series([v for s in out[h] for v in s.split('|') if v]).value_counts().to_dict())
    print('no colour label:', list(out.loc[out.colour == '', 'row']))
