"""Build labels.csv for the team's screenshots from SOUTHERN EUROPE DATA.xlsx.

Each screenshot takes the labels of the spreadsheet row it was matched to (scripts/match_labels.py).
The free-text colour / fabric / style columns become fixed classes with the team's whole-word keyword
rules (reference/.../prepare_labels.py). The sheet has no button / zipper / print column, so those
heads are left empty and skipped in training. The Instagram page is the group for the split.

    python scripts/build_labels.py
"""
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402

TYPOS = {'modren': 'modern', 'street wear': 'streetwear', 'cotto ': 'cotton ', 'cotto,': 'cotton,',
         'lilen': 'linen', 'furr': 'fur', 'polyster': 'polyester', 'shiffon': 'chiffon',
         'crape': 'crepe', 'eslastane': 'elastane', 'mahroon': 'maroon'}
STYLE_RULES = [  # first match wins
    ('Resort/Swim', ['resort', 'swim', 'beach']),
    ('Evening/Party', ['evening', 'party', 'gown', 'cocktail']),
    ('Tailored/Suiting', ['suit', 'suiting', 'tailored']),
    ('Winter/Outerwear', ['winter', 'outerwear', 'coat']),
    ('Street', ['street', 'streetwear']),
    ('Casual', ['casual', 'active', 'preppy', 'autumn']),
]
FABRIC_RULES = [
    ('Denim', ['denim']), ('Leather', ['leather', 'lambskin', 'suede', 'faux']),
    ('Knit', ['knit', 'ribbed', 'rib', 'crochet']), ('Satin/Silk', ['satin', 'silk', 'brocade']),
    ('Cotton/Linen', ['cotton', 'linen', 'poplin', 'twill', 'canvas', 'corduroy', 'ramie']),
    ('Wool/Tweed', ['wool', 'tweed', 'gabardine', 'flannel', 'mohair']),
    ('Sequin/Metallic', ['sequin', 'metallic', 'lurex']),
    ('Lace/Mesh/Sheer', ['lace', 'mesh', 'tulle', 'chiffon', 'organza', 'sheer', 'georgette', 'voile', 'gauze']),
    ('Fur/Shearling', ['fur', 'shearling', 'fleece']),
    ('Jersey/Stretch', ['jersey', 'spandex', 'elastane', 'stretch', 'nylon', 'modal']),
]
COLOUR_RULES = [
    ('black', ['black']), ('white/cream', ['white', 'off white', 'cream', 'ivory']),
    ('beige', ['beige', 'tan', 'camel', 'nude', 'sand', 'khaki']),
    ('red', ['red', 'burgundy', 'maroon', 'wine']), ('blue', ['blue', 'navy', 'teal']),
    ('pink', ['pink', 'blush', 'fuchsia']), ('green', ['green', 'olive']),
    ('brown', ['brown', 'chocolate']), ('grey', ['grey', 'gray', 'charcoal']),
    ('metallic', ['metallic', 'gold', 'golden', 'silver']),
]


def compile_rules(rules):
    return [(n, re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in ws) + r')(?:s|es)?\b')) for n, ws in rules]


STYLE_RE, FABRIC_RE, COLOUR_RE = map(compile_rules, [STYLE_RULES, FABRIC_RULES, COLOUR_RULES])


def clean(text):
    if pd.isna(text):
        return ''
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    text = ' '.join(re.sub(r'[,./]', ', ', text).split()) + ' '
    for wrong, right in TYPOS.items():
        text = text.replace(wrong, right)
    return text


def many(text, rules):
    t = clean(text)
    return '|'.join(n for n, p in rules if p.search(t))


def style(text):
    t = clean(text)
    return next((n for n, p in STYLE_RE if p.search(t)), None)


sheet = pd.read_excel(C.TEAM_DIR / 'SOUTHERN EUROPE DATA.xlsx')
sheet['row'] = range(1, len(sheet) + 1)
m = pd.read_csv(C.ROOT / 'data' / 'screenshot_matches.csv')[['image', 'row']].merge(sheet, on='row')
labels = pd.DataFrame({
    'image': m['image'], 'group': m['INSTAGRAM PAGE'].str.strip(), 'row': m['row'],
    'style': m['STYLE'].apply(style), 'fabric': m['FABRIC'].apply(lambda t: many(t, FABRIC_RE)),
    'colour': m['COLOUR'].apply(lambda t: many(t, COLOUR_RE)),
    'button': pd.NA, 'zipper': pd.NA, 'motif': pd.NA,   # not in the sheet: to be labelled by hand
    'region': m['REGION'], 'date': m['DATE'].astype(str),
})
dropped = labels['style'].isna().sum()
labels = labels.dropna(subset=['style'])
labels.to_csv(C.LABEL_FILE, index=False)
print(f'{len(labels)} labelled screenshots from {labels.row.nunique()} posts, {labels.group.nunique()} '
      f'Instagram pages ({dropped} dropped: no style in the sheet)')
print(labels['style'].value_counts().to_string())
for col in ('fabric', 'colour'):
    print(col, labels[col].str.split('|').explode().replace('', pd.NA).dropna().value_counts().to_dict())
