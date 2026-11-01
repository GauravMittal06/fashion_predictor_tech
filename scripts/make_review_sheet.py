"""Build review/label_review.html: a local page to check every team photo's labels by eye.

The page shows each photo with the labels from the spreadsheet (never the model's guesses, so the check
cannot flatter the model). The team corrects style / fabric / colour, ticks button / zipper / print, and
clicks "Save CSV" to download labels_reviewed.csv; put that file in the project folder and run
python scripts/build_labels.py --reviewed  (or train_tagger / cv_tagger pick it up automatically).

The page stays on this laptop: photos are linked from dataset_images_1/, not copied or published.

    python scripts/make_review_sheet.py
"""
import html
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402

OUT = C.ROOT / 'review'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(C.LABEL_FILE).fillna('')
rows = df.to_dict('records')

cards = []
for i, r in enumerate(rows):
    style_opts = ''.join(f'<option{" selected" if s == r["style"] else ""}>{html.escape(s)}</option>' for s in C.STYLES)
    fab = ''.join(f'<label><input type="checkbox" data-k="fabric" value="{html.escape(f)}"'
                  f'{" checked" if f in str(r["fabric"]).split("|") else ""}>{html.escape(f)}</label>' for f in C.FABRICS)
    col = ''.join(f'<label><input type="checkbox" data-k="colour" value="{html.escape(c)}"'
                  f'{" checked" if c in str(r["colour"]).split("|") else ""}>{html.escape(c)}</label>' for c in C.COLOURS)
    cue = ''.join(f'<label class="cue"><input type="checkbox" data-k="{c}">{c}</label>' for c in C.CUES)
    src = f'../{C.IMAGE_DIR.name}/' + r['image']
    cards.append(f'''<div class="card" data-i="{i}">
<img src="{html.escape(src)}" loading="lazy">
<div class="f"><b>#{i + 1}</b> · {html.escape(str(r["group"]))} · {html.escape(str(r.get("region", "")))}
<label class="bad"><input type="checkbox" data-k="wrong_post"> photo is not this post / not an outfit</label>
<div>Style <select data-k="style">{style_opts}</select></div>
<div class="g"><span>Fabric</span>{fab}</div><div class="g"><span>Colour</span>{col}</div>
<div class="g"><span>Visible</span>{cue}</div></div></div>''')

page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Label review</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1c2532}}
header{{position:sticky;top:0;background:#1c2532;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:2}}
button{{font-size:15px;padding:6px 14px;border-radius:6px;border:0;background:#e3a33b;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;padding:12px}}
.card{{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0002}}
.card img{{width:100%;height:340px;object-fit:contain;background:#222}}
.f{{padding:8px 10px;font-size:13px;line-height:1.7}}.g span{{font-weight:600;margin-right:4px}}
.g label{{margin-right:8px;white-space:nowrap}}.bad{{color:#b42318;display:block}}.cue{{margin-right:10px}}
</style></head><body><header><b>Label review · {len(rows)} photos</b>
<span>Check each photo: fix style, fabric and colour if wrong; tick button / zipper / print if visible.</span>
<button onclick="save()">Save CSV</button></header><div class="grid">{"".join(cards)}</div>
<script>const ROWS={json.dumps(rows)};
function save(){{const cols=['image','group','row','style','fabric','colour','button','zipper','motif','region','date','wrong_post'];
const out=[cols.join(',')];
document.querySelectorAll('.card').forEach(c=>{{const r={{...ROWS[c.dataset.i]}};
r.style=c.querySelector('[data-k=style]').value;
for(const k of ['fabric','colour'])r[k]=[...c.querySelectorAll(`[data-k=${{k}}]:checked`)].map(x=>x.value).join('|');
for(const k of ['button','zipper','motif','wrong_post'])r[k]=c.querySelector(`[data-k=${{k}}]`).checked?1:0;
out.push(cols.map(k=>'"'+String(r[k]??'').replace(/"/g,'""')+'"').join(','));}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.join('\\n')],{{type:'text/csv'}}));
a.download='labels_reviewed.csv';a.click();}}</script></body></html>'''
(OUT / 'label_review.html').write_text(page, encoding='utf-8')
print(f'{len(rows)} photos -> {OUT / "label_review.html"}')
