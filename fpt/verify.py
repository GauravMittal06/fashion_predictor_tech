"""Layer 3 checks against REAL photos (the team's 195 outfit screenshots).

  calibrate()  : how close real outfit photos are to their nearest other real photo. Two thresholds
                 come from this, not from guesses:
                   near_copy = 99th percentile  (only duplicate screenshots are this close)
                   realism   = 5th percentile   (95% of real photos have a neighbour at least this close)
  check()      : every concept must (1) show the brief's attributes - its style in the tagger's top 2,
                 each colour / fabric in the top 2 (the rule whose accuracy we measured), (2) look real,
                 (3) not be a near-copy. A print/pattern check is reported but cannot reject a concept,
                 because the team photos have no print labels to measure its accuracy.
"""
import numpy as np
from . import config as C
from .tagger import zero_shot_ens


def calibrate(real_vecs):
    S = real_vecs @ real_vecs.T
    np.fill_diagonal(S, -1)
    nn = S.max(1)
    return {'near_copy': float(np.percentile(nn, 99)), 'realism': float(np.percentile(nn, 5)),
            'median': float(np.median(nn)), 'nn': nn}


def top2(p, names, want):
    return want in [names[i] for i in np.argsort(-p)[:2]]


def check(gen_vecs, brief, emb, real_vecs, cal):
    probs = zero_shot_ens(emb, gen_vecs)
    S = gen_vecs @ real_vecs.T
    out = []
    for i in range(len(gen_vecs)):
        detail = {}
        if brief.get('style'):
            detail[f"style {brief['style']}"] = top2(probs['style'][i], C.STYLES, brief['style'])
        for f in brief.get('fabrics', []):
            detail[f'fabric {f}'] = top2(probs['fabric'][i], C.FABRICS, f)
        for c in brief.get('colours', []):
            detail[f'colour {c}'] = top2(probs['colour'][i], C.COLOURS, c)
        info = {}
        if brief.get('motif'):                        # shown, but not used to reject: no print labels to measure it
            info['print (not measured)'] = bool(probs['motif'][i, 0] >= 0.5)
        nearest = float(S[i].max())
        rank = float((cal['nn'] <= nearest).mean())       # where it sits among real photos' neighbours
        ok_attr = all(detail.values())
        real = nearest >= cal['realism']
        copy = nearest >= cal['near_copy']
        out.append({'index': i, 'detail': detail, 'info': info, 'matched': sum(detail.values()), 'needed': len(detail),
                    'matches_brief': ok_attr, 'nearest_real': round(nearest, 3), 'realism_percentile': round(rank, 2),
                    'looks_real': real, 'near_copy': copy, 'kept': ok_attr and real and not copy,
                    'nearest_index': int(S[i].argmax())})
    return out
