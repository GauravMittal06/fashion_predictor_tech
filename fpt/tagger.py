"""Layer 1, Plan A: Marqo-FashionSigLIP (a ~150M-parameter vision transformer trained on 1M+
fashion products), frozen, used two ways.

  zero-shot    : compare each photo with text phrases (no training at all)
  linear probe : a small logistic-regression classifier per attribute on frozen embeddings

Ported from the team's tagger_clip.py; adds the GeoStyle trend attributes so the same model can
check generated designs against the attributes the forecaster called rising.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from . import config as C

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------- Person crop (pretrained Faster R-CNN, COCO) ----------------
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                                  FasterRCNN_MobileNet_V3_Large_FPN_Weights)
        _detector = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).to(DEVICE).eval()
    return _detector


@torch.no_grad()
def crop_person(img, min_score=0.6, margin=0.08):
    """Return (largest detected person with a small margin, box or None); whole photo if none found."""
    from torchvision.transforms.functional import to_tensor
    img = img.convert('RGB')
    out = get_detector()([to_tensor(img).to(DEVICE)])[0]
    keep = (out['labels'] == 1) & (out['scores'] >= min_score)      # COCO label 1 = person
    if not keep.any():
        return img, None
    boxes = out['boxes'][keep]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x0, y0, x1, y1 = boxes[areas.argmax()].tolist()
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    W, H = img.size
    box = (max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy))
    return img.crop(box), box


# ---------------- Frozen fashion image-text model ----------------
class FashionEmbedder:
    def __init__(self, name=C.TAGGER_MODEL):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name)
        self.tokenizer = open_clip.get_tokenizer(name)
        self.device = DEVICE
        self.model = self.model.to(self.device).eval()
        self._text_cache = {}

    def to(self, device):
        """Move the model off the GPU (e.g. while the image generator needs the memory) and back."""
        self.device = device
        self.model.to(device)
        if device == 'cpu' and DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def images(self, imgs, crop=True, batch=16, flip=False):
        """L2-normalised image embeddings, one row per photo (PIL images or file paths).
        flip=True also looks at the mirrored photo and averages the two views."""
        out = []
        for i in range(0, len(imgs), batch):
            ims = [Image.open(p).convert('RGB') if isinstance(p, (str, Path)) else p.convert('RGB')
                   for p in imgs[i:i + batch]]
            if crop:
                ims = [crop_person(im)[0] for im in ims]
            e = self._encode(ims)
            if flip:
                e = e + self._encode([im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims])
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def _encode(self, ims):
        x = torch.stack([self.preprocess(im) for im in ims]).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1)

    @torch.no_grad()
    def texts(self, phrases):
        key = tuple(phrases)
        if key not in self._text_cache:
            e = self.model.encode_text(self.tokenizer(list(phrases)).to(self.device))
            self._text_cache[key] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        return self._text_cache[key]

    def scale(self):
        return float(self.model.logit_scale.exp().item())


class FashionCLIPEmbedder(FashionEmbedder):
    """Plan B: FashionCLIP 2.0 (CLIP ViT-B/32 fine-tuned on ~700k Farfetch products), same interface."""

    def __init__(self, name=C.TAGGER_MODEL_B):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.device = DEVICE
        self.model = self.model.to(self.device).eval()
        self._text_cache = {}

    @torch.no_grad()
    def _encode(self, ims):
        x = self.processor(images=ims, return_tensors='pt').to(self.device)
        return torch.nn.functional.normalize(_as_tensor(self.model.get_image_features(**x)).float(), dim=-1)

    @torch.no_grad()
    def texts(self, phrases):
        key = tuple(phrases)
        if key not in self._text_cache:
            x = self.processor(text=list(phrases), return_tensors='pt', padding=True, truncation=True).to(self.device)
            e = _as_tensor(self.model.get_text_features(**x))
            self._text_cache[key] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        return self._text_cache[key]


def _as_tensor(out):
    """transformers 5 returns an output object; the projected embedding is its pooler_output."""
    return out if torch.is_tensor(out) else out.pooler_output


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def zero_shot(emb, img_vecs):
    """Probabilities for all six heads from text phrases only (no training)."""
    s = emb.scale()
    pick = lambda txt: [f'a photo of a person wearing {t}' for t in txt]
    out = {'style': _softmax(s * img_vecs @ emb.texts(pick(C.STYLE_TEXT.values())).T)}
    for head, table in (('fabric', C.FABRIC_TEXT), ('colour', C.COLOUR_TEXT)):
        cols = []
        for phrase in table.values():   # each class: "with X" against a neutral outfit
            t = emb.texts([f'a photo of an outfit made of {phrase}' if head == 'fabric'
                           else f'a photo of a {phrase} outfit', 'a photo of an outfit'])
            cols.append(_softmax(s * img_vecs @ t.T)[:, 0])
        out[head] = np.stack(cols, 1)
    for cue, (yes, no) in C.CUE_TEXT.items():
        t = emb.texts([f'a photo of an outfit with {yes}', f'a photo of an outfit with {no}'])
        out[cue] = _softmax(s * img_vecs @ t.T)[:, :1]
    return out


# Several descriptions per class, averaged ("prompt ensembling", as in the CLIP paper)
TEMPLATES = ['a photo of a person wearing {}', 'a street style photo of a person in {}',
             'an instagram outfit photo: {}', 'a fashion photo of {}']
STYLE_WORDS = {'Evening/Party': ['an evening party outfit', 'a cocktail dress for a night out', 'a glamorous party look'],
               'Casual': ['a casual everyday outfit', 'relaxed casual clothes', 'a simple day-to-day look'],
               'Street': ['a streetwear outfit', 'urban street style clothes', 'an oversized sporty street look'],
               'Resort/Swim': ['a resort beachwear outfit', 'swimwear at the beach', 'a summer holiday look'],
               'Tailored/Suiting': ['a tailored suit outfit', 'a smart blazer and trousers', 'formal office wear'],
               'Winter/Outerwear': ['a winter coat outfit', 'a warm jacket and layers', 'cold weather outerwear']}
FABRIC_WORDS = {'Denim': ['denim', 'jeans fabric'], 'Leather': ['leather', 'faux leather'],
                'Knit': ['knitwear', 'a knitted sweater'], 'Satin/Silk': ['satin or silk', 'shiny silky fabric'],
                'Cotton/Linen': ['cotton or linen', 'light linen fabric'], 'Wool/Tweed': ['wool or tweed', 'a wool coat'],
                'Sequin/Metallic': ['sequins', 'sparkly metallic fabric'], 'Lace/Mesh/Sheer': ['lace or sheer fabric', 'see-through mesh'],
                'Fur/Shearling': ['fur or shearling', 'a fluffy faux fur coat'], 'Jersey/Stretch': ['stretch jersey', 'a tight stretchy top']}


def _class_text(emb, words):
    """Average text embedding over templates x descriptions for one class."""
    t = emb.texts([tpl.format(w) for w in words for tpl in TEMPLATES]).mean(0)
    return t / np.linalg.norm(t)


def zero_shot_ens(emb, img_vecs):
    """Like zero_shot(), with several descriptions per class averaged; same output format."""
    s = emb.scale()
    neutral = _class_text(emb, ['an outfit', 'clothes'])
    out = {'style': _softmax(s * img_vecs @ np.stack([_class_text(emb, STYLE_WORDS[c]) for c in C.STYLES]).T)}
    cols = []
    for f in C.FABRICS:
        T = np.stack([_class_text(emb, [w + ' clothes' for w in FABRIC_WORDS[f]]), neutral])
        cols.append(_softmax(s * img_vecs @ T.T)[:, 0])
    out['fabric'] = np.stack(cols, 1)
    cols = []
    for c in C.COLOURS:
        words = [f'{w} clothes' for w in c.split('/')] + [f'a {c.replace("/", " or ")} outfit']
        T = np.stack([_class_text(emb, words), neutral])
        cols.append(_softmax(s * img_vecs @ T.T)[:, 0])
    out['colour'] = np.stack(cols, 1)
    for cue, (yes, no) in C.CUE_TEXT.items():
        T = np.stack([_class_text(emb, [f'an outfit with {yes}']), _class_text(emb, [f'an outfit with {no}'])])
        out[cue] = _softmax(s * img_vecs @ T.T)[:, :1]
    return out


def trend_attributes(emb, img_vecs, attrs):
    """Zero-shot scores for GeoStyle trend attributes ('Colour: White', 'Hat', ...).
    Multi-class groups (Colour, Pattern, ...) are scored against the other classes of the group;
    yes/no attributes against their absence. Returns {attribute: probability per image}."""
    s = emb.scale()
    groups = {}
    for a in attrs:
        g, _, v = a.partition(': ')
        groups.setdefault(g, []).append(v)
    out = {}
    for g, values in groups.items():
        if values == ['']:                             # yes/no attribute such as 'Hat'
            phrase = C.GEO_TEXT[g](None)
            t = emb.texts([f'a photo of a person wearing {phrase}',
                           f'a photo of a person not wearing {phrase}'])
            out[g] = _softmax(s * img_vecs @ t.T)[:, 0]
        else:
            t = emb.texts([f'a photo of a person wearing {C.GEO_TEXT[g](v)}' for v in values])
            p = _softmax(s * img_vecs @ t.T)
            for j, v in enumerate(values):
                out[f'{g}: {v}'] = p[:, j]
    return out


class LinearProbe:
    """One small classifier per attribute on frozen embeddings (about a thousand weights each)."""

    def fit(self, X, y, C_reg=1.0):
        self.models = {'style': LogisticRegression(C=C_reg, max_iter=2000, class_weight='balanced')
                       .fit(X, y['style'].argmax(1))}
        for head in ['fabric', 'colour'] + C.CUES:
            Y = y[head]
            self.models[head] = []
            for j in range(Y.shape[1]):
                pos = int(Y[:, j].sum())
                if pos < 2 or pos > len(Y) - 2:      # too few examples: leave this class to zero-shot
                    self.models[head].append(None)
                else:
                    self.models[head].append(LogisticRegression(
                        C=C_reg, max_iter=2000, class_weight='balanced').fit(X, Y[:, j]))
        return self

    def predict(self, X, fallback=None):
        out = {}
        m = self.models['style']
        p = np.zeros((len(X), len(C.STYLES)))
        p[:, m.classes_] = m.predict_proba(X)
        out['style'] = p
        for head in ['fabric', 'colour'] + C.CUES:
            cols = []
            for j, m in enumerate(self.models[head]):
                if m is None:
                    cols.append(fallback[head][:, j] if fallback else np.full(len(X), np.nan))
                else:
                    cols.append(m.predict_proba(X)[:, 1])
            out[head] = np.stack(cols, 1)
        return out


def head_names(head):
    return {'style': C.STYLES, 'fabric': C.FABRICS, 'colour': C.COLOURS}.get(head, [head])
