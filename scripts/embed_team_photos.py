"""Embed the team's outfit photos (dataset_images_1/) once with the frozen tagger, after cropping
each to the person. Used by the app for the tagger demo, the concept board and the near-copy check.
Needs no labels.

    python scripts/embed_team_photos.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpt import config as C  # noqa: E402
from fpt.tagger import FashionEmbedder  # noqa: E402

paths = sorted(str(p) for p in C.TEAM_DIR.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
vecs = FashionEmbedder().images(paths)
out = C.OUT / 'tagger'
out.mkdir(parents=True, exist_ok=True)
np.savez(out / 'team_embeddings.npz', vecs=vecs, paths=np.array([Path(p).name for p in paths]))   # names: portable
print(f'{len(paths)} team photos embedded -> {out / "team_embeddings.npz"}')
