# Fashion Prediction Technology: MVP

Team: Gaurav Chandra Mittal, Hari Krishna Tiwari · Guide: Dr. D.P. Sharma

A local web app that runs all four layers end to end on real data. It:
1. **tags** an outfit photo;
2. **forecasts** whether each of its details will rise or fade in Spain or Italy;
3. **designs** a next-season version (keeping what rises, swapping what fades) and checks it against real photos;
4. writes a **brand report**.

The final numbers are in **[RESULTS.md](RESULTS.md)**, the plan in [PLAN.md](PLAN.md), and the demo script in
[DEMO_GUIDE.md](DEMO_GUIDE.md).

## Deep learning models

| Layer | Model | Type | Trained by us? |
|---|---|---|---|
| 1 · Tagger | Faster R-CNN MobileNetV3 (finds the person) | CNN detector | no (COCO) |
| 1 · Tagger | Marqo-FashionSigLIP, FashionCLIP 2.0 | vision transformers (image–text) | linear probe on our 295 labelled photos |
| 2 · Forecaster | Chronos-2 (Amazon, 120M) | time-series transformer | no (zero-shot) |
| 2 · Forecaster | Quantile LSTM (12 months → 64 cells → 6 months × low/middle/high) | recurrent network | **yes**: 44-city street-style trends + Spain searches up to 2023 |
| 3 · Designer | SDXL-Turbo | diffusion model | no |

> **Private repository.** It contains the team's and influencers' photos and their Instagram labels. Remove
> `dataset_images_1/`, `dataset_images_2/`, `data/barcelona_photos/` and the label files before making it public.

## Set up (once)

Needs Python 3.13 and, for image generation, an NVIDIA GPU with 6 GB or more (tested on an RTX 3050 Laptop).

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_models.py
```

The last step downloads the five pretrained models from Hugging Face (about 10 GB, once).

## Run the app

Every result file is in `outputs/`, so the app works straight after setup:

```bash
streamlit run app.py
```

Two pages:
- **Pipeline:** one photo flows through every layer. Upload a photo (or pick a team or Barcelona photo) and the app:
  1. tags its style, fabric, colour and print (sources: upload, team photo, influencer photo, Barcelona street photo);
  2. links each tag to the matching Google searches and shows their 2027 forecast, range and 2024–26 track record
     (Spain or Italy);
  3. writes a brief: it keeps what is trending and swaps what is fading for the strongest rising option;
  4. draws concepts with SDXL-Turbo and checks each against the 195 real photos;
  5. builds a downloadable report that starts with the photo.

  Use **Correct a tag** to fix the tagger, and every later step follows the correction.
- **Evidence:** how each layer was tested: forecast replays 2024–26, all 2027 calls, tagger cross-validation, the
  designer evaluation, and the Barcelona street-photo proof.

## Rebuild every result from scratch

After the setup above, each script reads the previous step's output. Close the app first (the GPU is shared).
Run them in this order:

```bash
python scripts/download_data.py
```

```bash
python scripts/build_trends.py
```

```bash
python scripts/check_tagger_barcelona.py
```

```bash
python scripts/embed_team_photos.py
```

```bash
python scripts/match_labels.py
```

```bash
python scripts/build_labels.py
```

```bash
python scripts/build_labels_batch2.py
```

```bash
python scripts/cv_tagger.py
```

```bash
python scripts/check_batch2_gain.py
```

```bash
python scripts/build_replays.py
```

```bash
python scripts/build_designer_eval.py
```

What each one does:
- `download_data.py`: GeoStyle Barcelona metadata and the 44-city style trends (about 40 MB, into `data/raw/`).
- `build_trends.py`: Barcelona street-photo series, LSTM pre-training and backtests. Takes about 4 minutes.
- `check_tagger_barcelona.py`: screens the 72 Barcelona sample photos in `data/barcelona_photos/` and measures
  agreement with GeoStyle's labels. (`fetch_barcelona_photos.py` downloaded that sample from Flickr; it is included,
  so you do not need to run it.)
- `embed_team_photos.py`: the batch 1 photo vectors used by the app and the designer checks.
- `match_labels.py` and `build_labels.py`: match the screenshots to the spreadsheet rows and build `labels.csv`.
- `build_labels_batch2.py`: maps batch 2's free-text CSV to the tagger's classes (corrections checked by eye are
  listed in the script).
- `cv_tagger.py`: nested cross-validation of the tagger on both batches, a same-photo before/after comparison,
  then freezes the recipe.
- `check_batch2_gain.py`: did batch 2 help the shipped recipe? Same held-out pages, trained without vs with it.
- `build_replays.py`: freezes the forecaster on data up to Dec 2023, then replays 2024–2026 for Spain and Italy.
- `build_designer_eval.py`: Spring 2027 concepts, real-photo checks and the 2025 generation replay. The optional
  IP-Adapter step (a real photo as style reference) is off unless `USE_REF=1` is set; it has not been evaluated.

**Label check (recommended):** open `review/label_review.html` (made by `python scripts/make_review_sheet.py`).
Correct the labels by eye, tick button/zipper/print, and save `labels_reviewed.csv` into this folder. Every script
then uses it automatically; re-run `cv_tagger.py`.

**Refreshing Google Trends:** open each group of terms in Google Trends (country, Nov 2019 – today, monthly) and
save it in the same CSV format under `data/google_trends/` (Spain), `data/google_trends/IT/` or
`data/google_trends/PT/`.

## Data

- **Team photos (batch 1):** 195 outfit screenshots (Southern Europe, 2023–2026) with `SOUTHERN EUROPE DATA.xlsx`;
  `data/screenshot_matches.csv` links each screenshot to its spreadsheet row.
- **Batch 2:** 100 influencer photos from 10 Instagram pages (`dataset_images_2/photos/`) with the team's CSV, used
  to train and test the tagger.
- **Barcelona sample:** 72 public Flickr photos from GeoStyle (`data/barcelona_photos/`), resized to 640 px.
- **Google Trends** (`data/google_trends/`, downloaded by hand): monthly search interest, Nov 2019 – Oct 2026:
  - 32 Spanish fashion searches;
  - 20 Italian ones (the untouched test country);
  - 5 Portuguese ones, collected but not used (too sparse, and Google rate-limited further downloads).
- **GeoStyle (Cornell, ICCV 2019):** 96,965 dated Barcelona street photos tagged with 12 attributes (2013–2018),
  plus 50 styles × 44 cities from *From Paris to Berlin* (CVPR 2020) to pre-train the LSTM.

## How leakage is prevented

- **Tagger:** folds split by Instagram page; the recipe is chosen inside each fold.
- **Forecaster:**
  - every forecast uses only earlier months;
  - the model was frozen on data up to Dec 2023 before 2024–2026 were scored;
  - Italy was never used to build anything;
  - the LSTM never saw Barcelona.
- **Designer:** the realism and copy thresholds come from real photos only. The 2025 generation replay uses a
  forecast made from data up to Dec 2024.

## Folder layout

```
app.py              Streamlit app
fpt/                config, geodata (GeoStyle), tagger, forecast, design, verify, report
scripts/            build steps listed above
data/               Google Trends CSVs, Barcelona photo sample, screenshot-to-row matches (raw/ is downloaded)
dataset_images_1/   batch 1: the team's 195 screenshots + spreadsheet
dataset_images_2/   batch 2: 100 influencer photos (photos/) + their CSV
labels.csv          batch 1 labels (build_labels.py);  labels_batch2.csv: batch 2 labels (build_labels_batch2.py)
outputs/            every result file the app reads (trends/, replay/, tagger/, designer/)
Fashion_Prediction_Technology_Run_and_Test_It_Yourself.pdf   walkthrough of the app
```

## Differences from the synopsis

- **Data:** Google Trends and public research data stand in for the team's own dated photo collection, which
  doesn't exist yet.
- **LSTM:** written in PyTorch instead of Keras, so it can train on the laptop GPU under Windows.
- **Designer:** SDXL-Turbo instead of SDXL base, so it fits the 6 GB GPU. Its licence is non-commercial.
- **EfficientNetB0 tagger (Plan C):** not used. The frozen-model recipes did better and need no GPU training.
