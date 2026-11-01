# Results summary: Fashion Prediction Technology MVP (1 November 2026)

Every number below was measured on data the model did not see when it was built or chosen.

## 1. Tagger: what is in an outfit photo?

**Data:** 295 labelled photos from 30 Instagram pages:
- **batch 1:** the team's 195 Southern Europe screenshots (20 pages), labels from `SOUTHERN EUROPE DATA.xlsx`;
- **batch 2:** 100 influencer photos (10 pages, 10 each), labels mapped from the team's free-text CSV by
  `scripts/build_labels_batch2.py`. Every photo was checked by eye; the 29 corrections are listed with reasons in
  that script, and were made before any model saw these photos.

**Test:** nested 5-fold cross-validation by Instagram page. Every photo is tagged by a model that never saw its
page, and the recipe is chosen inside each fold.

| Attribute | Main guess right | Range across folds | True answer in top 2 | Always guess the most common |
|---|---|---|---|---|
| Style (6 classes) | **58%** | 46–72% | **83%** | 25% |
| Fabric (10 classes) | **69%** | 61–75% | **90%** | 47% |
| Colour (10 classes) | **71%** | 63–80% | **86%** | 47% |

**By batch** (same run): batch 1 style 51% (top 2 78%), fabric 65% (87%), colour 62% (80%); batch 2 style 72% (94%),
fabric 77% (97%), colour 89% (96%). Batch 2 photos are clearer and carry more labels each, so its higher scores are
partly easier photos. **The rise from the earlier 49% / 64% / 64% is therefore not all model improvement**; the
same-photo comparison below is the fair test.

**Did the 100 new photos help?** The shipped recipe, the same held-out pages, probes trained without vs with batch 2
(`scripts/check_batch2_gain.py`):

| Photos | Attribute | Main guess: before → after | Top 2: before → after |
|---|---|---|---|
| Batch 1 (195) | Style | 54% → 54% | 79% → 78% |
| Batch 1 (195) | Fabric | 65% → **66%** | 84% → **86%** |
| Batch 1 (195) | Colour | 65% → 65% | 81% → 81% |
| Batch 2 (100) | Style | 72% → **74%** | 88% → **90%** |
| Batch 2 (100) | Fabric | 80% → 79% | 95% → **97%** |
| Batch 2 (100) | Colour | 89% → 89% | 95% → 95% |

- **Gains:** fabric top 2 on both batches; style on new pages, notably **Tailored/Suiting 8 → 10 of 12** (batch 1
  had no Tailored examples, so the old probe could not learn that style) and Winter/Outerwear 9 → 10 of 13.
- **No real losses:** each drop is about one photo. Colour uses zero-shot only, so it is the same model.
- The nested-CV version of this comparison (which also re-chooses the recipe per fold) is in
  `outputs/tagger/batch_comparison.csv`. It is noisier: style +3 and fabric +1 points on batch 1, colour −2 points
  from a different recipe pick in some folds.
- **Sanity check:** rebuilt from batch 1 alone, the script reproduces the earlier results exactly (style 48.7%,
  fabric 63.9%, colour 64.5%; same recipe).
- **Frozen recipe (unchanged):** Marqo-FashionSigLIP zero-shot (several descriptions per class + mirrored view), plus
  a linear probe, now trained on all 295 photos, for style and fabric.
- **Earlier results** (batch 1 only) are kept in `outputs/tagger/before_batch2/`.
- **Style is still the weak spot,** especially Casual vs Evening/Party on batch 2 (Casual 5 of 17).
- **Region:** batch 2's region column is empty and its scenes include New York, London and Paris. It is used for the
  tagger only. The designer's real-photo checks still use the 195 Southern Europe photos.

## 2. Forecaster: will a fashion search rise or fade over the next 6 months?

**Data:** Google Trends search interest, Nov 2019 – Oct 2026, for 32 Spanish and 20 Italian fashion searches.

**Frozen first:** on Spain's data up to Dec 2023 only, the LSTM was retrained and **Chronos-2 + LSTM** was chosen.
It was best on 21 earlier forecasts, with 12% less error than "same month last year".

**Replays:** from every month Jan 2024 – May 2026 it forecast the next 6 months using earlier data only.
**Italy was never used to build anything.**

| | Spain | **Italy (untouched)** |
|---|---|---|
| Less error than "same month last year" | **16.5%** | **9.6%** |
| Direction right (baseline) | 60% (49%) | 64% (55%) |
| Truth inside the 80% range | 80% | 77% |
| Rising/fading calls that came true | **24 of 29 (83%)** | **18 of 20 (90%)** |

Example: in Jan 2025 the system forecast **linen trousers** at +25% versus the year before. What really happened was +22%.

**Second proof, on street photos:** 96,965 dated Barcelona photos, 2013–2018. Chronos-2 made 18% less error than
"same month last year" on the most recent year, and its rising/fading calls came true 8 of 8 times.

**Forecast for Nov 2026 – Apr 2027:**
- **Spain:** *linen shirt* rising (+33%). Leaning up: striped dress (+26%), linen trousers (+19%), leather trousers
  (+18%), shearling jacket (+15%). Leaning down: brown dress (−14%).
- **Italy:** *linen trousers* rising (+54%), *beige dress* rising (+23%). Leaning up: linen shirt (+28%) and linen
  dress (+24%).

## 3. Designer: new concepts from the forecast, checked against real photos

The thresholds come from the 195 real photos:
- **Looks real:** the concept's closest real photo is at least 0.73 similar. 95% of real photos reach this.
- **Not a copy:** the closest real photo is below 0.95 similar. Only the duplicate screenshots are that close.

**Spring 2027, Spain:**
- Brief: a green striped linen shirt, built from the rising searches.
- SDXL-Turbo concepts: **12 of 12 kept**. All show linen and green, all look real, and none is a copy.

**2025 generation replay:**
- Brief from the forecast made in Jan 2025 (data up to Dec 2024): knit, brown, polka dot.
- **9 of 12** concepts look more like the team's real 2025 photos than their 2024 photos. A control prompt with no
  forecast scored **0 of 12**.
- Only 31 photos per year, so this result is indicative.

**Honest notes:**
- The prompt wording was revised four times while watching the checks; the checks never changed. Asking for two
  fabrics at once made SDXL-Turbo draw silk-like shirts.
- The print check is shown but cannot reject a concept, because there are no print labels to measure it.
- The IP-Adapter real-photo style reference was downloaded but not evaluated.

## Limits

- **Small data:** 295 tagger photos, so results vary by about ±10 points between folds.
- **Search interest is a proxy for demand, not sales.**
- **Untouched-country test:** Italy only. Google Trends rate limits stopped the Portugal download after one sparse
  group.
- **Licence:** SDXL-Turbo is non-commercial. FLUX.1-schnell (Apache 2.0) would replace it for brand use.
