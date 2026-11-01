# MVP plan: Fashion Prediction Technology, end to end with real numbers

**Constraint:** no new photos. We use the team's 195 labelled screenshots (`dataset_images_1/`), the public
Barcelona street-photo data (GeoStyle, already downloaded) and Google Trends search data (numbers, not photos).

**Golden rule (no leakage):** every model and setting is **frozen before** it is scored on the data used to
report results. Each result below says which data was used to choose and which was used only to score.

## What the finished MVP proves

| Goal | How it is proven | Realistic target |
|---|---|---|
| 1. Tag accurately | 5-fold cross-validation by Instagram page on all 195 photos, with the method chosen *inside* each fold (nested), so every photo is scored by a model that never saw its page | style right ≥ 65%; true colour in top 2 ≥ 80% |
| 2. Forecast accurately | "Replays": pretend it is Jan 2024 (and Jan 2025), forecast, then compare with what really happened; plus **Italy and Portugal locked away completely** as untouched countries | less error than "same as last year"; truth inside the 80% range ~75–85%; most rising/fading calls come true |
| 3. Lifelike, verified generation | Every concept checked against real photos: attributes match the brief, looks as close to real outfits as real outfits are to each other, not a copy (threshold calibrated on real duplicate screenshots); plus a 2025 generation replay against the team's real 2025 photos | ≥ 75% of concepts pass; 0 copies |

## Steps

### 1. Tagger (about 3 h)
1. **Label check (team, 20 min, optional but valuable):** I list the screenshots whose match to the spreadsheet
   row was weakest (from the matching score, *not* from the model's opinion, so the check can't flatter the
   model). The team marks each as right or wrong.
2. **Candidates fixed in advance:** zero-shot with several descriptions per class; plus mirrored second look;
   linear probe; second expert model (FashionCLIP 2.0); and their averages.
3. **Nested cross-validation:** 5 folds split by Instagram page. Inside each fold the best candidate per
   attribute is chosen on the training pages only, then scored on the held-out pages.
4. **Report in plain percentages:** style right %, main colour right %, colour in top 2 %, the same for
   fabric, next to "always guess the most common answer". Then freeze the winning recipe and train it on all 195.

### 2. Forecaster: freeze, then replay 2024 and 2025 (about 3 h)
1. **Development data only up to Dec 2023 (Spain):** retrain the LSTM on the 44-city photo trends plus Spain's
   searches up to 2023; choose between Chronos-2, LSTM, baselines and averages on forecasts that end by Dec 2023.
2. **Freeze.**
3. **Replays for Spain:** forecasts made in Jan 2024, Jul 2024, Jan 2025, Jul 2025, Jan 2026 and Apr 2026,
   each compared with what then happened. One scorecard: error vs baseline, direction right %, truth inside
   range %, rising/fading calls right.
4. The Barcelona 2013–2018 street-photo backtest stays as the second, photo-based proof.

### 3. Forecaster: untouched countries (about 2 h)
1. Fetch the same 33 searches for **Italy** (Italian) and **Portugal** (Portuguese), Nov 2019 – Oct 2026.
   These match the team's Milan and Lisbon photos.
2. Run the frozen forecaster once on them: they were never looked at while building, so they are the cleanest
   accuracy test in the project.

### 4. Designer verified with real data (about 3 h)
1. **Brief from the forecast:** the rising searches (e.g. linen trousers) become the brief; fading ones are avoided.
2. **Style reference (synopsis Plan A+C):** the real team photo closest to the brief steers SDXL-Turbo
   (IP-Adapter, ~3 GB download; skipped if it does not fit the 6 GB GPU, with the result reported).
3. **Three checks on every concept**, all against real data:
   - *Attributes:* our tagger confirms the brief's fabric, colour and style.
   - *Realism:* its closest real outfit photo must be at least as close as real photos usually are to each
     other (threshold from the 195 real photos).
   - *Not a copy:* threshold calibrated on the real duplicate screenshots in the dataset.
4. **2025 generation replay:** a brief from the forecast made in Jan 2025 generates concepts; we measure
   whether they look more like the team's real 2025 photos (31) than their 2024 photos (31). Small sample,
   reported as indicative.

### 5. Report, app and documents (about 2 h)
- **Brand report:** Spain, Summer 2027: calls with ranges, the replay scorecard as "how far to trust this",
  checked concepts, method and limits.
- **App tabs:** Overview · Tagger · Forecast proof (Barcelona) · Replays 2024–25 + Italy/Portugal ·
  Forecast 2027 · Designer · Report.
- README, demo guide and a one-page results summary updated with the final numbers.

### 6. End-to-end check (about 1 h)
Every tab in the browser, a full report generated from scratch, all numbers cross-checked against the saved
result files.

## Known limits (stated in the app and report)
- 195 photos is small: results vary by a few points between folds; we report the spread.
- Search interest is a proxy for demand, not sales; street photos stop in 2018.
- Forecasts a whole year ahead are much less certain than 6 months ahead.
- SDXL-Turbo is non-commercial; FLUX.1-schnell (Apache 2.0) would replace it for brand use.
