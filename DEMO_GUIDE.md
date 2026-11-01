# Demo guide: showing the MVP to Dr. Sharma (about 12 minutes)

## Before the demo (5 minutes)

1. In the `fashion_predictor` folder, run:

   ```bash
   streamlit run app.py
   ```

2. On the **Pipeline** page choose **Team photo** (number 0) once. Designs start by themselves; the first run loads the
   image model, which takes about a minute. After that, each new photo takes about 30 seconds end to end.

No internet is needed. Every model and result is on the laptop.

## The story in one sentence

*"Give it an outfit photo: it reads what the person is wearing, forecasts whether each detail will rise or fade next
season, designs an updated look, and checks that design against real photos. Every step was tested on data it never
saw."*

## Pipeline page (7 min): one photo, every layer

1. **① Tagger.** Pick **Team photo 133**. *"A CNN finds the person; fashion vision transformers read the style,
   fabric and colour."* Point at the trend line inside each card.
2. **The link.** The yellow verdict: *"Brown is leaning down in Spain (brown dress −14%), so the design swaps it for
   green. Wool and the winter style are leaning up, so they stay."*
3. **② Forecast.** The table lists each matching search with its forecast, range and **2024–26 track record**. Switch
   **Market** to Italy: the same photo is re-linked to Italian searches.
4. **③ Brief → ④ Designer.** *"SDXL-Turbo draws the brief; our tagger then checks every concept against our 195 real
   photos: right attributes, looks real, not a copy."*
5. **⑤ Report.** Click **Download**: the page a brand would receive, starting from this photo.
6. **Live test:** upload a photo the teacher chooses. That is the honest test, because no model has seen it.

## Evidence page (4 min)

- **Headline numbers:** say that each was measured on unseen data.
- **Forecast replays 2024–26:** *"We froze the forecaster on data up to December 2023, then pretended it was each
  month from January 2024 onwards. Italy was never used to build anything."*
  - Spain had 16.5% less error than "same as last year", with calls right 24 of 29.
  - Italy had 9.6% less error, with calls right 18 of 20.
  - Live example: Spain, linen trousers, Jan 2025. The forecast was +25% and the real result +22%.
- **Tagger accuracy:** on 295 photos from 30 pages, style 58%, fabric 69%, colour 71% (top 2: 83%, 90%, 86%).
  Be fair about it: the 100 new photos are easier. On the same photos, the extra data raised fabric top 2 on both
  batches and Tailored/Suiting from 8 to 10 of 12; colour is unchanged.
- **Designer test:** 12 of 12 concepts were kept. In the 2025 replay, 9 of 12 forecast concepts were closer to the
  real 2025 photos, against 0 of 12 for the control.
- **Barcelona street-photo proof:** the method was proven on 96,965 dated street photos.

## Questions the teacher may ask

| Question | Short answer |
|---|---|
| How accurate is it, in percent? | **Tagger** (295 photos, nested cross-validation): main guess right style 58%, fabric 69%, colour 71%; top 2: 83%, 90%, 86% (guessing: 25–47%). **Forecaster:** 16.5% (Spain) and 9.6% (untouched Italy) less error than "same as last year"; rising/fading calls right 42 of 49 (86%); the truth falls inside the range about 80% of the time. **Designer:** 12 of 12 concepts pass every check. |
| Where is the deep learning? | Faster R-CNN (CNN), Marqo-FashionSigLIP and FashionCLIP (vision transformers), Chronos-2 (transformer), our LSTM (trained by us), SDXL-Turbo (diffusion). |
| What did *you* train? | The quantile LSTM (pinball loss, early stopping) on 44-city street-style trends plus Spain's searches up to 2023, and a linear probe on our 295 labelled photos. |
| How do you prevent data leakage? | Tagger: folds split by Instagram page, recipe chosen inside each fold. Forecaster: frozen on data to Dec 2023 before 2024–26 were scored; Italy never touched; every forecast uses only earlier months. Designer: thresholds from real photos only. |
| Why Google search data? | Our own dated photo collection doesn't exist yet. Search interest is free, monthly and current; the street-photo proof shows the method also works on photos. |
| Why not scrape Instagram, brand sites or Street View? | Their terms forbid it, and Street View photographs each street only every few years, with faces blurred. |
| Why not train everything from scratch? | 295 photos is far too few; transfer learning reuses models trained on millions of examples. |
| Is SDXL-Turbo allowed for brands? | Its licence is non-commercial, which is fine for a student prototype; FLUX.1-schnell (Apache 2.0) would replace it. |
| What's next? | Check the labels by eye, add our own dated photo collection, more countries, and the VISUELLE sales check. |
