# Findings — live accuracy on the phone (2026-11-14)

Status: hardware session A done. The phone streams from the armband correctly, but live accuracy is poor. This file records what's known, so the next session can start from it.

## 1. The armband works on the phone

- The armband firmware sends **WUA2 packets**: 216 bytes = 2 samples × 108 bytes, with 8 EMG int32 values at × 0.045 µV. MindRove's Android SDK can't decode them; the Pi's Python SDK can.
  - The decoder (`core/MindRoveWua2.kt`) follows `WUA2_PACKET::interpret` from the mindrove 5.2.3 wheel.
- The armband streams to **one device, chosen when it powers on.** Power-cycle it with only the phone joined.
- Live: 500 samples/s, 0 lost, and the rest level is on the same scale as the recordings.

## 2. The phone computes exactly what the Pi computes

The guided check (`yad_check_guy_20261114_173611.bin`, `yad_check_tomer_20261114_174022.bin`) was run through the Pi's own code with `tools/analyze_check.py`. The phone and the Pi code agree to max |Δp| ≈ 1e-5:

| Check | Accuracy (phone = Pi code) |
|---|---|
| Guy, model `guy` | 46%: REST 25, CLOSE 62, OPEN 88, RIGHT 38, LEFT 17 |
| Tomer, model `tomer` | 74%: REST 96, CLOSE 33, OPEN 54, RIGHT 88, LEFT 100 |

**The Pi would score the same on this data.** The problem is not the phone.

## 3. The armband isn't simply rotated

No rotation or mirroring of the 8 channels improves *all* gestures. Better overall scores only trade some gestures for others, which is chance across 16 variants with ~24 windows per gesture.

## 4. The training accuracy (~99%) never measured generalization

`fine_tune.py` validates on a random split of windows that overlap by 90%, so the validation set is almost a copy of the training set. A more honest version is to retrain the same way but **hold out one whole take** (each recording has 3 takes per gesture):

| Model | Held-out take, same session | Worst gesture |
|---|---|---|
| `guy` | 87–96% | LEFT 44% in one take |
| `tomer` | 64–90% | REST 0% in one take |
| `tomer_long` | 94–99% | — |

And **across sessions** (same person, different day; `tomer` ↔ `tomer_long`, both with RMS over time):

| Train → test | Accuracy | Per gesture (REST, CLOSE, OPEN, RIGHT, LEFT) |
|---|---|---|
| tomer → tomer_long | 47.5% | 86, 35, **0**, 98, 18 |
| tomer_long → tomer | 46.4% | 65, 11, **0**, 70, 86 |
| both sessions in training | 100% (sanity check) | |

**A model trained on one session falls to ~47% on another session.** That matches today's live results. Putting the armband on again is effectively a new session.

## 5. Adapting to today's data

Tested on the check data: adapt with the first 3 s of each gesture, test on the last 3 s.

| Approach | Guy | Tomer |
|---|---|---|
| Existing model, as is | 48% | 72% |
| Existing model, scaler refit on today's data | **22%** (worse: normalisation alone doesn't fix it) | — |
| **Retrain: old data + 3 s/gesture today** | **95%** (REST 100, CLOSE 94, OPEN 82, RIGHT 100, LEFT 98) | **99%** (100, 99, 97, 100, 97) |
| Retrain: only 3 s/gesture today | 86% | 86% |

**3 seconds per gesture from today, added to the old recordings, restores ~95–99%.** Caveat: the test windows are the second half of the same 6 s hold, so this is optimistic. A separate check after calibration is the real test. Old data + new data clearly beats new data alone, so the old recordings stay valuable.

## 6. Plan

1. **A short calibration each time the armband goes on (~1–2 min instead of 30).** Record a guided ~10 s per gesture (section 5: even 3 s helps a lot), then retrain the user's head with old data + today's data: on the PC first, on the phone later. Then run the guided check to confirm.
2. **Honest scoring.**
   - In `fine_tune.py`: validate on a held-out take, not a random split. This is a change to the training code, so it needs the user's OK.
   - On the phone: the guided check is the "good enough today?" test.
3. **More sessions per person.** Section 4 shows that training on both sessions handles both. Every calibration adds a session, so the model gets more robust over time.
4. **Consistent placement**, e.g. a mark on the skin or sleeve aligned with the armband.
5. Later: weigh the gestures that fail most across sessions (OPEN, LEFT) when choosing gestures or adding training data.
