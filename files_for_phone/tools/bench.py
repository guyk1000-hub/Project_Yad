"""
Bench: how fast and how reliably does a pipeline setting reach the right gesture?

Replays labelled phone recordings (guided checks, calibrations, recordings: any yad_check_*,
yad_calib_*, rec_* file) through the pipeline at a fine hop, and for each decision rule reports:
  - latency: from the start of a gesture to the first correct decision (median / 90th percentile);
  - reached: share of gestures that got a correct decision at all;
  - steady accuracy: decisions that are right, counted from 1 s into each gesture;
  - wrong moves per minute: the hand switching to a gesture that is neither the one being held
    nor the one just before it (REST moves nothing, so it never counts).

Validation is across recordings: each test file is scored with a model trained on all the user's
other files from the same day(s) (leave one out), never on itself. Compute time on the phone is
not included (the CNN takes ~95 ms per window on the S23 FE; a hop shorter than that needs a faster CNN).

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/bench.py tomer
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/bench.py tomer --features td
"""
import argparse
import glob
import os
import re

import _env  # noqa: F401  must precede numpy
import numpy as np

import analyze_check
import yad_ref

STEP = 25                 # samples between stream windows (50 ms); every rule's hop is a multiple
L = 100                   # window length (200 ms)
SR = 500
REST = 0
CHECKS_DIR = os.path.join(yad_ref.OUT_DIR, "checks")


def user_files(user: str, day: str = None) -> list:
    """The user's labelled phone files, oldest first. File names carry the time (yyyyMMdd_HHmmss)."""
    pats = [f"yad_check_{user}[@_-]*.bin", f"yad_check_{user}.bin", f"yad_calib_{user}_*.bin", f"rec_{user}_*.bin"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(CHECKS_DIR, p))}, key=file_time)
    return [f for f in files if day is None or file_time(f).startswith(day)]


def file_time(path: str) -> str:
    return re.findall(r"\d{8}_\d{6}", os.path.basename(path))[-1]


def td_features(X: np.ndarray) -> np.ndarray:
    """Classic time-domain EMG features per channel (Hudgins): MAV, RMS, WL, ZC, SSC -> [N, 40]."""
    W = X[..., 0].astype(np.float64)                       # [N, 8, L]
    d = np.diff(W, axis=2)
    thr = 0.01 * np.abs(W).mean(axis=(1, 2), keepdims=True)  # ignore tiny wiggles
    mav = np.abs(W).mean(2)
    rms = np.sqrt((W * W).mean(2))
    wl = np.abs(d).sum(2)
    zc = ((W[..., :-1] * W[..., 1:] < 0) & (np.abs(d) > thr)).sum(2)
    ssc = ((d[..., :-1] * d[..., 1:] < 0) & ((np.abs(d[..., :-1]) > thr) | (np.abs(d[..., 1:]) > thr))).sum(2)
    return np.hstack([mav, rms, wl, zc, ssc]).astype(np.float64)


class Stream:
    """One file: windows every STEP samples, with features, labels and the gesture phases."""

    def __init__(self, path: str, features: str):
        c = analyze_check.read_check(path)
        self.name = os.path.basename(path)
        self.rms_over = c["header"]["scalar_rms_over"]
        labels = c["labels"]
        filtered = analyze_check.pi_filter(c["raw"])
        self.ends = np.arange(L - 1, len(labels), STEP)
        X = np.stack([filtered[e - L + 1:e + 1].T for e in self.ends])[..., None].astype(np.float32)
        if features == "cnn":
            self.F = yad_ref.head_inputs(X, yad_ref.cnn_features(X), self.rms_over)
        else:
            self.F = td_features(X)
        self.label = labels[self.ends]                     # gesture at the window's last sample
        starts = self.ends - L + 1
        phase_start = np.zeros(len(labels), dtype=int)
        for i in range(1, len(labels)):
            phase_start[i] = i if labels[i] != labels[i - 1] else phase_start[i - 1]
        # Training windows: the phone's rule (inside one gesture, >= 1 s in, every 50 samples).
        self.train_mask = ((starts % 50 == 0) & (labels[starts] >= 0) & (labels[starts] == self.label)
                           & (starts >= phase_start[starts] + SR))
        # Phases: (label, first sample, last sample, label of the phase before).
        self.phases = []
        prev, s = -1, 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[s]:
                if labels[s] >= 0:
                    self.phases.append((int(labels[s]), s, i - 1, prev))
                prev = int(labels[s]) if labels[s] >= 0 else prev
                s = i
        self.minutes = sum(p[2] - p[1] + 1 for p in self.phases) / SR / 60


def fit(F: np.ndarray, y: np.ndarray):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F)
    m = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, activation="tanh",
                      solver="lbfgs", alpha=0.01).fit(sc.transform(F), y)
    return lambda A: m.predict_proba(sc.transform(A)), m.classes_


def cached_predictions(features: str, test: Stream, train: list, F: np.ndarray, y: np.ndarray):
    """Per-window probabilities for the test file from a model trained on [train]; cached, since only the rules change."""
    import hashlib
    key = hashlib.sha256((features + test.name + "|".join(s.name for s in train)).encode() + F.tobytes()).hexdigest()[:16]
    path = os.path.join(yad_ref.OUT_DIR, "cache", f"bench_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["P"], z["classes"]
    predict, classes = fit(F, y)
    P = predict(test.F)
    np.savez(path, P=P, classes=classes)
    return P, classes


# ---- decision rules ----------------------------------------------------------------------------

def rule_group(P, every, k, theta):
    """Today's rule: windows every `every` steps, non-overlapping groups of k averaged, decide if max >= theta."""
    idx = np.arange(0, len(P), every)
    out = []
    for g in range(k - 1, len(idx), k):
        avg = P[idx[g - k + 1:g + 1]].mean(0)
        if avg.max() >= theta:
            out.append((idx[g], int(avg.argmax())))
    return out


def rule_sliding(P, every, k, theta):
    """Average of the last k windows, re-evaluated at every window (every `every` steps)."""
    idx = np.arange(0, len(P), every)
    out = []
    for g in range(k - 1, len(idx)):
        avg = P[idx[g - k + 1:g + 1]].mean(0)
        if avg.max() >= theta:
            out.append((idx[g], int(avg.argmax())))
    return out


def rule_agree(P, every, k, theta):
    """The last k windows all have the same top class, each with p >= theta."""
    idx = np.arange(0, len(P), every)
    top, conf = P.argmax(1), P.max(1)
    out = []
    for g in range(k - 1, len(idx)):
        w = idx[g - k + 1:g + 1]
        if (top[w] == top[w[-1]]).all() and (conf[w] >= theta).all():
            out.append((idx[g], int(top[w[-1]])))
    return out


def score(stream: Stream, decisions, classes) -> dict:
    """Latencies per phase, steady accuracy, wrong moves (see the module docstring)."""
    t = np.array([stream.ends[i] for i, _ in decisions], dtype=int)
    d = np.array([classes[c] for _, c in decisions], dtype=int)
    lat, reached, steady_ok, steady_n, wrong = [], 0, 0, 0, 0
    # Wrong moves: the hand state changes to a non-REST gesture that isn't held now or just before.
    state = None
    for ti, di in zip(t, d):
        if di == state:
            continue
        state = di
        if di == REST:
            continue
        ph = next((p for p in stream.phases if p[1] <= ti <= p[2]), None)
        if ph is not None and di not in (ph[0], ph[3]):
            wrong += 1
    for label, s, e, _ in stream.phases:
        inside = (t >= s) & (t <= e)
        hit = inside & (d == label)
        if hit.any():
            reached += 1
            lat.append((t[hit][0] - s) / SR * 1000)
        late = inside & (t >= s + SR)
        steady_n += late.sum()
        steady_ok += (late & (d == label)).sum()
    return {"lat": lat, "reached": reached, "phases": len(stream.phases), "ok": steady_ok,
            "n": steady_n, "wrong": wrong, "minutes": stream.minutes}


RULES = [("today: group of 6, 200 ms, ≥0.7", rule_group, 4, 6, 0.7)]
for every, ms in ((4, 200), (2, 100), (1, 50)):
    for k in (2, 3, 4):
        for theta in (0.7, 0.9):
            if ms == 200 and k == 2 and theta == 0.7:
                continue
            RULES.append((f"sliding avg of {k}, every {ms} ms, ≥{theta}", rule_sliding, every, k, theta))
            RULES.append((f"{k} agree, every {ms} ms, each ≥{theta}", rule_agree, every, k, theta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user")
    ap.add_argument("--day", help="only files from this day (yyyyMMdd); default: all")
    ap.add_argument("--features", choices=("cnn", "td"), default="cnn",
                    help="cnn = today's NaviFlame dense_8 + 2 scalars; td = 40 classic time-domain features")
    ap.add_argument("--top", type=int, default=12, help="rules to print, best by latency among the safe ones")
    ap.add_argument("--all", action="store_true", help="print every rule, fastest first, not only the safe ones")
    args = ap.parse_args()

    files = user_files(args.user, args.day)
    print(f"{len(files)} files for {args.user}: " + ", ".join(os.path.basename(f) for f in files), flush=True)
    streams = [Stream(f, args.features) for f in files]
    totals = {r[0]: [] for r in RULES}
    for i, test in enumerate(streams):
        train = [s for j, s in enumerate(streams) if j != i and s.train_mask.any()]
        F = np.vstack([s.F[s.train_mask] for s in train])
        y = np.concatenate([s.label[s.train_mask] for s in train])
        P, classes = cached_predictions(args.features, test, train, F, y)
        for name, fn, every, k, theta in RULES:
            totals[name].append(score(test, fn(P, every, k, theta), classes))
        base = totals[RULES[0][0]][-1]
        print(f"  test {test.name}: {len(y)} training windows from {len(train)} files; today's rule: "
              f"median latency {np.median(base['lat']) if base['lat'] else float('nan'):.0f} ms, "
              f"steady {100 * base['ok'] / max(base['n'], 1):.0f}%", flush=True)

    rows = []
    for name, *_ in RULES:
        r = totals[name]
        lat = np.concatenate([x["lat"] for x in r]) if any(x["lat"] for x in r) else np.array([np.nan])
        rows.append((name, np.median(lat), np.percentile(lat, 90),
                     100 * sum(x["reached"] for x in r) / sum(x["phases"] for x in r),
                     100 * sum(x["ok"] for x in r) / max(sum(x["n"] for x in r), 1),
                     sum(x["wrong"] for x in r) / sum(x["minutes"] for x in r)))
    today = rows[0]
    print(f"\nfeatures: {args.features}; {sum(s.minutes for s in streams):.1f} min of gestures in "
          f"{len(streams)} files (leave one file out)\n")
    header = f"{'rule':38s} {'median ms':>9s} {'p90 ms':>7s} {'reached':>8s} {'steady':>7s} {'wrong/min':>9s}"
    print(header)
    fmt = "{:38s} {:9.0f} {:7.0f} {:7.0f}% {:6.0f}% {:9.2f}"
    print(fmt.format(*today))
    safe = [r for r in rows[1:] if r[5] <= today[5] + 0.05 and r[4] >= today[4] - 2]
    for r in sorted(rows[1:] if args.all else safe, key=lambda r: r[1])[:None if args.all else args.top]:
        print(fmt.format(*r))
    print(f"\n(only rules with no more wrong moves than today (+0.05/min) and steady accuracy within 2 points; "
          f"{len(safe)} of {len(rows) - 1})")


if __name__ == "__main__":
    main()
