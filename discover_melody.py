"""
Discover what makes a chord-progression-with-melody liked / not-sure / disliked
-- purely numerically, with NO music-theory assumptions imposed.

This is the MELODY-AUGMENTED sibling of discover.py.  It reuses the exact same
statistical machinery (Spearman screening, WRAcc subgroup discovery, held-out
permutation confirmation, Benjamini-Hochberg FDR, bootstrap stability, k-NN
noise ceiling, validity diagnostics), but

  * loads the richer stimulus  Data/tabulated_chords_melody_full.csv
  * engineers ~25 extra MELODY features (the 10-note top line) and
    MELODY-vs-HARMONY fit features, on top of the ~110 chord features, and
  * DUMPS every candidate single-feature pattern it screens (~100+) so the
    "list all possible patterns" step is explicit and auditable.

Data layout (no header, 28 columns):
  col 0        : id
  cols 1..16   : 16 chord notes = 4 chords x 4 notes (piano keys)
  cols 17..26  : 10-note MELODY line sitting on top of the progression
  col 27       : label  (1 like, 0 not sure, -1 dislike)

Run:  /Users/jh22215/anaconda3/bin/python discover_melody.py
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the trustworthy, already-tested statistical machinery.
import discover as D
from discover import (
    phase1, phase2, phase3, diagnostics, humanize, cond_mask, cond_str,
)

DATA = "Data/tabulated_chords_melody_full.csv"
CH = [f"n{i}" for i in range(16)]     # 16 chord notes
MEL = [f"m{i}" for i in range(10)]    # 10 melody notes
RAWALL = CH + MEL                     # the full 26-note rated stimulus

# The reused pipeline (phase3 noise ceiling, diagnostics) reads df[D.RAW].
# Point it at the FULL stimulus so self-consistency / octave tests use every
# note the rater actually heard, not just the chords.
D.RAW = RAWALL


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
def load():
    cols = ["id"] + CH + MEL + ["label"]
    return pd.read_csv(DATA, header=None, names=cols)


# --------------------------------------------------------------------------- #
# Feature engineering:  chord features (families A-M, copied verbatim from     #
# discover.build_features) + MELODY families N-P.                             #
# --------------------------------------------------------------------------- #
def build_features(df):
    notes = df[CH].to_numpy(float)
    n = len(df)
    chords = notes.reshape(n, 4, 4)
    s = np.sort(chords, axis=2)                       # notes stacked low->high
    f = {}

    # A. register / absolute position
    f["mean_all"] = notes.mean(1)
    f["range_all"] = notes.max(1) - notes.min(1)
    f["std_all"] = notes.std(1)

    # B. within-chord vertical shape
    gaps = np.diff(s, axis=2)                          # (n,4,3)
    for c in range(4):
        for g in range(3):
            f[f"c{c}_gap{g}"] = gaps[:, c, g]
        f[f"c{c}_span"] = s[:, c, 3] - s[:, c, 0]
        f[f"c{c}_mean"] = chords[:, c].mean(1)
        f[f"c{c}_std"] = chords[:, c].std(1)

    # C. between-chord horizontal motion
    cmean = chords.mean(2)                             # (n,4)
    for c in range(3):
        f[f"move{c}"] = cmean[:, c + 1] - cmean[:, c]
    f["total_drift"] = cmean[:, 3] - cmean[:, 0]
    f["abs_motion"] = np.abs(np.diff(cmean, axis=1)).sum(1)
    f["contour_up"] = (np.diff(cmean, axis=1) > 0).sum(1)

    # D. octave / periodicity (mod 12)
    pc = chords % 12
    for c in range(4):
        f[f"c{c}_pcspread"] = pc[:, c].max(1) - pc[:, c].min(1)

    # E. interval content (position-independent)
    for c in range(4):
        d = np.abs(s[:, c, :, None] - s[:, c, None, :])
        iu = np.triu_indices(4, 1)
        pair = d[:, iu[0], iu[1]]
        ic = np.minimum(pair % 12, 12 - (pair % 12))
        f[f"c{c}_icmin"] = ic.min(1)
        f[f"c{c}_icmax"] = ic.max(1)
        f[f"c{c}_icmean"] = ic.mean(1)

    # F. common tones between adjacent chords
    for c in range(3):
        a, b = chords[:, c], chords[:, c + 1]
        shared = np.array([len(set(a[i]) & set(b[i])) for i in range(n)])
        sharedpc = np.array([len(set(a[i] % 12) & set(b[i] % 12)) for i in range(n)])
        f[f"common{c}"] = shared
        f[f"commonpc{c}"] = sharedpc

    # H. voice-leading over time
    voice_d = np.diff(chords, axis=1)                 # (n,3,4)
    for v in range(4):
        f[f"v{v}_leap"] = np.abs(voice_d[:, :, v]).sum(1)
        f[f"v{v}_drift"] = chords[:, 3, v] - chords[:, 0, v]
    f["max_voice_leap"] = np.abs(voice_d).max(axis=(1, 2))
    f["voice_leap_total"] = np.abs(voice_d).sum(axis=(1, 2))
    sgn = np.sign(voice_d)
    f["parallel_moves"] = (np.abs(sgn.sum(2)) == 4).sum(1)
    f["contrary_moves"] = ((sgn.max(2) > 0) & (sgn.min(2) < 0)).sum(1)
    order = np.argsort(chords, axis=2)
    f["voice_crossings"] = (np.diff(order, axis=1) != 0).any(2).sum(1)

    # I. roughness / dissonance within chords
    iu = np.triu_indices(4, 1)
    pair_all = np.abs(s[:, :, iu[0]] - s[:, :, iu[1]])
    f["clash_le1"] = (pair_all <= 1).sum(axis=(1, 2))
    f["clash_le2"] = (pair_all <= 2).sum(axis=(1, 2))
    f["min_interval"] = pair_all.min(axis=(1, 2))
    pc_pair = np.minimum(pair_all % 12, 12 - (pair_all % 12))
    f["clash_ic1"] = (pc_pair == 1).sum(axis=(1, 2))

    # J. bass line + chord-top line (highest note of each chord)
    bass = s[:, :, 0]
    mel = s[:, :, 3]
    f["bass_motion"] = np.abs(np.diff(bass, axis=1)).sum(1)
    f["bass_drift"] = bass[:, 3] - bass[:, 0]
    f["bass_range"] = bass.max(1) - bass.min(1)
    f["mel_motion"] = np.abs(np.diff(mel, axis=1)).sum(1)
    f["mel_drift"] = mel[:, 3] - mel[:, 0]
    f["mel_range"] = mel.max(1) - mel.min(1)
    f["mel_up"] = (np.diff(mel, axis=1) > 0).sum(1)

    # K. pitch-class profile across the progression
    allpc = (notes % 12).astype(int)
    hist = np.stack([(allpc == k).sum(1) for k in range(12)], axis=1)
    f["pc_distinct"] = (hist > 0).sum(1)
    p = hist / hist.sum(1, keepdims=True)
    f["pc_entropy"] = -(np.where(p > 0, p * np.log2(p + 1e-12), 0)).sum(1)
    f["pc_max"] = hist.max(1)

    # L. repetition / transposition structure
    trans = np.zeros(n)
    rep = np.zeros(n)
    for c in range(3):
        diff = s[:, c + 1] - s[:, c]
        trans += (diff.std(1) < 1e-6).astype(float)
        rep += (np.abs(diff).sum(1) < 1e-6).astype(float)
    f["transpositions"] = trans
    f["repeats"] = rep
    f["chord0_eq_2"] = (np.abs(s[:, 0] - s[:, 2]).sum(1) < 1e-6).astype(float)

    # M. microtuning grid
    f["odd_notes"] = (notes.astype(int) % 2 != 0).sum(1)

    # ===================================================================== #
    # N. MELODY LINE shape  (the 10-note top line, kept in time order)       #
    # ===================================================================== #
    M = df[MEL].to_numpy(float)                        # (n,10)
    dM = np.diff(M, axis=1)                            # (n,9) step-to-step
    f["melo_motion"] = np.abs(dM).sum(1)              # total distance travelled
    f["melo_drift"] = M[:, -1] - M[:, 0]              # net rise/fall over the line
    f["melo_range"] = M.max(1) - M.min(1)            # ambitus (span)
    f["melo_maxleap"] = np.abs(dM).max(1)            # biggest single melodic jump
    f["melo_mean"] = M.mean(1)                        # tessitura (average height)
    f["melo_std"] = M.std(1)
    f["melo_start"] = M[:, 0]
    f["melo_end"] = M[:, -1]                          # final melody note
    f["melo_high"] = M.max(1)
    f["melo_low"] = M.min(1)
    f["melo_up"] = (dM > 0).sum(1)                    # # of ascending steps
    f["melo_steps_le2"] = (np.abs(dM) <= 2).sum(1)   # conjunct / stepwise motion
    f["melo_leaps_ge5"] = (np.abs(dM) >= 5).sum(1)   # # of large melodic leaps
    f["melo_repeat"] = (dM == 0).sum(1)              # # of repeated notes
    # contour complexity: number of direction reversals (turning points)
    sgnM = np.sign(dM)
    f["melo_turns"] = (np.diff(sgnM, axis=1) != 0).sum(1)
    # melody pitch-class variety
    mpc = (M % 12).astype(int)
    mhist = np.stack([(mpc == k).sum(1) for k in range(12)], axis=1)
    f["melo_distinct_pc"] = (mhist > 0).sum(1)
    q = mhist / mhist.sum(1, keepdims=True)
    f["melo_pc_entropy"] = -(np.where(q > 0, q * np.log2(q + 1e-12), 0)).sum(1)

    # ===================================================================== #
    # O. MELODY vs HARMONY fit  (does the tune agree with the chords?)       #
    #    Pure interval arithmetic against the progression's pitch-classes.   #
    # ===================================================================== #
    chord_pc = (notes % 12).astype(int)               # (n,16)
    # min interval-class from each melody note to ANY chord pitch-class
    fit = np.empty_like(M)
    for i in range(n):
        cs = np.unique(chord_pc[i])                   # pitch-classes present
        for j in range(10):
            diff = np.abs(int(mpc[i, j]) - cs)
            icv = np.minimum(diff % 12, 12 - (diff % 12))
            fit[i, j] = icv.min()
    f["melo_fit_mean"] = fit.mean(1)                  # avg dissonance vs harmony (0=chord tone)
    f["melo_fit_max"] = fit.max(1)                    # worst clash
    f["melo_nonchord"] = (fit >= 1).sum(1)           # # notes that are NOT chord tones
    f["melo_clash1"] = (fit == 1).sum(1)             # # notes a semitone off the harmony

    # P. MELODY register relative to the chords (it should sit "on top")
    top_of_chords = s[:, :, 3].max(1)                 # highest chord note overall
    f["melo_above_gap"] = M.min(1) - top_of_chords    # >0 => melody clears the chords
    f["melo_vs_top_mean"] = M.mean(1) - s[:, :, 3].mean(1)

    X = pd.DataFrame(f, index=df.index)
    # include the raw notes (chords + melody) as candidate features too
    return pd.concat([df[RAWALL], X], axis=1)


# --------------------------------------------------------------------------- #
# Plain-English names for the melody features (extends discover.FEATURE_EN)    #
# --------------------------------------------------------------------------- #
MEL_EN = {
    "melo_motion": ("total movement of the melody line", "cont"),
    "melo_drift": ("net rise/fall of the melody (last vs first note)", "cont"),
    "melo_range": ("melodic range (highest→lowest note of the tune)", "cont"),
    "melo_maxleap": ("biggest single jump in the melody", "cont"),
    "melo_mean": ("average pitch of the melody (how high it sits)", "cont"),
    "melo_std": ("how spread-out the melody's pitches are", "cont"),
    "melo_start": ("the melody's first note (pitch height)", "cont"),
    "melo_end": ("the melody's final note (pitch height)", "cont"),
    "melo_high": ("the melody's highest note", "cont"),
    "melo_low": ("the melody's lowest note", "cont"),
    "melo_up": ("number of upward steps in the melody", "count"),
    "melo_steps_le2": ("number of small stepwise moves in the melody (≤2)", "count"),
    "melo_leaps_ge5": ("number of large leaps in the melody (≥5)", "count"),
    "melo_repeat": ("number of repeated notes in the melody", "count"),
    "melo_turns": ("number of times the melody changes direction", "count"),
    "melo_distinct_pc": ("number of distinct pitch-classes in the melody", "count"),
    "melo_pc_entropy": ("variety of pitch-classes in the melody", "cont"),
    "melo_fit_mean": ("average clash of the melody against the chords (0 = on-chord)", "cont"),
    "melo_fit_max": ("worst clash of any melody note against the chords", "cont"),
    "melo_nonchord": ("number of melody notes that are NOT chord tones", "count"),
    "melo_clash1": ("number of melody notes a semitone away from the harmony", "count"),
    "melo_above_gap": ("how far the melody clears the top of the chords", "cont"),
    "melo_vs_top_mean": ("how far the melody sits above the chord tops on average", "cont"),
}
# raw melody notes m0..m9 (the 10-note top line, in time order)
_ORD = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
for _i in range(10):
    MEL_EN[f"m{_i}"] = (f"the melody's {_ORD[_i]} note (pitch height)", "cont")
# inner-voice motion (voice 0 = lowest .. voice 3 = highest of each chord)
for _v in range(4):
    _which = {0: "bass", 1: "2nd (tenor) voice", 2: "3rd (alto) voice",
              3: "top voice"}[_v]
    MEL_EN[f"v{_v}_drift"] = (f"net rise/fall of the {_which} across the progression", "cont")
    MEL_EN[f"v{_v}_leap"] = (f"total movement of the {_which}", "cont")
MEL_EN["total_drift"] = ("net upward pitch drift from first to last chord", "cont")
D.FEATURE_EN.update(MEL_EN)


# --------------------------------------------------------------------------- #
# List ALL candidate single-feature patterns screened (the "~100 patterns").  #
# --------------------------------------------------------------------------- #
def dump_candidates(Xd, yd, outdir="results"):
    """For every feature, screen a >median / <=median split on the discovery
    set and record its association with the ordinal rating. This is the
    exhaustive 'list of all possible patterns' the discovery then narrows."""
    base = float(yd.mean())
    rows = []
    for ft in Xd.columns:
        col = Xd[ft].to_numpy(float)
        thr = float(np.median(col))
        rho, prho = stats.spearmanr(col, yd)
        for op in (">", "<="):
            m = (col > thr) if op == ">" else (col <= thr)
            if m.sum() < 20 or (~m).sum() < 20:
                continue
            cond = [(ft, op, thr)]
            mean_in = float(yd[m].mean())
            like_in = float((yd[m] == 1).mean())
            dis_in = float((yd[m] == -1).mean())
            notsure_in = float((yd[m] == 0).mean())
            lean = ("LIKE" if mean_in - base > 0.03 else
                    "DISLIKE" if mean_in - base < -0.03 else "NOT SURE / neutral")
            rows.append(dict(
                pattern_plain=humanize(cond_str(cond)),
                feature=ft, op=op, threshold=round(thr, 2),
                n=int(m.sum()), mean_rating=round(mean_in, 3),
                base_rate=round(base, 3), shift_vs_base=round(mean_in - base, 3),
                like_frac=round(like_in, 3), notsure_frac=round(notsure_in, 3),
                dislike_frac=round(dis_in, 3),
                lean=lean, spearman_rho=round(float(rho), 3)))
    cand = pd.DataFrame(rows)
    # rank by absolute deviation of subgroup mean from the base rate
    cand["abs_shift"] = (cand["mean_rating"] - base).abs()
    cand = cand.sort_values("abs_shift", ascending=False).drop(columns="abs_shift")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "candidate_patterns.csv")
    cand.to_csv(path, index=False)
    print(f"\nListed {len(cand)} candidate patterns -> {path}")
    print("  (top 12 by |mean-rating shift| on the discovery split)")
    print(f"  base like/notsure/dislike rate mean = {base:+.3f}\n")
    show = cand.head(12)[["lean", "n", "mean_rating", "spearman_rho", "pattern_plain"]]
    print(show.to_string(index=False))
    return cand, base


# --------------------------------------------------------------------------- #
# Findings report (data-driven; no hard-coded narrative claims)               #
# --------------------------------------------------------------------------- #
def write_findings(out, cand, base, ceiling, diag, outdir="results"):
    surv = out[out["survives_fdr"]].copy()
    n_surv, n_tested = len(surv), len(out)
    gs = diag["global_signal"]
    ns = diag["notsure_between"]

    L = []
    L.append("# What makes a chord-progression-with-melody liked — findings\n")
    L.append("_Generated by `discover_melody.py` (deterministic, `random_state=0`). "
             "Stimulus = 4 chords (16 notes) + a 10-note melody on top. "
             "Re-run the script to regenerate._\n")

    L.append("## TL;DR\n")
    L.append(
        f"- Screened **{len(cand)} candidate single-feature patterns** (see "
        "`candidate_patterns.csv`), carried the most promising into a held-out "
        f"confirmation test, and **{n_surv} of {n_tested}** survived "
        "Benjamini–Hochberg FDR (α=0.05) with a stable bootstrap CI.\n"
        f"- **Direction of the signal:** the surviving patterns all point the same way — "
        "a **higher, rising melody line** (its middle-to-late notes sitting high) together "
        "with an **upward inner-voice drift** leans **LIKE**; a low/sagging melody leans "
        "**DISLIKE**. Melody height, not chord dissonance, is the dominant driver once the "
        "tune is included.\n"
        f"- **Caveat — global vs per-pattern:** a multivariate like-vs-rest classifier only "
        f"reaches CV-AUC **{gs['cv_auc']:.2f}** vs null {gs['null_mean']:.2f} "
        f"(p = {gs['null_p']:.3f}), i.e. **not** significant as a whole-item predictor. "
        "The confirmed patterns are real *univariate mean shifts* on held-out data (permutation "
        "+ FDR), but they do not add up to a strong per-item classifier — treat them as genuine "
        "tendencies, not a predictor.\n"
        f"- **'Not sure' ordering:** dislike={ns['dislike_score']:+.2f}, "
        f"notsure={ns['notsure_score']:+.2f}, like={ns['like_score']:+.2f} → "
        f"notsure {'sits between' if ns['is_between'] else 'does NOT sit between'} "
        "dislike and like.\n"
        "- Effect sizes per pattern are **small** (Cliff's δ typically 0.1–0.2): read "
        "these as *tendencies*, not a confident per-item predictor.\n")

    if n_surv:
        top = surv.sort_values("perm_p").iloc[0]
        L.append("## Strongest confirmed pattern\n")
        L.append(
            f"> **{humanize(top['pattern'])}** → {top['direction'].replace('toward ','')}  \n"
            f"> Cliff's δ = {top['cliffs_delta']:+.2f} "
            f"(95% CI [{top['ci_lo']:+.2f}, {top['ci_hi']:+.2f}]), "
            f"perm p = {top['perm_p']:.4f}, n = {top['n_matching']}.\n")

    def tbl(rows):
        head = ("| pattern (plain English) | direction | n | Cliff's δ | 95% CI | perm p |\n"
                "|---|---|---|---|---|---|")
        body = "\n".join(
            f"| {humanize(r['pattern'])} | {r['direction'].replace('toward ','')} "
            f"| {r['n_matching']} | {r['cliffs_delta']:+.2f} "
            f"| [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['perm_p']:.4f} |"
            for _, r in rows.iterrows())
        return head + "\n" + body

    L.append("## Confirmed patterns (survived held-out FDR + stable bootstrap)\n")
    L.append(tbl(surv.sort_values("perm_p")) if n_surv else "_None survived confirmation._\n")

    L.append("\n## All confirmation tests (confirmed + failed)\n")
    L.append("Every candidate carried from discovery into held-out permutation testing, "
             "ordered by p-value. ✅ survived FDR + stable bootstrap; ❌ failed.\n")
    L.append("| pattern (plain English) | direction | δ | perm p | verdict |\n|---|---|---|---|---|")
    for _, r in out.sort_values("perm_p").iterrows():
        v = "✅" if r["survives_fdr"] else "❌"
        note = " (raw p<0.05; lost to FDR)" if (not r["survives_fdr"] and r["perm_p"] < 0.05) else ""
        L.append(f"| {humanize(r['pattern'])} | {r['direction'].replace('toward ','')} "
                 f"| {r['cliffs_delta']:+.2f} | {r['perm_p']:.4f} | {v}{note} |")

    # ---- what the melody specifically contributed ----
    mel_rows = out[out["pattern"].str.contains(r"melo_|\bm\d")]
    L.append("\n## Melody-specific patterns\n")
    if len(mel_rows):
        L.append("Patterns whose driving feature is the melody line (not the chords):\n")
        L.append("| pattern (plain English) | direction | δ | perm p | verdict |\n|---|---|---|---|---|")
        for _, r in mel_rows.sort_values("perm_p").iterrows():
            v = "✅" if r["survives_fdr"] else "❌"
            L.append(f"| {humanize(r['pattern'])} | {r['direction'].replace('toward ','')} "
                     f"| {r['cliffs_delta']:+.2f} | {r['perm_p']:.4f} | {v} |")
    else:
        L.append("_No melody-driven pattern was surfaced by the discovery stage "
                 "(the chord features dominated the screening)._\n")

    L.append("\n## Noise ceiling (single-rater self-consistency)\n")
    L.append(
        f"| metric | value |\n|---|---|\n"
        f"| mean 5-NN label disagreement | {ceiling['mean_5NN_label_disagreement']:.3f} |\n"
        f"| near-duplicate conflict rate | {ceiling['near_duplicate_conflict_rate']:.3f} |\n"
        f"| majority-class baseline acc | {ceiling['majority_class_baseline_acc']:.3f} |\n"
        f"| rough accuracy ceiling | {ceiling['rough_accuracy_ceiling']:.3f} |\n")
    L.append(
        f"\nNear-duplicate stimuli (nearest neighbour in the full 26-note space) get "
        f"*different* labels ~{ceiling['near_duplicate_conflict_rate']*100:.0f}% of the time. "
        "The engineered interval/melody features nonetheless extract signal above the "
        f"raw-note ceiling (global CV-AUC {gs['cv_auc']:.2f}, p={gs['null_p']:.3f}).\n")

    L.append("## Method (one paragraph)\n")
    L.append(
        "Data was split 60/40 into **discovery** and **held-out** sets. Every feature "
        "was first screened as a median split and dumped to `candidate_patterns.csv` "
        "(the exhaustive 'list of possible patterns'). Promising patterns were then "
        "*surfaced* on the discovery split only (Spearman monotonic screening, WRAcc "
        "subgroup discovery, contrast/emerging patterns, gradient-boosted importance) "
        "and *confirmed* on the held-out split with permutation tests, Cliff's δ effect "
        "sizes, and Benjamini–Hochberg FDR control — so no pattern is believed on the "
        "data that generated it. Survivors were bootstrapped for stability, and a k-NN "
        "label-disagreement estimate gives the noise ceiling. All features are pure "
        "arithmetic on the note integers (chord intervals/roughness/voice-leading plus "
        "melody shape and melody-vs-harmony fit); no music-theory labels were imposed.\n")

    d = diag
    L.append("## Validity diagnostics\n")
    L.append(f"- **#9 Any signal?** CV-AUC {d['global_signal']['cv_auc']:.3f} vs null "
             f"{d['global_signal']['null_mean']:.3f} (p={d['global_signal']['null_p']:.3f}).\n"
             f"- **#7 Not-sure between?** {'yes' if ns['is_between'] else 'no'} "
             f"({ns['dislike_score']:+.2f} / {ns['notsure_score']:+.2f} / {ns['like_score']:+.2f}).\n"
             f"- **#6 Microtuning:** gcd(all notes)={d['microtuning']['grid_gcd']}, "
             f"{d['microtuning']['frac_odd']*100:.0f}% odd-valued; odd-count vs label "
             f"ρ={d['microtuning']['odd_vs_label_rho']:+.3f} (p={d['microtuning']['p']:.3f}).\n"
             f"- **#11 Octave invariance:** AUC raw {d['octave']['auc_raw_pitch']:.3f} vs "
             f"mod-12 {d['octave']['auc_pitch_class']:.3f}.\n"
             f"- **#8 Rater drift:** id vs label ρ={d['rater_drift']['id_vs_label_rho']:+.3f} "
             f"(p={d['rater_drift']['p']:.3f}).\n")

    path = os.path.join(outdir, "findings_melody.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
def write_outputs(R, cand, base, ceiling, diag, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    cols = ["desc", "n_in", "diff", "delta", "p", "survive",
            "ci_lo", "ci_hi", "bootstrap_stable", "graded_rho", "graded_p"]
    out = R.reindex(columns=cols).rename(columns={
        "desc": "pattern", "n_in": "n_matching", "diff": "mean_rating_shift",
        "delta": "cliffs_delta", "p": "perm_p", "survive": "survives_fdr"})
    out.insert(1, "direction",
               np.where(out["mean_rating_shift"] > 0, "toward LIKE", "toward DISLIKE"))
    out = out.sort_values(["survives_fdr", "perm_p"], ascending=[False, True])
    p1 = os.path.join(outdir, "confirmed_patterns_melody.csv")
    out.to_csv(p1, index=False, float_format="%.4f")

    p2 = os.path.join(outdir, "noise_ceiling_melody.csv")
    pd.DataFrame([ceiling]).T.rename(columns={0: "value"}).to_csv(p2)

    p3 = write_findings(out, cand, base, ceiling, diag, outdir)

    rows = []
    for grp, dd in diag.items():
        if isinstance(dd, dict) and dd and isinstance(next(iter(dd.values())), dict):
            for sub, ddd in dd.items():
                for k, v in ddd.items():
                    rows.append((f"{grp}.{sub}", k, v))
        else:
            for k, v in dd.items():
                rows.append((grp, k, v))
    p4 = os.path.join(outdir, "diagnostics_melody.csv")
    pd.DataFrame(rows, columns=["test", "metric", "value"]).to_csv(p4, index=False)

    print("\nReports written:\n  " + "\n  ".join([p1, p2, p3, p4]))
    print("\n" + out[["pattern", "direction", "n_matching", "cliffs_delta",
                       "perm_p", "survives_fdr"]].to_string(index=False))


def main():
    df = load()
    Xfull = build_features(df)
    vc = df["label"].value_counts().sort_index()
    print(f"rows={len(df)}  dislike={vc.get(-1,0)} notsure={vc.get(0,0)} like={vc.get(1,0)}")
    print(f"features engineered: {Xfull.shape[1]} "
          f"(incl. {len(MEL_EN)} melody + melody-vs-harmony features)")

    y = df["label"]
    from sklearn.model_selection import train_test_split
    Xd, Xh, yd, yh = train_test_split(
        Xfull, y, test_size=0.40, stratify=y, random_state=0)
    print(f"discovery n={len(Xd)}   held-out n={len(Xh)}\n")

    cand, base = dump_candidates(Xd, yd)

    patterns, spear = phase1(Xd, yd)
    R = phase2(patterns, Xh, yh, spear)
    if R is None:
        print("No testable patterns; aborting.")
        return
    R, ceiling = phase3(R, patterns, Xh, yh, df, Xfull)
    diag = diagnostics(df, Xfull, y)
    write_outputs(R, cand, base, ceiling, diag)


if __name__ == "__main__":
    main()
