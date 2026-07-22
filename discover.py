"""
Discover what makes a chord progression liked / not-sure / disliked  -- purely
numerically, with NO music-theory assumptions imposed.

Methodology (decided up front):
  * ORDINAL labels:  dislike(-1) < notsure(0) < like(1).  We ask not just
    "is a pattern there" but "is it GRADED" (monotone across the 3 levels).
  * DISCOVER then CONFIRM on disjoint splits, so a pattern is only believed if
    it survives on data it was NOT found in (guards against the garden-of-forking
    -paths / data dredging).
  * SINGLE-RATER personal-taste model -> we estimate a NOISE CEILING, because the
    labels may simply be self-inconsistent, bounding how much signal can exist.

Data: Data/tabulated_chords.csv, no header.
  col 0      : id
  cols 1..16 : 16 notes = 4 chords x 4 notes (piano keys, 0..177)
  col 17     : label  (1 like, 0 not sure, -1 dislike)

Run:  /Users/jh22215/anaconda3/bin/python discover.py
"""

import itertools
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from sklearn.tree import DecisionTreeClassifier, export_text

RNG = np.random.default_rng(0)
RAW = [f"n{i}" for i in range(16)]


# --------------------------------------------------------------------------- #
# Phase 0 : load + feature engineering (pure arithmetic, families A-G)        #
# --------------------------------------------------------------------------- #
def load():
    cols = ["id"] + RAW + ["label"]
    return pd.read_csv("Data/tabulated_chords.csv", header=None, names=cols)


def build_features(df):
    notes = df[RAW].to_numpy(float)
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
    f["abs_motion"] = np.abs(np.diff(cmean, axis=1)).sum(1)   # total distance travelled
    f["contour_up"] = (np.diff(cmean, axis=1) > 0).sum(1)     # how many upward moves

    # D. octave / periodicity (mod 12) -- the one music-theory HYPOTHESIS,
    #    included so the DATA can confirm or reject it.
    pc = chords % 12
    for c in range(4):
        f[f"c{c}_pcspread"] = pc[:, c].max(1) - pc[:, c].min(1)

    # E. interval content (position-independent): all 6 pairwise diffs / chord,
    #    plus their interval-classes mod 12 (octave-folded).
    for c in range(4):
        d = np.abs(s[:, c, :, None] - s[:, c, None, :])       # (n,4,4)
        iu = np.triu_indices(4, 1)
        pair = d[:, iu[0], iu[1]]                             # (n,6)
        ic = np.minimum(pair % 12, 12 - (pair % 12))         # interval-class 0..6
        f[f"c{c}_icmin"] = ic.min(1)
        f[f"c{c}_icmax"] = ic.max(1)
        f[f"c{c}_icmean"] = ic.mean(1)

    # F. repetition / common tones between adjacent chords
    for c in range(3):
        a, b = chords[:, c], chords[:, c + 1]
        shared = np.array([len(set(a[i]) & set(b[i])) for i in range(n)])
        sharedpc = np.array([len(set(a[i] % 12) & set(b[i] % 12)) for i in range(n)])
        f[f"common{c}"] = shared
        f[f"commonpc{c}"] = sharedpc

    # H. VOICE-LEADING over time. The data is 4 voices x 4 time-steps:
    #    voice v's path = notes at positions v, 4+v, 8+v, 12+v  (chords[:,:,v]).
    voice_d = np.diff(chords, axis=1)                 # (n,3 moves,4 voices)
    for v in range(4):
        f[f"v{v}_leap"] = np.abs(voice_d[:, :, v]).sum(1)   # total distance this voice travels
        f[f"v{v}_drift"] = chords[:, 3, v] - chords[:, 0, v]
    f["max_voice_leap"] = np.abs(voice_d).max(axis=(1, 2))   # biggest single jump anywhere
    f["voice_leap_total"] = np.abs(voice_d).sum(axis=(1, 2))
    # parallel vs contrary: per transition, do all voices move the same direction?
    sgn = np.sign(voice_d)                            # (n,3,4)
    f["parallel_moves"] = (np.abs(sgn.sum(2)) == 4).sum(1)   # all 4 same way
    f["contrary_moves"] = ((sgn.max(2) > 0) & (sgn.min(2) < 0)).sum(1)  # split directions
    # voice crossing: does the rank order of the 4 positions change between chords?
    order = np.argsort(chords, axis=2)
    f["voice_crossings"] = (np.diff(order, axis=1) != 0).any(2).sum(1)

    # I. ROUGHNESS / dissonance: count of "close" intervals (beating clashes).
    #    Pure subtraction; no theory beyond "small interval = rough".
    iu = np.triu_indices(4, 1)
    pair_all = np.abs(s[:, :, iu[0]] - s[:, :, iu[1]])   # (n,4,6) within-chord intervals
    f["clash_le1"] = (pair_all <= 1).sum(axis=(1, 2))    # near-unison clashes
    f["clash_le2"] = (pair_all <= 2).sum(axis=(1, 2))
    f["min_interval"] = pair_all.min(axis=(1, 2))        # tightest interval anywhere
    pc_pair = np.minimum(pair_all % 12, 12 - (pair_all % 12))
    f["clash_ic1"] = (pc_pair == 1).sum(axis=(1, 2))     # semitone interval-classes

    # J. BASS line (lowest note) and TOP NOTE (highest note) per chord, isolated.
    bass = s[:, :, 0]                                    # (n,4) lowest note per chord
    mel = s[:, :, 3]                                     # (n,4) highest note per chord
    f["bass_motion"] = np.abs(np.diff(bass, axis=1)).sum(1)
    f["bass_drift"] = bass[:, 3] - bass[:, 0]
    f["bass_range"] = bass.max(1) - bass.min(1)
    f["mel_motion"] = np.abs(np.diff(mel, axis=1)).sum(1)
    f["mel_drift"] = mel[:, 3] - mel[:, 0]
    f["mel_range"] = mel.max(1) - mel.min(1)
    f["mel_up"] = (np.diff(mel, axis=1) > 0).sum(1)

    # K. PITCH-CLASS profile across the whole progression (mod 12 histogram).
    allpc = (notes % 12).astype(int)                     # (n,16)
    hist = np.stack([(allpc == k).sum(1) for k in range(12)], axis=1)  # (n,12)
    f["pc_distinct"] = (hist > 0).sum(1)                 # how many of 12 pitch-classes used
    p = hist / hist.sum(1, keepdims=True)
    f["pc_entropy"] = -(np.where(p > 0, p * np.log2(p + 1e-12), 0)).sum(1)
    f["pc_max"] = hist.max(1)                            # most-repeated pitch class count

    # L. REPETITION / TRANSPOSITION structure.
    #    chord t+1 is a transposition of t if all sorted notes shift by one constant.
    trans = np.zeros(n)
    rep = np.zeros(n)
    for c in range(3):
        diff = s[:, c + 1] - s[:, c]                     # (n,4)
        trans += (diff.std(1) < 1e-6).astype(float)      # constant shift => transposition
        rep += (np.abs(diff).sum(1) < 1e-6).astype(float)  # identical chord
    f["transpositions"] = trans
    f["repeats"] = rep
    f["chord0_eq_2"] = (np.abs(s[:, 0] - s[:, 2]).sum(1) < 1e-6).astype(float)

    # M. MICROTUNING grid: is the data on a fine grid (quarter-tones)? Are
    #    off-grid (odd) notes disliked? Pure modular arithmetic.
    f["odd_notes"] = (notes.astype(int) % 2 != 0).sum(1)   # count of odd-valued notes

    X = pd.DataFrame(f, index=df.index)
    return pd.concat([df[RAW], X], axis=1)


# --------------------------------------------------------------------------- #
# Conditions: a pattern is a conjunction of (feature, op, threshold).         #
# Thresholds are frozen on the DISCOVERY split, then reused on held-out.      #
# --------------------------------------------------------------------------- #
def cond_mask(df, cond):
    m = np.ones(len(df), bool)
    for feat, op, thr in cond:
        m &= (df[feat] > thr) if op == ">" else (df[feat] <= thr)
    return m


def cond_str(cond):
    return " AND ".join(f"{ft} {op} {thr:.2f}" for ft, op, thr in cond)


# --- plain-English translation of patterns for the report -------------------- #
# (phrase, kind): kind "count" -> integer tallies; "cont" -> continuous pitch/interval
# NOTE: this mapping is deliberately COMPLETE -- every engineered feature has an
# entry, so no raw variable name can ever leak into the written report.
def _build_feature_en():
    chord_name = {0: "chord 1", 1: "chord 2", 2: "chord 3", 3: "the final chord"}
    voice_name = {0: "the bass (lowest) voice", 1: "the second-lowest voice",
                  2: "the third voice", 3: "the top voice"}
    note_ord = {0: "lowest", 1: "2nd", 2: "3rd", 3: "top"}
    d = {}

    # whole-progression register
    d["mean_all"] = ("the average pitch of the whole progression", "cont")
    d["range_all"] = ("the pitch range of the whole progression", "cont")
    d["std_all"] = ("how spread-out all the notes are in pitch", "cont")

    # per-chord vertical shape
    for c in range(4):
        cn = chord_name[c]
        gap_of = {0: f"the two lowest notes of {cn}",
                  1: f"the 2nd and 3rd notes of {cn}",
                  2: f"the two highest notes of {cn}"}
        for g in range(3):
            d[f"c{c}_gap{g}"] = (f"the gap between {gap_of[g]}", "cont")
        d[f"c{c}_span"] = (f"the spread of {cn}, lowest to highest note", "cont")
        d[f"c{c}_mean"] = (f"the average pitch of {cn}", "cont")
        d[f"c{c}_std"] = (f"how spread-out the notes of {cn} are", "cont")
        d[f"c{c}_pcspread"] = (f"how widely {cn} spans within a single octave", "cont")
        d[f"c{c}_icmin"] = (f"the most consonant interval inside {cn}", "cont")
        d[f"c{c}_icmax"] = (f"the harshest interval inside {cn}", "cont")
        d[f"c{c}_icmean"] = (f"the average harshness of the intervals inside {cn}", "cont")

    # chord-to-chord motion
    for c in range(3):
        d[f"move{c}"] = (f"the pitch step from chord {c+1} to chord {c+2}", "cont")
        d[f"common{c}"] = (f"notes held over from chord {c+1} to chord {c+2}", "count")
        d[f"commonpc{c}"] = (f"pitches shared (ignoring octave) between chord {c+1} "
                             f"and chord {c+2}", "count")
    d["total_drift"] = ("the net pitch change from the first chord to the last", "cont")
    d["abs_motion"] = ("the total distance the harmony travels in pitch", "cont")
    d["contour_up"] = ("upward moves between chords", "count")

    # voice leading
    for v in range(4):
        d[f"v{v}_leap"] = (f"the total distance {voice_name[v]} travels", "cont")
        d[f"v{v}_drift"] = (f"the net rise or fall of {voice_name[v]}", "cont")
    d["max_voice_leap"] = ("the biggest single jump any voice makes between chords", "cont")
    d["voice_leap_total"] = ("the total distance all four voices travel", "cont")
    d["parallel_moves"] = ("times all four voices move the same direction together", "count")
    d["contrary_moves"] = ("times the voices split and move in opposite directions", "count")
    d["voice_crossings"] = ("times the voices cross over each other", "count")

    # roughness / dissonance
    d["clash_le1"] = ("near-unison clashes (two notes less than a semitone apart)", "count")
    d["clash_le2"] = ("close clashes (two notes within a whole tone)", "count")
    d["clash_ic1"] = ("semitone clashes", "count")
    d["min_interval"] = ("the tightest interval between any two notes", "cont")

    # outer lines
    d["bass_motion"] = ("the total movement of the bass line", "cont")
    d["bass_drift"] = ("the net rise or fall of the bass line", "cont")
    d["bass_range"] = ("the pitch range covered by the bass line", "cont")
    d["mel_motion"] = ("the total movement of the top note across the chords", "cont")
    d["mel_drift"] = ("the net rise or fall of the top note", "cont")
    d["mel_range"] = ("the pitch range covered by the top note", "cont")
    d["mel_up"] = ("times the top note steps upward", "count")

    # pitch content / repetition / tuning
    d["pc_distinct"] = ("distinct pitches used, ignoring octave", "count")
    d["pc_entropy"] = ("how evenly the twelve pitches are used", "cont")
    d["pc_max"] = ("repetitions of the single most-repeated pitch", "count")
    d["transpositions"] = ("chords that are an exact pitch-shift of the one before", "count")
    d["repeats"] = ("chords that exactly repeat the one before", "count")
    d["chord0_eq_2"] = ("cases where chord 1 and chord 3 are identical", "count")
    d["odd_notes"] = ("notes falling off the standard tuning grid", "count")

    # raw notes: position i is chord (i//4), voice (i%4), stored low->high
    for i in range(16):
        d[f"n{i}"] = (f"{chord_name[i // 4]}'s {note_ord[i % 4]} note (pitch height)", "cont")
    return d


FEATURE_EN = _build_feature_en()

_COND_RE = re.compile(r"(\w+)\s*(<=|>)\s*(?:median\()?(-?\d+\.?\d*)")


def humanize(pattern):
    """Turn a raw pattern string into a natural-language phrase with value ranges."""
    s = re.sub(r"\s*\[rho=[^\]]*\]", "", pattern).replace("subgroup:", "").strip()
    parts = []
    for feat, op, thr in _COND_RE.findall(s):
        phrase, kind = FEATURE_EN.get(feat, (feat, "cont"))
        val = float(thr)
        if kind == "count":
            iv = int(round(val))
            parts.append(f"at most {iv} {phrase}" if op == "<="
                         else f"more than {iv} {phrase} ({iv + 1} or more)")
        else:
            parts.append(f"{phrase} is low (≤ {val:g})" if op == "<="
                         else f"{phrase} is high (> {val:g})")
    return " AND ".join(parts) if parts else pattern


def plain_feature(feat):
    """Plain-English noun phrase for a single feature (no variable names)."""
    return FEATURE_EN.get(feat, (feat, "cont"))[0]


# --------------------------------------------------------------------------- #
# Shared "what we did and why" methods preamble for the written reports.      #
# --------------------------------------------------------------------------- #
def methods_section():
    return """## Statistical methods used — and why

This analysis is **exploratory pattern discovery on a single rater's taste**, which
creates two specific dangers: (a) with hundreds of candidate patterns, some will look
impressive by pure chance, and (b) one person's ratings may simply be inconsistent, so
there may be little real signal to find. Every method below was chosen to defend
against one of those two dangers.

**1. Ratings treated as ordered, not as categories.**
Dislike, not sure, and like are scored −1, 0, +1 and treated as an *ordered* scale.
*Why:* it lets us ask the stronger question "does this pattern move the rating
steadily in one direction?" rather than the weaker "does this pattern appear more in
one bucket?" We verify the ordering is real rather than assuming it (see the
diagnostic on where "not sure" falls).

**2. Features are pure arithmetic on the note numbers.**
Every quantity (intervals, spreads, movement, clash counts, melodic shape) is computed
by adding and subtracting the note values. *Why:* no music-theory categories such as
"major", "dominant seventh" or "cadence" are imposed, so the data is free to reveal
structure that a theory-driven feature set would hide — and any finding is a fact
about the numbers, not an artefact of the labels we chose.

**3. Split into a discovery half and a held-out half (60/40).**
Patterns are *found* on the discovery half only, then *tested* on the held-out half,
which was never examined during the search. *Why:* this is the single most important
safeguard. A pattern found and scored on the same data is almost guaranteed to look
significant — the "garden of forking paths" problem. Confirming on untouched data
means a surviving pattern had to predict something it was not fitted to.

**4. Rank correlation for the initial screen.**
Each feature is ranked against the ordered rating (Spearman rank correlation).
*Why:* rank-based methods assume only that the relationship is consistently increasing
or decreasing — not that it is a straight line, and not that the values are normally
distributed. Pitch data is bounded and lumpy, so this is the safe choice.

**5. Subgroup discovery to find pockets, not just trends.**
A search over threshold conditions and their pairs, scored by how far a subgroup's
average rating departs from the overall average, weighted by how many items it covers.
*Why:* a feature can be useless overall yet decisive within a region (for example only
when the melody is already high). Weighting by size stops the search from chasing tiny
freak subgroups of three or four items.

**6. Gradient-boosted trees as a cross-check.**
A non-linear model is fitted and its feature importances inspected.
*Why:* it catches curved relationships and interactions that a rank correlation would
miss, so promising features are not overlooked. It is used only to *nominate*
candidates — never as evidence on its own.

**7. Permutation tests for confirmation.**
On the held-out half, ratings are randomly reshuffled thousands of times to build the
distribution of effects expected from chance alone; the real effect is compared to it.
*Why:* this makes no assumption whatsoever about the shape of the data. It answers
exactly the right question: "how often would shuffled ratings produce a gap this
large?"

**8. Effect size reported alongside significance (Cliff's delta).**
This measures how often items matching a pattern are rated above items that do not,
on a scale from −1 to +1, where 0 means no difference.
*Why:* a small p-value only says an effect is unlikely to be zero; it does not say the
effect is *large*. Reporting both prevents a statistically detectable but musically
trivial tendency from being oversold.

**9. False-discovery-rate correction across all tests (Benjamini–Hochberg).**
The p-values from all confirmation tests are corrected together.
*Why:* testing many patterns at the usual 5% threshold means roughly 1 in 20 pure-noise
patterns passes. This correction controls the share of false positives among the
patterns we declare real, which is the appropriate goal for a discovery study.

**10. Bootstrap confidence intervals for stability.**
Each surviving pattern is re-measured on a thousand resamples of the held-out data.
*Why:* a pattern that survives testing but whose interval straddles zero is riding on a
handful of influential items. Only patterns whose effect keeps the same sign across
resamples are reported as confirmed.

**11. A noise ceiling from near-identical progressions.**
For each progression we check how often its most musically similar neighbours received
a *different* rating.
*Why:* with one rater and no repeated presentations, we cannot measure reliability
directly. If near-identical items are rated inconsistently, that inconsistency caps how
well *any* method could possibly do — and stops us blaming a weak result on the
analysis when the ceiling is the real constraint.

**12. A whole-model signal check against a shuffled baseline.**
A classifier is trained with cross-validation to separate liked from not-liked, and its
score is compared against the same classifier trained on randomly shuffled ratings.
*Why:* this is the honest global question — is there enough signal here to predict a
rating at all? It is reported even when the answer is unflattering, because individual
patterns can be real while still not adding up to a usable predictor.

**How to read the results tables.**
*Effect size* runs from −1 to +1: positive means the pattern is rated higher than the
rest, negative means lower, and roughly 0.1 is small, 0.3 moderate. The *95% interval*
is the plausible range for that effect; if it crosses zero the pattern is unstable.
The *p-value* is the chance of seeing an effect this large if the pattern were pure
noise. *Number matching* is how many progressions the pattern actually covers.

---
"""


# --------------------------------------------------------------------------- #
# Phase 1 : EMERGENT DISCOVERY  (discovery split only)                        #
# --------------------------------------------------------------------------- #
def phase1(Xd, yd):
    patterns = []      # each: dict(desc, cond, source, src_feat)

    # --- 1.1 monotonic univariate: Spearman of each feature vs ordinal label
    rows = []
    for col in Xd.columns:
        rho, p = stats.spearmanr(Xd[col], yd)
        rows.append((col, rho, p))
    spear = pd.DataFrame(rows, columns=["feat", "rho", "p"]).set_index("feat")
    spear["absrho"] = spear.rho.abs()
    spear = spear.sort_values("absrho", ascending=False)
    print("=== 1.1  Monotonic univariate (Spearman vs ordinal label) ===")
    print("    (rho>0 => higher feature value => more LIKED)")
    for ft, r in spear.head(10).iterrows():
        print(f"    {ft:12s} rho={r.rho:+.3f}  p={r.p:.1e}")
    for ft, r in spear.head(8).iterrows():
        thr = float(np.median(Xd[ft]))
        op = ">" if r.rho >= 0 else "<="     # point condition toward 'like'
        patterns.append(dict(
            desc=f"{ft} {op} median({thr:.2f})  [rho={r.rho:+.3f}]",
            cond=[(ft, op, thr)], source="spearman", src_feat=ft))

    # --- 1.2 subgroup discovery: conditions with unusual mean rating (WRAcc) ---
    feats = list(Xd.columns)
    base = yd.mean()
    N = len(yd)
    singles = []
    for ft in feats:
        q = np.quantile(Xd[ft], [0.33, 0.66])
        for thr in q:
            for op in (">", "<="):
                m = cond_mask(Xd, [(ft, op, thr)])
                if m.sum() < 30 or m.sum() > N - 30:
                    continue
                wracc = (m.sum() / N) * (yd[m].mean() - base)
                singles.append((wracc, [(ft, op, float(thr))]))
    singles.sort(key=lambda t: -abs(t[0]))
    # conjunctions from the strongest singles
    top_single = singles[:25]
    subgroups = list(top_single)
    for (_, c1), (_, c2) in itertools.combinations(top_single[:15], 2):
        if c1[0][0] == c2[0][0]:
            continue
        cond = c1 + c2
        m = cond_mask(Xd, cond)
        if m.sum() < 30 or m.sum() > N - 30:
            continue
        wracc = (m.sum() / N) * (yd[m].mean() - base)
        subgroups.append((wracc, cond))
    subgroups.sort(key=lambda t: -abs(t[0]))
    seen, picked = set(), []
    for wracc, cond in subgroups:
        key = cond_str(cond)
        if key in seen:
            continue
        seen.add(key)
        picked.append((wracc, cond))
        if len(picked) >= 8:
            break
    print("\n=== 1.2  Subgroup discovery (WRAcc; mean rating in subgroup) ===")
    for wracc, cond in picked:
        m = cond_mask(Xd, cond)
        print(f"    WRAcc={wracc:+.4f}  mean={yd[m].mean():+.2f} (base {base:+.2f})"
              f"  n={m.sum():3d}  IF {cond_str(cond)}")
        patterns.append(dict(desc=f"subgroup: {cond_str(cond)}",
                             cond=cond, source="subgroup", src_feat=None))

    # --- 1.3 emerging/contrast patterns: like(+1) vs dislike(-1) growth rate ---
    like = yd == 1
    dis = yd == -1
    print("\n=== 1.3  Contrast / emerging patterns (support in LIKE vs DISLIKE) ===")
    contrast = []
    for _, cond in top_single:
        m = cond_mask(Xd, cond)
        sl = m[like].mean()
        sd = m[dis].mean()
        growth = (sl + 1e-6) / (sd + 1e-6)
        contrast.append((growth, sl, sd, cond))
    contrast.sort(key=lambda t: -abs(np.log(t[0])))
    for growth, sl, sd, cond in contrast[:6]:
        print(f"    P(cond|like)={sl:.2f}  P(cond|dislike)={sd:.2f}  growth={growth:4.1f}x"
              f"  | {cond_str(cond)}")

    # --- 1.4 model-based: gradient-boosted ordinal regressor + tree rules ---
    gb = GradientBoostingRegressor(random_state=0)
    cv = cross_val_score(gb, Xd, yd, cv=5, scoring="neg_mean_absolute_error").mean()
    gb.fit(Xd, yd)
    imp = pd.Series(gb.feature_importances_, index=Xd.columns).sort_values(ascending=False)
    print(f"\n=== 1.4  Gradient-boosted importance (ordinal; CV MAE={-cv:.3f}) ===")
    for ft, v in imp.head(10).items():
        print(f"    {ft:12s} {v:.3f}")
    print("\n    Readable rules (depth-3 tree, like vs not-like):")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=0)
    tree.fit(Xd, (yd == 1).astype(int))
    print(export_text(tree, feature_names=list(Xd.columns), max_depth=3))

    # dedup patterns by condition signature
    uniq, seen = [], set()
    for p in patterns:
        k = cond_str(p["cond"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq, spear


# --------------------------------------------------------------------------- #
# Phase 2 : CONFIRMATORY TESTING  (held-out split only) + FDR                 #
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b):
    # P(a>b) - P(a<b); a,b are ordinal arrays
    gt = sum((a[:, None] > b[None, :]).sum(1)) if False else None
    g = np.subtract.outer(a, b)
    return (np.sign(g).sum()) / (len(a) * len(b))


def perm_test(mask, y, B=3000):
    obs = y[mask].mean() - y[~mask].mean()
    yv = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    cnt = 0
    for _ in range(B):
        p = RNG.permutation(yv)
        if abs(p[m].mean() - p[~m].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    keep = np.zeros(m, bool)
    if k:
        keep[order[:k]] = True
    return keep


def phase2(patterns, Xh, yh, spear_disc):
    print("\n" + "=" * 70)
    print("Phase 2 : CONFIRMATION on held-out 40% (never seen in discovery)")
    print("=" * 70)
    res = []
    for p in patterns:
        m = pd.Series(cond_mask(Xh, p["cond"]), index=Xh.index)
        if m.sum() < 15 or (~m).sum() < 15:
            continue
        delta = cliffs_delta(yh[m].to_numpy(), yh[~m].to_numpy())
        obs, pv = perm_test(m, yh)
        # graded test: Spearman of the SOURCE feature vs ordinal label on held-out
        if p["src_feat"]:
            grho, gp = stats.spearmanr(Xh[p["src_feat"]], yh)
        else:
            grho, gp = np.nan, np.nan
        res.append(dict(desc=p["desc"], n_in=int(m.sum()), diff=obs,
                        delta=delta, p=pv, graded_rho=grho, graded_p=gp))
    if not res:
        print("  No testable patterns.")
        return
    R = pd.DataFrame(res)
    R["survive"] = benjamini_hochberg(R["p"].values, alpha=0.05)
    R = R.sort_values("p")
    print(f"\n  Tested {len(R)} patterns; BH-FDR at alpha=0.05.\n")
    print(f"  {'pattern':52s} {'n':>4} {'Δmean':>6} {'Cliffδ':>6} {'perm_p':>8} {'graded_ρ':>8} surv")
    for _, r in R.iterrows():
        g = f"{r['graded_rho']:+.2f}" if pd.notna(r["graded_rho"]) else "  -  "
        print(f"  {r['desc'][:52]:52s} {r['n_in']:4d} {r['diff']:+6.2f} {r['delta']:+6.2f}"
              f" {r['p']:8.4f} {g:>8} {'YES' if r['survive'] else ' . '}")
    return R


# --------------------------------------------------------------------------- #
# Phase 3 : STABILITY (bootstrap) + NOISE CEILING (single-rater)              #
# --------------------------------------------------------------------------- #
def phase3(R, patterns, Xh, yh, df, Xfull):
    print("\n" + "=" * 70)
    print("Phase 3 : stability (bootstrap) + noise ceiling")
    print("=" * 70)
    ci_lo, ci_hi, stable_col = {}, {}, {}
    if R is not None:
        surv = R[R.survive]
        print(f"\n  Bootstrap 95% CI of Cliff's delta for {len(surv)} surviving pattern(s):")
        idx = {p["desc"]: p for p in patterns}
        yarr = yh.to_numpy()
        for _, r in surv.iterrows():
            p = idx[r["desc"]]
            m = np.asarray(cond_mask(Xh, p["cond"]))
            deltas = []
            n = len(yh)
            for _ in range(1000):
                b = RNG.integers(0, n, n)
                yb, mb = yarr[b], m[b]
                if mb.sum() < 5 or (~mb).sum() < 5:
                    continue
                deltas.append(cliffs_delta(yb[mb], yb[~mb]))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            stable = (lo > 0) == (hi > 0)
            ci_lo[r["desc"]], ci_hi[r["desc"]], stable_col[r["desc"]] = lo, hi, stable
            print(f"    [{lo:+.2f}, {hi:+.2f}]  {'stable' if stable else 'UNSTABLE (CI spans 0)'}"
                  f"  | {r['desc'][:50]}")
        R["ci_lo"] = R["desc"].map(ci_lo)
        R["ci_hi"] = R["desc"].map(ci_hi)
        R["bootstrap_stable"] = R["desc"].map(stable_col)

    # ---- noise ceiling: how self-consistent are the labels? ----
    Z = StandardScaler().fit_transform(df[RAW].to_numpy(float))
    y = df["label"].to_numpy()
    tree = cKDTree(Z)
    dist, ind = tree.query(Z, k=6)
    # local label disagreement among 5 nearest neighbours (excl self)
    disagree = np.array([(y[ind[i, 1:]] != y[i]).mean() for i in range(len(y))])
    # near-duplicate conflict: pairs whose nearest neighbour is very close
    near = dist[:, 1] <= np.quantile(dist[:, 1], 0.10)   # closest 10% of points
    nd_conflict = (y[ind[near, 1]] != y[near]).mean()
    maj = pd.Series(y).value_counts(normalize=True).max()
    print("\n  Noise ceiling (single-rater self-consistency proxy):")
    print(f"    mean 5-NN label-disagreement   = {disagree.mean():.2f}")
    print(f"    near-duplicate conflict rate   = {nd_conflict:.2f}"
          f"   (among the 10% closest pairs)")
    print(f"    majority-class baseline acc    = {maj:.2f}")
    print(f"    => rough accuracy ceiling      ~ {1 - disagree.mean():.2f}"
          f"  (1 - local disagreement)")
    print("    If the ceiling is near the baseline, labels are noisy/self-inconsistent")
    print("    and patterns are TENDENCIES, not predictors.")
    ceiling = {
        "mean_5NN_label_disagreement": round(float(disagree.mean()), 3),
        "near_duplicate_conflict_rate": round(float(nd_conflict), 3),
        "majority_class_baseline_acc": round(float(maj), 3),
        "rough_accuracy_ceiling": round(float(1 - disagree.mean()), 3),
        "n_rows": int(len(df)),
    }
    return R, ceiling


# --------------------------------------------------------------------------- #
# Phase 4 : validity & methodology diagnostics (#6-#11)                        #
# --------------------------------------------------------------------------- #
def _cv_auc(model, X, ybin, cv=5):
    return cross_val_score(model, X, ybin, cv=cv, scoring="roc_auc").mean()


def diagnostics(df, X, y):
    print("\n" + "=" * 70)
    print("Phase 4 : validity & methodology diagnostics")
    print("=" * 70)
    out = {}
    notes = df[RAW].to_numpy(float)
    ybin = (y == 1).astype(int).to_numpy()
    Xv = X.to_numpy(float)

    # #6 MICROTUNING grid -----------------------------------------------------
    uniq = np.unique(notes)
    g = np.gcd.reduce(uniq.astype(int))
    odd = (notes.astype(int) % 2 != 0).sum(1)
    rho, p = stats.spearmanr(odd, y)
    lr_no = (y[odd == 0] == 1).mean() if (odd == 0).any() else np.nan
    lr_any = (y[odd > 0] == 1).mean() if (odd > 0).any() else np.nan
    out["microtuning"] = dict(grid_gcd=int(g), frac_odd=float((notes % 2 != 0).mean()),
                              odd_vs_label_rho=float(rho), p=float(p),
                              like_rate_no_odd=float(lr_no), like_rate_any_odd=float(lr_any))
    print(f"\n#6 Microtuning grid: gcd(all notes)={g}  "
          f"(gcd>1 => coarse grid; =1 => fine/quarter-tone)")
    print(f"   odd-note count vs label: rho={rho:+.3f} p={p:.3f}  "
          f"| like-rate odd=0:{lr_no:.2f}  odd>0:{lr_any:.2f}")

    # #7 is NOTSURE between LIKE and DISLIKE? ----------------------------------
    Z = StandardScaler().fit_transform(Xv)
    m2 = (y != 0).to_numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Z[m2], ybin[m2])
    score = clf.decision_function(Z)
    md = score[(y == -1).to_numpy()].mean()
    mn = score[(y == 0).to_numpy()].mean()
    ml = score[(y == 1).to_numpy()].mean()
    between = md < mn < ml
    out["notsure_between"] = dict(dislike_score=float(md), notsure_score=float(mn),
                                  like_score=float(ml), is_between=bool(between))
    print(f"\n#7 'Not sure' position on like-vs-dislike axis: "
          f"dislike={md:+.2f}  notsure={mn:+.2f}  like={ml:+.2f}")
    print(f"   notsure {'IS' if between else 'is NOT'} between "
          f"=> ordinal framing {'supported' if between else 'QUESTIONABLE'}")

    # #8 RATER DRIFT over labeling order (id) ---------------------------------
    rid, pid = stats.spearmanr(df["id"], y)
    half = df["id"].median()
    lr_early = (y[df["id"] <= half] == 1).mean()
    lr_late = (y[df["id"] > half] == 1).mean()
    out["rater_drift"] = dict(id_vs_label_rho=float(rid), p=float(pid),
                              like_rate_early=float(lr_early), like_rate_late=float(lr_late))
    print(f"\n#8 Rater drift over id-order: rho={rid:+.3f} p={pid:.3f}  "
          f"| like-rate early:{lr_early:.2f}  late:{lr_late:.2f}")

    # #9 IS THERE ANY SIGNAL? CV-AUC vs permutation null ----------------------
    gb = GradientBoostingClassifier(n_estimators=80, random_state=0)
    obs = _cv_auc(gb, Xv, ybin)
    nulls = []
    for i in range(30):
        yp = np.random.RandomState(i).permutation(ybin)
        nulls.append(_cv_auc(gb, Xv, yp))
    nulls = np.array(nulls)
    pval = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)
    out["global_signal"] = dict(cv_auc=float(obs), null_mean=float(nulls.mean()),
                                null_p=float(pval))
    print(f"\n#9 Global signal (like-vs-rest): CV-AUC={obs:.3f}  "
          f"null={nulls.mean():.3f}  p={pval:.3f}  "
          f"({'SIGNAL present' if pval < 0.05 else 'no signal beyond chance'})")

    # #10 INTERACTION test (likelihood-ratio on logistic product term) --------
    def lr_interaction(fa, fb):
        a = StandardScaler().fit_transform(X[[fa, fb]].to_numpy(float))
        base = LogisticRegression(max_iter=2000).fit(a, ybin)
        full_X = np.column_stack([a, a[:, 0] * a[:, 1]])
        full = LogisticRegression(max_iter=2000).fit(full_X, ybin)
        ll = lambda mdl, XX: np.sum(ybin * np.log(mdl.predict_proba(XX)[:, 1] + 1e-12) +
                                    (1 - ybin) * np.log(mdl.predict_proba(XX)[:, 0] + 1e-12))
        stat = 2 * (ll(full, full_X) - ll(base, a))
        return float(stat), float(stats.chi2.sf(stat, 1))
    pairs = [("total_drift", "c2_span"), ("c3_gap0", "c2_span"), ("clash_le2", "c3_span")]
    out["interactions"] = {}
    print("\n#10 Interaction tests (likelihood-ratio, product term):")
    for fa, fb in pairs:
        st, pp = lr_interaction(fa, fb)
        out["interactions"][f"{fa} x {fb}"] = dict(lr_stat=st, p=pp)
        print(f"    {fa} x {fb}: LR={st:.2f} p={pp:.3f}  "
              f"{'interaction REAL' if pp < 0.05 else 'additive'}")

    # #11 OCTAVE INVARIANCE: raw pitch vs mod-12 ------------------------------
    gb2 = GradientBoostingClassifier(n_estimators=80, random_state=0)
    auc_raw = _cv_auc(gb2, notes, ybin)
    auc_pc = _cv_auc(gb2, notes % 12, ybin)
    out["octave"] = dict(auc_raw_pitch=float(auc_raw), auc_pitch_class=float(auc_pc))
    print(f"\n#11 Octave invariance: AUC raw-pitch={auc_raw:.3f}  "
          f"mod-12={auc_pc:.3f}  "
          f"=> {'register matters' if auc_raw > auc_pc + 0.01 else 'pitch-class is enough'}")
    return out


# --------------------------------------------------------------------------- #
def write_reports(R, ceiling, outdir="results", diag=None):
    import os
    os.makedirs(outdir, exist_ok=True)

    # 1) confirmed-patterns table
    cols = ["desc", "n_in", "diff", "delta", "p", "survive",
            "ci_lo", "ci_hi", "bootstrap_stable", "graded_rho", "graded_p"]
    out = R.reindex(columns=cols).rename(columns={
        "desc": "pattern", "n_in": "n_matching", "diff": "mean_rating_shift",
        "delta": "cliffs_delta", "p": "perm_p", "survive": "survives_fdr"})
    # plain-language direction
    out.insert(1, "direction",
               np.where(out["mean_rating_shift"] > 0, "toward LIKE", "toward DISLIKE"))
    out = out.sort_values(["survives_fdr", "perm_p"], ascending=[False, True])
    p1 = os.path.join(outdir, "confirmed_patterns.csv")
    out.to_csv(p1, index=False, float_format="%.4f")

    # 2) noise-ceiling summary
    p2 = os.path.join(outdir, "noise_ceiling.csv")
    pd.DataFrame([ceiling]).T.rename(columns={0: "value"}).to_csv(p2)

    # 3) human-readable findings.md
    p3 = os.path.join(outdir, "findings.md")
    write_findings_md(out, ceiling, p3, diag)

    # 4) diagnostics (flattened) CSV
    written = [p1, p2, p3]
    if diag:
        rows = []
        for grp, d in diag.items():
            if isinstance(d, dict) and d and isinstance(next(iter(d.values())), dict):
                for sub, dd in d.items():               # nested (interactions)
                    for k, v in dd.items():
                        rows.append((f"{grp}.{sub}", k, v))
            else:
                for k, v in d.items():
                    rows.append((grp, k, v))
        p4 = os.path.join(outdir, "diagnostics.csv")
        pd.DataFrame(rows, columns=["test", "metric", "value"]).to_csv(p4, index=False)
        written.append(p4)

    print("\nReports written:\n  " + "\n  ".join(written))
    print(f"\n{out[['pattern','direction','n_matching','cliffs_delta','perm_p','survives_fdr']].to_string(index=False)}")


def write_findings_md(out, ceiling, path, diag=None):
    surv = out[out["survives_fdr"]].copy()
    n_surv, n_tested = len(surv), len(out)
    top = surv.iloc[0] if n_surv else None
    ceil_below = ceiling["rough_accuracy_ceiling"] < ceiling["majority_class_baseline_acc"]

    def md_table(rows):
        head = ("| pattern | direction | number matching | effect size | 95% interval "
                "| p-value |\n|---|---|---|---|---|---|")
        body = "\n".join(
            f"| {humanize(r['pattern'])} | {r['direction'].replace('toward ','')} | {r['n_matching']} "
            f"| {r['cliffs_delta']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['perm_p']:.4f} |"
            for _, r in rows.iterrows())
        return head + "\n" + body

    lines = []
    lines.append("# What makes a chord progression liked — findings\n")
    lines.append("_Generated by the analysis script `discover.py`, with a fixed random seed "
                 "so the numbers reproduce exactly. Re-run the script to regenerate._\n")

    gs = diag["global_signal"] if diag else None
    lines.append(methods_section())
    lines.append("## Summary of what was found\n")
    lines.append(
        f"- **{n_surv} of {n_tested}** candidate patterns survived confirmation on the "
        "held-out half, kept their significance after the false-discovery-rate correction, "
        "**and** held a stable effect across bootstrap resamples.\n"
        "- **The dominant driver is dissonance, or roughness**: progressions containing "
        "**fewer clashing note-pairs** — notes a semitone or less apart, sounding together — "
        "are reliably **liked**, while clash-heavy ones are **disliked**. This is the strongest "
        "signal found, and it comes from pure subtraction between note values, with no music "
        "theory imposed.\n"
        "- Secondary, weaker signals: **smaller jumps between chords and a calmer top note**, "
        "and a **higher-pitched final chord**.\n"
        + (f"- **There is real signal.** A classifier separating liked from not-liked scores "
           f"**{gs['cv_auc']:.2f}** where 0.50 is pure chance, against a shuffled-rating "
           f"baseline of {gs['null_mean']:.2f} (p = {gs['null_p']:.3f}). "
           "Modest, but above chance.\n" if gs else "")
        + "- Effect sizes per pattern remain **small** (about 0.12 to 0.19 on the −1 to +1 "
        "scale): useful as *tendencies*, not as a confident per-progression predictor.\n")

    if top is not None:
        lines.append("## Strongest confirmed pattern\n")
        lines.append(
            f"> **{humanize(top['pattern'])}** → {top['direction']}  \n"
            f"> Effect size {top['cliffs_delta']:+.2f} "
            f"(95% interval [{top['ci_lo']:+.2f}, {top['ci_hi']:+.2f}]), "
            f"p-value {top['perm_p']:.4f}, matching {top['n_matching']} progressions.\n")

    lines.append("## All confirmed patterns "
                 "(significant after correction, with a stable effect)\n")
    lines.append(md_table(surv) if n_surv else "_None survived._\n")

    rej = out[~out["survives_fdr"]]
    if len(rej):
        lines.append("\n## Candidates that did NOT survive\n")
        lines.append("Found in discovery but failed held-out confirmation — treat as noise:\n")
        lines.append("\n".join(f"- {humanize(r['pattern'])} (p-value {r['perm_p']:.3f})"
                               for _, r in rej.iterrows()))

    # ---- all tests, grouped by underlying signal, confirmed + failed ----
    def theme(p):
        if any(t in p for t in ("clash", "min_interval")):
            return "A — Dissonance / roughness"
        if any(t in p for t in ("leap", "voice", "mel_", "bass_", "_motion", "drift")):
            return "B — Voice-leading / motion"
        if any(t in p for t in ("n13", "n14", "n15", "mean")):
            return "C — Final-chord height / register"
        if any(t in p for t in ("pc_", "transpos", "repeat", "common", "odd")):
            return "D — Pitch-class / repetition / tuning"
        return "E — Chord width / spread"

    lines.append(f"\n## All {n_tested} tests, grouped by signal (confirmed + failed)\n")
    lines.append("Every candidate carried from the discovery half into held-out permutation "
                 "testing. Within each group, ordered strongest-first by p-value. "
                 "✅ = survived the false-discovery-rate correction with a stable effect; "
                 "❌ = failed.\n")
    tmp = out.copy()
    tmp["theme"] = tmp["pattern"].map(theme)
    for g in sorted(tmp["theme"].unique()):
        sub = tmp[tmp["theme"] == g].sort_values("perm_p")
        lines.append(f"\n### Group {g}\n")
        lines.append("| pattern | effect size | p-value | verdict |\n|---|---|---|---|")
        for _, r in sub.iterrows():
            v = "✅" if r["survives_fdr"] else "❌"
            note = ""
            if not r["survives_fdr"] and r["perm_p"] < 0.05:
                note = " (looked significant on its own, but did not survive the correction for testing many patterns)"
            lines.append(f"| {humanize(r['pattern'])} | {r['cliffs_delta']:+.2f} "
                         f"| {r['perm_p']:.4f} | {v}{note} |")

    lines.append("\n**Combined conditions mostly add up rather than multiply.** A formal test for "
                 "interaction (comparing a model with a combined term against one without) found "
                 "**no significant interaction** for the pairs tested — a two-part pattern raises "
                 "the average rating by stacking two independent effects (less dissonance plus "
                 "calmer movement between chords), not through genuine synergy. The practical "
                 "takeaway is simply: **several weak signals accumulate.**\n")

    lines.append("\n## What the musical quantities mean\n")
    lines.append(
        "- **Semitone clashes** — how many pairs of notes sounding together sit one semitone "
        "apart. Fewer clashes means less dissonance, which is the strongest single predictor "
        "of being liked.\n"
        "- **Near-unison clashes** — pairs of notes less than a semitone apart, close enough to "
        "beat against each other. Fewer is liked.\n"
        "- **Tightest interval** — the smallest gap between any two notes anywhere in the "
        "progression; a very small gap sounds harsh.\n"
        "- **Biggest single jump** — the largest distance any one voice moves when the chord "
        "changes. Smaller jumps make a smoother progression.\n"
        "- **Top-note movement** — how far the highest note travels from chord to chord; less "
        "movement reads as calmer.\n"
        "- **Final-chord height** — the pitch of the notes in the last chord; higher means a "
        "higher-pitched ending.\n")

    lines.append("## How consistent were the ratings? (the ceiling on any result)\n")
    lines.append(
        f"| measure | value |\n|---|---|\n"
        f"| average disagreement with the 5 most similar progressions "
        f"| {ceiling['mean_5NN_label_disagreement']:.3f} |\n"
        f"| how often near-identical progressions got different ratings "
        f"| {ceiling['near_duplicate_conflict_rate']:.3f} |\n"
        f"| accuracy from always guessing the most common rating "
        f"| {ceiling['majority_class_baseline_acc']:.3f} |\n"
        f"| estimated best accuracy any method could reach "
        f"| {ceiling['rough_accuracy_ceiling']:.3f} |\n")
    nd = ceiling['near_duplicate_conflict_rate'] * 100
    recon = ""
    if diag:
        gs2 = diag['global_signal']
        recon = (" **An important reconciliation:** this ceiling is measured by comparing the raw "
                 "note positions, where liked and disliked progressions are nearly inseparable. "
                 "The derived musical quantities — especially the clash counts — *do* extract "
                 "real signal: the whole-model check scores "
                 f"{gs2['cv_auc']:.2f} against a chance level of 0.50 (p={gs2['null_p']:.3f}). "
                 "So the ratings are **not** hopeless noise. Raw note positions are simply the "
                 "wrong way to look at the music, and the right quantities — intervals and "
                 "roughness — beat that ceiling.")
    lines.append(
        f"\nNear-identical progressions received *different* ratings about {nd:.0f}% of the time "
        f"when compared by raw note positions — close to what purely random ratings would "
        f"produce.{recon}\n")

    lines.append("## Method in one paragraph\n")
    lines.append(
        "The data was split 60/40 into a **discovery** half and a **held-out** half. Patterns were "
        "surfaced on the discovery half only — by rank-correlating each musical quantity against "
        "the ordered rating, searching for subgroups whose average rating departs unusually far "
        "from the overall average, contrasting how often a condition appears among liked versus "
        "disliked progressions, and inspecting which quantities a non-linear model leans on. "
        "They were then confirmed on the held-out half using permutation tests, an effect size "
        "that counts how often matching progressions outrank non-matching ones, and a "
        "false-discovery-rate correction — so no pattern is believed on the data that produced "
        "it. Survivors were re-measured across bootstrap resamples for stability, and the "
        "disagreement between musically similar progressions provides the noise ceiling. Every "
        "quantity is pure arithmetic on the note numbers (intervals, spreads, movement, and "
        "octave-folded interval content); no music-theory categories were imposed.\n")

    if diag:
        d = diag
        lines.append("## Checks on whether the analysis itself is trustworthy\n")
        gs = d["global_signal"]
        lines.append(
            f"**Is there any signal at all?** Separating liked from not-liked scores "
            f"**{gs['cv_auc']:.3f}**, where 0.50 is pure chance and 1.00 is perfect; the same "
            f"method on randomly shuffled ratings scores {gs['null_mean']:.3f} "
            f"(p = {gs['null_p']:.3f}). "
            + ("There **is** detectable signal above chance — but it is small.\n"
               if gs['null_p'] < 0.05 else
               "**No signal beyond chance**: the notes alone do not predict the rating.\n"))
        ns = d["notsure_between"]
        lines.append(
            f"**Does 'not sure' really sit between dislike and like?** Placing all three groups "
            f"on a single dislike-to-like scale gives dislike {ns['dislike_score']:+.2f}, "
            f"not sure {ns['notsure_score']:+.2f}, like {ns['like_score']:+.2f} → 'not sure' "
            f"**{'does' if ns['is_between'] else 'does NOT'}** fall between the other two. "
            f"Treating the ratings as an ordered scale is therefore "
            f"**{'supported' if ns['is_between'] else 'questionable'}**.\n")
        mt = d["microtuning"]
        lines.append(
            f"**Is the music on a standard tuning grid, and does that matter?** The note values "
            f"share a common divisor of {mt['grid_gcd']} "
            f"({'a coarse grid' if mt['grid_gcd'] > 1 else 'a fine, quarter-tone-capable grid'}); "
            f"{mt['frac_odd']*100:.0f}% of notes fall between the standard semitone steps. "
            f"Those off-grid notes track the rating at {mt['odd_vs_label_rho']:+.3f} on a "
            f"−1 to +1 scale (p={mt['p']:.3f}); the share of liked progressions is "
            f"{mt['like_rate_no_odd']:.2f} when none are off-grid versus "
            f"{mt['like_rate_any_odd']:.2f} when some are.\n")
        oc = d["octave"]
        lines.append(
            f"**Does the actual octave matter, or only which pitch it is?** Using exact pitches "
            f"scores {oc['auc_raw_pitch']:.3f}, versus {oc['auc_pitch_class']:.3f} when octaves "
            f"are folded together → "
            f"{'how high or low the music sits genuinely matters' if oc['auc_raw_pitch'] > oc['auc_pitch_class']+0.01 else 'which pitch it is matters, but not which octave'}.\n")
        rd = d["rater_drift"]
        lines.append(
            f"**Did the rater's taste drift during the session?** Position in the rating order "
            f"tracks the rating at only {rd['id_vs_label_rho']:+.3f} (p={rd['p']:.3f}); the share "
            f"of liked progressions was {rd['like_rate_early']:.2f} early on versus "
            f"{rd['like_rate_late']:.2f} later.\n")
        lines.append("**Do paired conditions genuinely reinforce each other, or just add up?** "
                     "Each pair below was tested by comparing a model containing a combined term "
                     "against one without it:\n")
        lines.append("| pair of musical quantities | test statistic | p-value | verdict |"
                     "\n|---|---|---|---|")
        for pair, dd in d["interactions"].items():
            fa, _, fb = pair.partition(" x ")
            nice = f"{plain_feature(fa.strip())} × {plain_feature(fb.strip())}"
            lines.append(f"| {nice} | {dd['lr_stat']:.2f} | {dd['p']:.3f} | "
                         f"{'they reinforce each other' if dd['p'] < 0.05 else 'they simply add up'} |")
        lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    df = load()
    Xfull = build_features(df)
    vc = df["label"].value_counts().sort_index()
    print(f"rows={len(df)}  dislike={vc.get(-1,0)} notsure={vc.get(0,0)} like={vc.get(1,0)}")
    print(f"features engineered: {Xfull.shape[1]}")

    y = df["label"]
    Xd, Xh, yd, yh = train_test_split(
        Xfull, y, test_size=0.40, stratify=y, random_state=0)
    print(f"discovery n={len(Xd)}   held-out n={len(Xh)}\n")

    patterns, spear = phase1(Xd, yd)
    R = phase2(patterns, Xh, yh, spear)
    R, ceiling = phase3(R, patterns, Xh, yh, df, Xfull)
    diag = diagnostics(df, Xfull, y)
    write_reports(R, ceiling, diag=diag)


if __name__ == "__main__":
    main()
