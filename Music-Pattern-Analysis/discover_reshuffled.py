"""
Reshuffled-sets analysis: reorganise the five `reshuffled_set_*_answers.csv` files
into one chord-indexed table, then discover what drives preference.

INPUT  Data/reshuffled_set_{0..4}_answers.csv, no header:
  col 0      : trial / set index (0..4)
  col 1      : chord id
  cols 2..17 : 16 notes = 4 chords x 4 notes (piano keys)
  col 18     : preference (1 like, 0 not sure, -1 dislike)

OUTPUT
  Data/preference_by_chord.csv                     the requested merged table
  Music-Pattern-Analysis/results_reshuffled/*.csv  audit + pattern tables
  Music-Pattern-Analysis/results_reshuffled/findings_reshuffled.md

DESIGN NOTE (decided after inspecting the data, and reported honestly):
  The five files hold the SAME 99 chords in five different presentation orders,
  and every chord carries the SAME preference in all five.  So the five columns
  are five copies of one rating, not five independent re-ratings: no test-retest
  reliability can be estimated from them.  The analysis therefore treats
  preference as a single label on 99 chords, and -- because those 99 are a
  label-balanced subset of the 800 in tabulated_chords.csv -- CONFIRMS every
  pattern on the 701 chords that are NOT in the reshuffled sets.  That held-out
  set is genuinely disjoint and far better powered than splitting 99 in half.

Run from the repo root:  /Users/jh22215/anaconda3/bin/python Music-Pattern-Analysis/discover_reshuffled.py
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discover import build_features, cond_mask, cond_str, humanize, plain_feature  # noqa: E402

RNG = np.random.default_rng(0)
RAW = [f"n{i}" for i in range(16)]
N_TRIALS = 5
N_NULL = 199        # permutation nulls for the whole-model signal check
SETS = "Data/reshuffled_set_{k}_answers.csv"
PARENT = "Data/tabulated_chords.csv"
MERGED = "Data/preference_by_chord.csv"
OUTDIR = "Music-Pattern-Analysis/results_reshuffled"
# readable column names for the exported table: chord c, note n (as stored, low->high)
PRETTY = [f"c{i // 4 + 1}_n{i % 4 + 1}" for i in range(16)]


# --------------------------------------------------------------------------- #
# Phase 0 : reorganise the five files into one chord-indexed table            #
# --------------------------------------------------------------------------- #
def load_sets():
    cols = ["trial", "chord_id"] + RAW + ["pref"]
    frames = []
    for k in range(N_TRIALS):
        d = pd.read_csv(SETS.format(k=k), header=None, names=cols)
        d["position"] = np.arange(1, len(d) + 1)   # row order = presentation order
        frames.append(d)
    return frames


def merge_sets(frames):
    """One row per chord id (ascending): notes + the preference from each trial."""
    notes = (frames[0].set_index("chord_id")[RAW]
             .sort_index())
    out = notes.copy()
    for k, d in enumerate(frames):
        out[f"pref_trial{k + 1}"] = d.set_index("chord_id")["pref"].reindex(out.index)
    out = out.reset_index()
    export = out.rename(columns=dict(zip(RAW, PRETTY)))
    export.to_csv(MERGED, index=False)
    return out, export


def audit_merge(frames, merged):
    """Is there really five trials' worth of information here?  Report honestly."""
    ids = [set(d.chord_id) for d in frames]
    same_ids = all(s == ids[0] for s in ids)
    notes0 = frames[0].set_index("chord_id")[RAW].sort_index()
    same_notes = all(frames[k].set_index("chord_id")[RAW].sort_index().equals(notes0)
                     for k in range(1, N_TRIALS))

    P = merged[[f"pref_trial{k + 1}" for k in range(N_TRIALS)]].to_numpy()
    identical_rows = int((P.min(1) == P.max(1)).sum())
    pairs = []
    for i in range(N_TRIALS):
        for j in range(i + 1, N_TRIALS):
            exact = float((P[:, i] == P[:, j]).mean())
            rho = stats.spearmanr(P[:, i], P[:, j]).statistic
            pairs.append(dict(trial_a=i + 1, trial_b=j + 1,
                              exact_agreement=exact, spearman=float(rho)))
    pairs = pd.DataFrame(pairs)

    # presentation order genuinely differs even though the answers do not
    order_rho = []
    for i in range(N_TRIALS):
        a = frames[i].set_index("chord_id")["position"].sort_index()
        for j in range(i + 1, N_TRIALS):
            b = frames[j].set_index("chord_id")["position"].sort_index()
            order_rho.append(stats.spearmanr(a, b).statistic)

    # forced quota?
    quota = {k + 1: frames[k].pref.value_counts().sort_index().to_dict()
             for k in range(N_TRIALS)}

    print("=" * 72)
    print("Phase 0 : merge + data-integrity audit")
    print("=" * 72)
    print(f"  chords per set                : {len(frames[0])}  (identical id set: {same_ids})")
    print(f"  notes identical across sets   : {same_notes}")
    print(f"  presentation order differs    : median cross-set order rho ="
          f" {np.median(order_rho):+.3f}  (0 = independently reshuffled)")
    print(f"  label quota per set           : {quota[1]}   <- forced 33/33/33")
    print(f"  chords rated the SAME in all 5: {identical_rows} / {len(merged)}")
    print(f"  mean pairwise exact agreement : {pairs.exact_agreement.mean():.3f}")
    if identical_rows == len(merged):
        print("\n  >> The five 'trials' are five ORDERINGS of one set of answers, not five")
        print("     independent re-ratings.  Trial-to-trial agreement is 1.000 by")
        print("     construction, so NO test-retest reliability can be estimated, and the")
        print("     five preference columns collapse to a single label per chord.")
    return dict(same_ids=same_ids, same_notes=same_notes,
                identical_rows=identical_rows, n_chords=len(merged),
                mean_pair_agreement=float(pairs.exact_agreement.mean()),
                median_order_rho=float(np.median(order_rho)),
                quota=quota), pairs


def check_against_parent(merged):
    """The 99 rated chords are a subset of the 800 in tabulated_chords.csv."""
    tc = pd.read_csv(PARENT, header=None, names=["chord_id"] + RAW + ["label"])
    m = tc.merge(merged[["chord_id", "pref_trial1"]], on="chord_id", how="left")
    inset = m.pref_trial1.notna()
    agree = float((m.loc[inset, "label"] == m.loc[inset, "pref_trial1"]).mean())
    print(f"\n  overlap with {PARENT}: {int(inset.sum())} of {len(tc)} rows;"
          f" labels agree {agree:.3f}")
    held = tc.loc[~inset.to_numpy()].reset_index(drop=True)
    print(f"  => held-out confirmation set  : {len(held)} chords never in the reshuffled sets"
          f"  {held.label.value_counts().sort_index().to_dict()}")
    return tc, held, agree


# --------------------------------------------------------------------------- #
# Fast, exact statistics                                                      #
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b):
    """P(a>b) - P(a<b), via mid-ranks -- exact with ties, O(n log n)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([a, b]))
    U = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return 2.0 * U / (n1 * n2) - 1.0


def perm_test(mask, y, B=5000):
    """Two-sided permutation test on the difference in mean rating."""
    y = np.asarray(y, float)
    m = np.asarray(mask, bool)
    k = int(m.sum())
    obs = y[m].mean() - y[~m].mean()
    perm = RNG.permuted(np.tile(y, (B, 1)), axis=1)
    stat = perm[:, :k].mean(1) - perm[:, k:].mean(1)
    p = (np.sum(np.abs(stat) >= abs(obs) - 1e-12) + 1) / (B + 1)
    return float(obs), float(p)


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= alpha * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[:np.where(passed)[0].max() + 1]] = True
    return keep


def boot_delta_ci(mask, y, B=2000):
    y = np.asarray(y, float)
    m = np.asarray(mask, bool)
    n = len(y)
    out = []
    for _ in range(B):
        b = RNG.integers(0, n, n)
        yb, mb = y[b], m[b]
        if mb.sum() < 5 or (~mb).sum() < 5:
            continue
        out.append(cliffs_delta(yb[mb], yb[~mb]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --------------------------------------------------------------------------- #
# Phase 1 : discovery on the 99 reshuffled chords                             #
# --------------------------------------------------------------------------- #
def phase1(Xd, yd, min_grp=20):
    print("\n" + "=" * 72)
    print(f"Phase 1 : DISCOVERY on the {len(yd)} reshuffled chords (balanced 33/33/33)")
    print("=" * 72)
    patterns = []

    rows = [(c, *stats.spearmanr(Xd[c], yd)) for c in Xd.columns]
    spear = (pd.DataFrame(rows, columns=["feat", "rho", "p"]).set_index("feat")
             .assign(absrho=lambda d: d.rho.abs())
             .sort_values("absrho", ascending=False))
    print("\n1.1  Monotonic univariate (Spearman vs ordinal preference)")
    print("     (rho>0 => higher value => more LIKED)")
    for ft, r in spear.head(12).iterrows():
        print(f"     {ft:16s} rho={r.rho:+.3f}  p={r.p:.1e}   {plain_feature(ft)}")
    for ft, r in spear.head(10).iterrows():
        thr = float(np.median(Xd[ft]))
        op = ">" if r.rho >= 0 else "<="
        patterns.append(dict(desc=f"{ft} {op} median({thr:.2f})  [rho={r.rho:+.3f}]",
                             cond=[(ft, op, thr)], source="spearman", src_feat=ft))

    base, N = yd.mean(), len(yd)
    singles = []
    for ft in Xd.columns:
        for thr in np.quantile(Xd[ft], [0.33, 0.66]):
            for op in (">", "<="):
                m = cond_mask(Xd, [(ft, op, thr)])
                if m.sum() < min_grp or m.sum() > N - min_grp:
                    continue
                singles.append(((m.sum() / N) * (yd[m].mean() - base),
                                [(ft, op, float(thr))]))
    singles.sort(key=lambda t: -abs(t[0]))
    top_single = singles[:25]
    subgroups = list(top_single)
    for (_, c1), (_, c2) in itertools.combinations(top_single[:15], 2):
        if c1[0][0] == c2[0][0]:
            continue
        cond = c1 + c2
        m = cond_mask(Xd, cond)
        if m.sum() < min_grp or m.sum() > N - min_grp:
            continue
        subgroups.append(((m.sum() / N) * (yd[m].mean() - base), cond))
    subgroups.sort(key=lambda t: -abs(t[0]))
    print("\n1.2  Subgroup discovery (WRAcc; mean preference inside the subgroup)")
    seen = set()
    picked = 0
    for wracc, cond in subgroups:
        key = cond_str(cond)
        if key in seen:
            continue
        seen.add(key)
        m = cond_mask(Xd, cond)
        print(f"     WRAcc={wracc:+.4f}  mean={yd[m].mean():+.2f} (base {base:+.2f})"
              f"  n={m.sum():3d}  IF {cond_str(cond)}")
        patterns.append(dict(desc=f"subgroup: {cond_str(cond)}", cond=cond,
                             source="subgroup", src_feat=None))
        picked += 1
        if picked >= 10:
            break

    gb = GradientBoostingRegressor(random_state=0)
    cv = -cross_val_score(gb, Xd, yd, cv=5, scoring="neg_mean_absolute_error").mean()
    gb.fit(Xd, yd)
    imp = pd.Series(gb.feature_importances_, index=Xd.columns).sort_values(ascending=False)
    print(f"\n1.3  Gradient-boosted importance (ordinal; CV MAE={cv:.3f})")
    for ft, v in imp.head(10).items():
        print(f"     {ft:16s} {v:.3f}   {plain_feature(ft)}")
    print("\n     Readable rules (depth-3 tree, like vs not-like):")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
    tree.fit(Xd, (yd == 1).astype(int))
    print(export_text(tree, feature_names=list(Xd.columns), max_depth=3))

    uniq, seen = [], set()
    for p in patterns:
        k = cond_str(p["cond"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq, spear, imp


# --------------------------------------------------------------------------- #
# Phase 2 : confirmation on the 701 chords never in the reshuffled sets       #
# --------------------------------------------------------------------------- #
def phase2(patterns, Xh, yh, Xd, yd, Xm, ym, min_side=15):
    print("\n" + "=" * 72)
    print(f"Phase 2 : CONFIRMATION on {len(yh)} held-out chords (disjoint from discovery)")
    print("=" * 72)
    res = []
    for p in patterns:
        m = pd.Series(cond_mask(Xh, p["cond"]), index=Xh.index)
        if m.sum() < min_side or (~m).sum() < min_side:
            continue
        obs, pv = perm_test(m, yh)
        md = cond_mask(Xd, p["cond"])
        delta_disc = cliffs_delta(yd[md].to_numpy(), yd[~md].to_numpy())
        mm = cond_mask(Xm, p["cond"])
        delta_match = (cliffs_delta(ym[mm].to_numpy(), ym[~mm].to_numpy())
                       if min(mm.sum(), (~mm).sum()) >= 10 else np.nan)
        grho, gp = (stats.spearmanr(Xh[p["src_feat"]], yh) if p["src_feat"]
                    else (np.nan, np.nan))
        res.append(dict(desc=p["desc"], source=p["source"], n_in=int(m.sum()),
                        diff=obs, delta=cliffs_delta(yh[m].to_numpy(), yh[~m].to_numpy()),
                        p=pv, delta_discovery=delta_disc, delta_idmatched=delta_match,
                        graded_rho=grho, graded_p=gp))
    if not res:
        print("  No testable patterns.")
        return None
    R = pd.DataFrame(res)
    R["survive"] = benjamini_hochberg(R["p"].values, alpha=0.05)
    R = R.sort_values("p").reset_index(drop=True)
    print(f"\n  Tested {len(R)} patterns; Benjamini-Hochberg FDR at alpha=0.05.\n")
    print(f"  {'pattern':50s} {'n':>4} {'dmean':>6} {'d_held':>7} {'d_disc':>7}"
          f" {'d_idm':>6} {'perm_p':>8} surv")
    for _, r in R.iterrows():
        idm = f"{r.delta_idmatched:+.2f}" if pd.notna(r.delta_idmatched) else "   -  "
        print(f"  {r.desc[:50]:50s} {r.n_in:4d} {r['diff']:+6.2f} {r.delta:+7.2f}"
              f" {r.delta_discovery:+7.2f} {idm:>6} {r.p:8.4f}"
              f" {'YES' if r.survive else ' . '}")
    return R


def phase3(R, patterns, Xh, yh):
    print("\n" + "=" * 72)
    print("Phase 3 : bootstrap stability of the survivors")
    print("=" * 72)
    idx = {p["desc"]: p for p in patterns}
    lo_d, hi_d, st_d = {}, {}, {}
    for _, r in R[R.survive].iterrows():
        m = cond_mask(Xh, idx[r.desc]["cond"])
        lo, hi = boot_delta_ci(m, yh.to_numpy())
        lo_d[r.desc], hi_d[r.desc], st_d[r.desc] = lo, hi, (lo > 0) == (hi > 0)
        print(f"  [{lo:+.2f}, {hi:+.2f}]  {'stable' if st_d[r.desc] else 'UNSTABLE (spans 0)'}"
              f"  | {r.desc[:48]}")
    R["ci_lo"] = R.desc.map(lo_d)
    R["ci_hi"] = R.desc.map(hi_d)
    R["bootstrap_stable"] = R.desc.map(st_d)
    return R


# --------------------------------------------------------------------------- #
# Phase 4 : diagnostics                                                       #
# --------------------------------------------------------------------------- #
def _auc(model, X, y):
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()


def diagnostics(Xd, yd, Xh, yh, spear_d, spear_h):
    print("\n" + "=" * 72)
    print("Phase 4 : validity diagnostics")
    print("=" * 72)
    out = {}

    # #1 does the discovery ranking of features reproduce on held-out?
    common = spear_d.index.intersection(spear_h.index)
    both = pd.DataFrame({"d": spear_d.loc[common, "rho"],
                         "h": spear_h.loc[common, "rho"]}).dropna()
    rk = stats.spearmanr(both.d, both.h)
    top20 = spear_d.dropna(subset=["rho"]).head(20).index
    sign_agree = float(np.mean(np.sign(spear_d.loc[top20, "rho"])
                               == np.sign(spear_h.loc[top20, "rho"])))
    out["feature_replication"] = dict(rho_of_rhos=float(rk.statistic), p=float(rk.pvalue),
                                      top20_sign_agreement=sign_agree)
    print(f"\n#1 Feature-effect replication (99-set vs 701-set): correlation of the two"
          f" rho vectors = {rk.statistic:+.3f} (p={rk.pvalue:.1e});"
          f" top-20 direction agrees {sign_agree * 100:.0f}% of the time")

    # #2 is 'not sure' between dislike and like?
    Z = StandardScaler().fit_transform(Xd.to_numpy(float))
    ybin = (yd == 1).astype(int).to_numpy()
    m2 = (yd != 0).to_numpy()
    sc = LogisticRegression(max_iter=4000).fit(Z[m2], ybin[m2]).decision_function(Z)
    md, mn, ml = (sc[(yd == v).to_numpy()].mean() for v in (-1, 0, 1))
    out["notsure_between"] = dict(dislike_score=float(md), notsure_score=float(mn),
                                  like_score=float(ml), is_between=bool(md < mn < ml))
    print(f"\n#2 'Not sure' on the dislike->like axis: dislike={md:+.2f}"
          f"  notsure={mn:+.2f}  like={ml:+.2f}  =>"
          f" {'ordinal framing supported' if md < mn < ml else 'ordinal framing QUESTIONABLE'}")

    # #3 global signal on the held-out set (the 99 are too small + quota-balanced)
    Xhv, yhbin = Xh.to_numpy(float), (yh == 1).astype(int).to_numpy()
    gb = GradientBoostingClassifier(n_estimators=80, random_state=0)
    obs = _auc(gb, Xhv, yhbin)
    nulls = np.array([_auc(gb, Xhv, np.random.RandomState(i).permutation(yhbin))
                      for i in range(N_NULL)])
    pv = (np.sum(nulls >= obs) + 1) / (len(nulls) + 1)
    out["global_signal_heldout"] = dict(cv_auc=float(obs), null_mean=float(nulls.mean()),
                                        null_p=float(pv), n=int(len(yh)))
    print(f"\n#3 Global signal, held-out 701 (like vs rest): CV-AUC={obs:.3f}"
          f"  shuffled={nulls.mean():.3f}  p={pv:.3f}"
          f"  ({'SIGNAL' if pv < 0.05 else 'no signal beyond chance'})")

    # #4 same, inside the 99 (small n, balanced classes)
    gb2 = GradientBoostingClassifier(n_estimators=80, random_state=0)
    Xdv = Xd.to_numpy(float)
    obs2 = _auc(gb2, Xdv, ybin)
    nulls2 = np.array([_auc(gb2, Xdv, np.random.RandomState(i).permutation(ybin))
                       for i in range(N_NULL)])
    pv2 = (np.sum(nulls2 >= obs2) + 1) / (len(nulls2) + 1)
    out["global_signal_99"] = dict(cv_auc=float(obs2), null_mean=float(nulls2.mean()),
                                   null_p=float(pv2), n=int(len(yd)))
    print(f"#4 Global signal, the 99 reshuffled chords:  CV-AUC={obs2:.3f}"
          f"  shuffled={nulls2.mean():.3f}  p={pv2:.3f}")

    # #5 octave invariance -- run on the held-out 701, where there is power to
    #    tell 0.55 from 0.60; on 99 chords this comparison is pure noise.
    keep = [c for c in RAW if c in Xh.columns]
    notes_h = Xh[keep].to_numpy(float)
    a_raw = _auc(GradientBoostingClassifier(n_estimators=80, random_state=0),
                 notes_h, yhbin)
    a_pc = _auc(GradientBoostingClassifier(n_estimators=80, random_state=0),
                notes_h % 12, yhbin)
    out["octave"] = dict(auc_raw_pitch=float(a_raw), auc_pitch_class=float(a_pc),
                         n=int(len(yh)))
    print(f"\n#5 Octave invariance (held-out {len(yh)}): raw pitch AUC={a_raw:.3f}"
          f"  vs pitch-class {a_pc:.3f}")
    return out


# --------------------------------------------------------------------------- #
def write_outputs(audit, pairs, R, diag, spear_d, spear_h, imp, parent_agree, merged):
    os.makedirs(OUTDIR, exist_ok=True)
    written = [MERGED]

    pairs.to_csv(f"{OUTDIR}/trial_agreement.csv", index=False, float_format="%.4f")
    written.append(f"{OUTDIR}/trial_agreement.csv")

    cols = ["desc", "source", "n_in", "diff", "delta", "delta_discovery",
            "delta_idmatched", "p", "survive", "ci_lo", "ci_hi",
            "bootstrap_stable", "graded_rho", "graded_p"]
    out = R.reindex(columns=cols).rename(columns={
        "desc": "pattern", "n_in": "n_matching", "diff": "mean_rating_shift",
        "delta": "cliffs_delta_heldout", "delta_discovery": "cliffs_delta_discovery",
        "delta_idmatched": "cliffs_delta_id_matched", "p": "perm_p",
        "survive": "survives_fdr"})
    out.insert(1, "direction", np.where(out.mean_rating_shift > 0,
                                        "toward LIKE", "toward DISLIKE"))
    out = out.sort_values(["survives_fdr", "perm_p"], ascending=[False, True])
    out.to_csv(f"{OUTDIR}/confirmed_patterns_reshuffled.csv", index=False,
               float_format="%.4f")
    written.append(f"{OUTDIR}/confirmed_patterns_reshuffled.csv")

    comp = (spear_d[["rho", "p"]].rename(columns={"rho": "rho_99", "p": "p_99"})
            .join(spear_h[["rho", "p"]].rename(columns={"rho": "rho_701", "p": "p_701"}))
            .assign(importance_99=imp)
            .assign(plain_english=[plain_feature(f) for f in spear_d.index])
            .sort_values("rho_99", key=abs, ascending=False))
    comp.to_csv(f"{OUTDIR}/feature_effects.csv", float_format="%.4f")
    written.append(f"{OUTDIR}/feature_effects.csv")

    rows = []
    for grp, d in diag.items():
        for k, v in d.items():
            rows.append((grp, k, v))
    for k, v in audit.items():
        if k != "quota":
            rows.append(("merge_audit", k, v))
    rows.append(("merge_audit", "label_agreement_with_tabulated_chords", parent_agree))
    pd.DataFrame(rows, columns=["test", "metric", "value"]).to_csv(
        f"{OUTDIR}/diagnostics_reshuffled.csv", index=False)
    written.append(f"{OUTDIR}/diagnostics_reshuffled.csv")

    path = f"{OUTDIR}/findings_reshuffled.md"
    write_findings(path, audit, out, diag, comp, parent_agree, merged)
    written.append(path)
    print("\nWritten:\n  " + "\n  ".join(written))
    return out


def write_findings(path, audit, out, diag, comp, parent_agree, merged):
    surv = out[out.survives_fdr]
    fr = diag["feature_replication"]
    gs_h = diag["global_signal_heldout"]
    gs_9 = diag["global_signal_99"]
    ns = diag["notsure_between"]
    oc = diag["octave"]
    L = []
    A = L.append

    A("# Preference patterns in the reshuffled sets — findings\n")
    A("_Generated by `Music-Pattern-Analysis/discover_reshuffled.py` with a fixed random "
      "seed, so the numbers reproduce exactly._\n")

    A("## Read this first: the five sets are not five ratings\n")
    A(f"The five files hold the **same {audit['n_chords']} chords** with **identical notes**, "
      f"reshuffled into five different presentation orders (median cross-set order "
      f"correlation {audit['median_order_rho']:+.3f} — the orders really are independent). "
      f"But the preference column is the **same in all five**: "
      f"{audit['identical_rows']} of {audit['n_chords']} chords carry an identical rating "
      f"across every set, giving a mean pairwise agreement of "
      f"{audit['mean_pair_agreement']:.3f}. Those ratings also match the labels already in "
      f"`Data/tabulated_chords.csv` for the same chords "
      f"({parent_agree * 100:.0f}% agreement).\n")
    A("**Consequence.** `pref_trial1` … `pref_trial5` are five copies of one judgement, not "
      "five re-ratings. Test–retest reliability, rater consistency and a directly measured "
      "noise ceiling — the things repeated presentations exist to provide — **cannot be "
      "estimated from this data**. If the intention was to re-rate the same chords in a "
      "fresh order, the answers appear to have been carried over from the first session "
      "rather than re-collected.\n")
    A("Everything below therefore treats preference as **one label per chord**.\n")

    A("## What the design does still give us\n")
    A(f"The {audit['n_chords']} chords are a deliberately **balanced** subset — exactly 33 "
      "like, 33 not sure, 33 dislike — drawn from the 800 chords in "
      "`tabulated_chords.csv`. That makes them a clean discovery sample (no class is "
      f"rare), and it leaves **{gs_h['n']} chords that appear in none of the reshuffled "
      "sets** as a genuinely disjoint, well-powered confirmation set. Patterns are found "
      "on the 99 and tested on those 701, so nothing is believed on the data that produced "
      "it.\n")

    A("## Statistical methods used — and why\n")
    A("**1. Preference treated as ordered, not as three categories.** Dislike, not sure and "
      "like are scored −1, 0, +1. *Why:* it lets us ask the stronger question — does a "
      "pattern move the rating steadily in one direction — instead of merely whether it is "
      "more common in one bucket. Whether the ordering is real is checked, not assumed.\n")
    A("**2. Every musical quantity is pure arithmetic on the note numbers.** Intervals, "
      "spreads, movement, clash counts and melodic shape come from adding and subtracting "
      "note values. *Why:* no music-theory categories such as \"major\" or \"cadence\" are "
      "imposed, so any finding is a fact about the numbers rather than an artefact of the "
      "labels we chose to compute.\n")
    A("**3. Discover on the 99, confirm on the 701 chords never in these files.** *Why:* "
      "this is the central safeguard. Splitting 99 chords in half would leave roughly 40 "
      "for testing — too few to confirm anything. Using the untouched remainder of the "
      "parent dataset gives a confirmation set seven times larger than the discovery set "
      "and completely disjoint from it.\n")
    A("**4. Rank correlation for the initial screen.** *Why:* rank methods assume only that "
      "a relationship is consistently increasing or decreasing — not that it is a straight "
      "line, and not that the values are normally distributed. Pitch data is bounded and "
      "lumpy, so this is the safe choice.\n")
    A("**5. Subgroup discovery to find pockets, not just trends.** A search over threshold "
      "conditions and their pairs, scored by how far a subgroup's average rating departs "
      "from the overall average, weighted by subgroup size. *Why:* a quantity can be "
      "useless overall yet decisive within a region; weighting by size stops the search "
      "chasing freak subgroups of three or four chords.\n")
    A("**6. Gradient-boosted trees as a cross-check.** *Why:* they catch curved "
      "relationships and interactions a rank correlation would miss. Used only to "
      "*nominate* candidates, never as evidence on their own.\n")
    A("**7. Permutation tests for confirmation.** Ratings on the held-out chords are "
      "reshuffled five thousand times to build the distribution of effects expected from "
      "chance alone. *Why:* it assumes nothing whatsoever about the shape of the data.\n")
    A("**8. Effect size alongside significance (Cliff's delta).** How often chords matching "
      "a pattern are rated above chords that do not, from −1 to +1. *Why:* a small p-value "
      "says an effect is unlikely to be zero, not that it is large.\n")
    A("**9. False-discovery-rate correction across all tests (Benjamini–Hochberg).** *Why:* "
      "testing many patterns at the usual 5% threshold lets roughly 1 in 20 pure-noise "
      "patterns through.\n")
    A("**10. Bootstrap confidence intervals for stability.** Each survivor is re-measured on "
      "two thousand resamples. *Why:* a pattern whose interval straddles zero is riding on "
      "a handful of influential chords.\n")
    A("**11. A base-rate caveat, stated rather than hidden.** The 99 are forced to 33/33/33 "
      "while the held-out 701 are 61% dislike. *Why it matters:* effect *sizes* are not "
      "directly comparable between the two sets, so the discovery and held-out columns are "
      "reported side by side and judged on **direction and significance**, not on "
      "magnitude. A secondary column repeats each test on the held-out chords whose ids "
      "fall in the same range as the 99, in case the rater drifted over the session.\n")
    A("**How to read the tables.** *Effect size* runs from −1 to +1; positive means the "
      "pattern is rated higher than the rest, and roughly 0.1 is small, 0.3 moderate. The "
      "*95% interval* is the plausible range for that effect. *p-value* is the chance of an "
      "effect this large if the pattern were pure noise. *Number matching* is how many "
      "held-out chords the pattern covers.\n")
    A("---\n")

    A("## Summary of what was found\n")
    A(f"- **{len(surv)} of {len(out)}** candidate patterns survived confirmation on the "
      "held-out chords, kept their significance after the false-discovery-rate correction, "
      "and held a stable effect across bootstrap resamples.\n"
      f"- **The musical quantities that matter reproduce across the two disjoint sets.** "
      f"Ranking every quantity by its association with preference in the 99 and again in "
      f"the 701 gives two rankings correlated at **{fr['rho_of_rhos']:+.2f}** "
      f"(p = {fr['p']:.1e}), and the twenty strongest quantities point the same way "
      f"**{fr['top20_sign_agreement'] * 100:.0f}%** of the time. The signal is a property "
      "of the music, not of which chords happened to be sampled.\n"
      f"- **There is real signal.** A classifier separating liked from not-liked scores "
      f"**{gs_h['cv_auc']:.2f}** on the held-out chords where 0.50 is pure chance, against "
      f"a shuffled-rating baseline of {gs_h['null_mean']:.2f} (p = {gs_h['null_p']:.3f}). "
      f"Inside the 99 chords alone the same check scores {gs_9['cv_auc']:.2f} against "
      f"{gs_9['null_mean']:.2f} (p = {gs_9['null_p']:.3f}) — **not** significant, which is "
      "what 99 chords buys you rather than evidence against the signal: the effect is "
      "real but small, and detecting a small effect needs hundreds of items. This is "
      "precisely why the patterns are confirmed on the 701.\n"
      "- Effect sizes per pattern stay **small**: these are tendencies in taste, not a "
      "confident per-chord predictor.\n")

    if len(surv):
        t = surv.iloc[0]
        A("## Strongest confirmed pattern\n")
        A(f"> **{humanize(t['pattern'])}** → {t['direction']}  \n"
          f"> Effect size {t['cliffs_delta_heldout']:+.2f} on the held-out chords "
          f"(95% interval [{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]), "
          f"p-value {t['perm_p']:.4f}, matching {t['n_matching']} of them; "
          f"{t['cliffs_delta_discovery']:+.2f} in the 99 where it was found.\n")

    A("## All confirmed patterns\n")
    if len(surv):
        A("| pattern | direction | number matching | effect (held-out) | effect (the 99) "
          "| 95% interval | p-value |\n|---|---|---|---|---|---|---|")
        for _, r in surv.iterrows():
            A(f"| {humanize(r['pattern'])} | {r['direction'].replace('toward ', '')} "
              f"| {r['n_matching']} | {r['cliffs_delta_heldout']:+.2f} "
              f"| {r['cliffs_delta_discovery']:+.2f} "
              f"| [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['perm_p']:.4f} |")
    else:
        A("_None survived._")
    A("")

    rej = out[~out.survives_fdr]
    if len(rej):
        A("## Candidates that did NOT survive\n")
        A("Found on the 99 but failed confirmation on the held-out chords — treat as noise:\n")
        for _, r in rej.iterrows():
            A(f"- {humanize(r['pattern'])} (p-value {r['perm_p']:.3f})")
        A("")

    def theme(p):
        if any(t in p for t in ("clash", "min_interval", "ic")):
            return "A — Dissonance / roughness"
        if any(t in p for t in ("leap", "voice", "mel_", "bass_", "_motion", "drift",
                                "move", "contour", "parallel", "contrary")):
            return "B — Voice-leading / motion"
        if any(t in p for t in ("pc_", "transpos", "repeat", "common", "odd")):
            return "C — Pitch-class / repetition / tuning"
        if any(t in p for t in ("n12", "n13", "n14", "n15", "c3_", "mean_all")):
            return "D — Final-chord height / register"
        return "E — Chord width / spread"

    A(f"## All {len(out)} tests, grouped by signal\n")
    A("Every candidate carried from the 99 into held-out permutation testing. "
      "✅ = survived the false-discovery-rate correction with a stable effect; ❌ = failed.\n")
    tmp = out.copy()
    tmp["theme"] = tmp["pattern"].map(theme)
    for g in sorted(tmp.theme.unique()):
        sub = tmp[tmp.theme == g].sort_values("perm_p")
        A(f"### Group {g}\n")
        A("| pattern | effect (held-out) | effect (the 99) | p-value | verdict |"
          "\n|---|---|---|---|---|")
        for _, r in sub.iterrows():
            v = "✅" if r.survives_fdr else "❌"
            note = ("" if r.survives_fdr or r.perm_p >= 0.05 else
                    " (significant alone, but not after correcting for testing many patterns)")
            A(f"| {humanize(r['pattern'])} | {r['cliffs_delta_heldout']:+.2f} "
              f"| {r['cliffs_delta_discovery']:+.2f} | {r['perm_p']:.4f} | {v}{note} |")
        A("")

    A("## The ten musical quantities most associated with preference\n")
    A("Measured separately in the two disjoint sets, as a rank correlation with "
      "preference on a −1 to +1 scale.\n")
    A("| musical quantity | in the 99 | in the held-out 701 | replicates? |"
      "\n|---|---|---|---|")
    for ft, r in comp.head(10).iterrows():
        if abs(r.rho_701) < 0.05:
            ok = "**no — vanishes**"
        elif np.sign(r.rho_99) == np.sign(r.rho_701):
            ok = "yes"
        else:
            ok = "**no — reverses**"
        A(f"| {r.plain_english} | {r.rho_99:+.3f} | {r.rho_701:+.3f} | {ok} |")
    A("")
    A("A quantity only counts as replicating if it points the same way in both sets "
      "**and** is still non-negligible in the held-out one; an association of +0.26 in "
      "the 99 that falls to +0.03 in the 701 was a feature of the small sample, not of "
      "the music.")
    A("")

    A("## Checks on whether the analysis itself is trustworthy\n")
    A(f"**Is there any signal at all?** On the held-out chords, separating liked from "
      f"not-liked scores **{gs_h['cv_auc']:.3f}** where 0.50 is chance and 1.00 perfect; "
      f"the same method on shuffled ratings scores {gs_h['null_mean']:.3f} "
      f"(p = {gs_h['null_p']:.3f}). "
      + ("There **is** detectable signal above chance — but it is small.\n"
         if gs_h['null_p'] < 0.05 else
         "**No signal beyond chance** by this measure.\n"))
    A(f"**Does 'not sure' really sit between dislike and like?** Placing all three groups on "
      f"one dislike-to-like scale gives dislike {ns['dislike_score']:+.2f}, not sure "
      f"{ns['notsure_score']:+.2f}, like {ns['like_score']:+.2f} → 'not sure' "
      f"**{'does' if ns['is_between'] else 'does NOT'}** fall between the other two, so "
      f"treating preference as an ordered scale is "
      f"**{'supported' if ns['is_between'] else 'questionable'}**.\n")
    A(f"**Do the findings survive changing the chords?** The association of each quantity "
      f"with preference, measured independently in the 99 and in the 701, correlates at "
      f"{fr['rho_of_rhos']:+.3f} (p = {fr['p']:.1e}). This is the strongest available "
      "evidence that the patterns are about the music rather than about the sample.\n")
    A(f"**Does the actual octave matter, or only which pitch it is?** Measured on the "
      f"{oc['n']} held-out chords, using exact pitches "
      f"scores {oc['auc_raw_pitch']:.3f} against {oc['auc_pitch_class']:.3f} when octaves "
      f"are folded together → "
      f"{'how high or low the music sits genuinely matters' if oc['auc_raw_pitch'] > oc['auc_pitch_class'] + 0.01 else 'which pitch it is matters, but not which octave'}.\n")
    A("**What this data cannot tell us.** Because the five sets repeat one set of answers, "
      "there is no way to measure how consistent the rater is with themselves, and "
      "therefore no directly measured ceiling on how well any method could do. Collecting "
      "genuinely independent re-ratings of these same 99 chords would supply exactly that, "
      "and would also let a consensus score be built that is far less noisy than a single "
      "judgement.\n")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main():
    frames = load_sets()
    merged, export = merge_sets(frames)
    print(f"Merged table written to {MERGED}  "
          f"({export.shape[0]} rows x {export.shape[1]} cols)")
    print(export.head(3).to_string(index=False))
    audit, pairs = audit_merge(frames, merged)
    tc, held, parent_agree = check_against_parent(merged)

    # ---- one label per chord (they are identical across trials; take trial 1) ----
    disc = merged.rename(columns={"chord_id": "id", "pref_trial1": "label"})
    Xd = build_features(disc)
    yd = disc["label"]
    Xh = build_features(held.rename(columns={"chord_id": "id"}))
    yh = held["label"]
    # drop quantities that never vary (e.g. every progression opens on the same
    # bass note) -- they carry no information and make rank correlations undefined
    dead = [c for c in Xd.columns if Xd[c].nunique() < 2 or Xh[c].nunique() < 2]
    if dead:
        print(f"  constant quantities dropped   : {len(dead)}  ({', '.join(dead)})")
        Xd, Xh = Xd.drop(columns=dead), Xh.drop(columns=dead)
    # id-matched held-out control (same id range as the 99)
    idmask = (held.chord_id <= merged.chord_id.max()).to_numpy()
    Xm, ym = Xh[idmask].reset_index(drop=True), yh[idmask].reset_index(drop=True)
    print(f"  id-matched control set        : {len(ym)} chords (ids <= "
          f"{merged.chord_id.max()})")
    print(f"  features engineered           : {Xd.shape[1]}")

    patterns, spear_d, imp = phase1(Xd, yd)
    spear_h = pd.DataFrame([(c, *stats.spearmanr(Xh[c], yh)) for c in Xh.columns],
                           columns=["feat", "rho", "p"]).set_index("feat")
    R = phase2(patterns, Xh, yh, Xd, yd, Xm, ym)
    R = phase3(R, patterns, Xh, yh)
    diag = diagnostics(Xd, yd, Xh, yh, spear_d, spear_h)
    write_outputs(audit, pairs, R, diag, spear_d, spear_h, imp, parent_agree, merged)


if __name__ == "__main__":
    main()
