"""
Full-data discovery on ALL 800 chords -- no held-out split -- as a companion to
the discovery/confirmation split in discover.py.

Why a second version: splitting 60/40 protects against chance findings but
spends 40% of the data on testing, so discovery is weaker (it misses real but
modest signals).  Here every chord is used for discovery, and the protection
against chance comes instead from re-running the ENTIRE search on shuffled
ratings: a finding is only believed if it beats what the same search finds in
pure noise.

  * Every musical quantity is tested (not a shortlist), with
      - a permutation p-value per quantity,
      - Benjamini-Hochberg FDR across all quantities, and
      - step-down max-T (Westfall-Young) family-wise adjustment -- the strict
        tier, which accounts for the fact that we searched over all of them.
  * The subgroup search (single thresholds + pairs of the strongest) is run on
    the real ratings and on 2000 shuffles; each subgroup is judged against the
    distribution of the BEST subgroup the search finds in noise.
  * Forward selection with Freedman-Lane permutation asks which signals are
    INDEPENDENT (e.g. is final-chord height just dissonance in disguise?).
  * Verdicts are compared against the split analysis (results/confirmed_patterns.csv).

What this does NOT show: that a pattern predicts ratings of NEW chords.  That
is the split's job; this version only shows a pattern is real in these 800.

Data: Data/tabulated_chords.csv (id, 16 notes, label).
Run from the repo root:
  /Users/jh22215/anaconda3/bin/python Music-Pattern-Analysis/discover_fulldata.py
"""

import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discover import build_features, cond_str, humanize, plain_feature  # noqa: E402

RNG = np.random.default_rng(0)
RAW = [f"n{i}" for i in range(16)]
DATA = "Data/tabulated_chords.csv"
SPLIT_RESULTS = "Music-Pattern-Analysis/results/confirmed_patterns.csv"
OUTDIR = "Music-Pattern-Analysis/results_fulldata"
B_FEAT = 10000      # shuffles for per-quantity tests
B_SEARCH = 2000     # shuffles of the whole subgroup search
B_FWD = 2000        # shuffles per forward-selection step
B_BOOT = 1000
MIN_GRP = 30        # same minimum subgroup size as discover.py
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Load + features                                                             #
# --------------------------------------------------------------------------- #
def load():
    df = pd.read_csv(DATA, header=None, names=["id"] + RAW + ["label"])
    X = build_features(df)
    dead = [c for c in X.columns if X[c].nunique() < 2]
    X = X.drop(columns=dead)
    # exact duplicate columns (e.g. the top voice IS the top note) would count
    # the same test twice in the multiple-testing family -- keep one, record aliases
    aliases, seen = {}, {}
    for c in X.columns:
        key = X[c].to_numpy(float).tobytes()
        if key in seen:
            aliases.setdefault(seen[key], []).append(c)
        else:
            seen[key] = c
    dup = [c for v in aliases.values() for c in v]
    X = X.drop(columns=dup)
    return df, X, df["label"].to_numpy(float), dead, aliases


def zrank(a):
    """Column-wise average ranks, z-scored: Pearson on these == Spearman."""
    r = stats.rankdata(a, axis=0)
    return (r - r.mean(0)) / r.std(0)


def perm_rows(v, B):
    return RNG.permuted(np.tile(v, (B, 1)), axis=1)


def bh_q(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def fp(p):
    """p-values below 0.001 are at or near the permutation floor -- don't print false precision."""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    r = stats.rankdata(np.concatenate([a, b]))
    return 2.0 * (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2) - 1.0


# --------------------------------------------------------------------------- #
# Phase 1 : every quantity, permutation + FDR + step-down max-T               #
# --------------------------------------------------------------------------- #
def feature_tests(X, y):
    print("=" * 72)
    print(f"Phase 1 : every quantity vs preference, all {len(y)} chords")
    print("=" * 72)
    Xz, yz = zrank(X.to_numpy(float)), zrank(y)
    n, F = Xz.shape
    obs = Xz.T @ yz / n
    absobs = np.abs(obs)
    exceed = np.zeros(F)
    chunks = []
    for _ in range(B_FEAT // 1000):
        Nb = np.abs(perm_rows(yz, 1000) @ Xz / n)
        exceed += (Nb >= absobs - 1e-12).sum(0)
        chunks.append(Nb)
    null = np.vstack(chunks)
    p = (exceed + 1) / (B_FEAT + 1)
    q = bh_q(p)

    # step-down max-T: compare rank-i statistic with the null max over ranks i..F
    order = np.argsort(-absobs)
    suf = np.maximum.accumulate(null[:, order][:, ::-1], axis=1)[:, ::-1]
    raw = ((suf >= absobs[order][None, :] - 1e-12).sum(0) + 1) / (B_FEAT + 1)
    padj = np.empty(F)
    padj[order] = np.maximum.accumulate(raw)

    T = pd.DataFrame({"rho": obs, "perm_p": p, "fdr_q": q, "fwer_p": padj},
                     index=X.columns)
    T["verdict"] = np.where(T.fwer_p < ALPHA, "real (strict)",
                            np.where(T.fdr_q < ALPHA, "real (FDR)", "not significant"))
    T = T.reindex(T.rho.abs().sort_values(ascending=False).index)
    nullmax = null.max(1)
    glob = dict(max_abs_rho=float(absobs.max()),
                null_max_95=float(np.quantile(nullmax, 0.95)),
                null_max_mean=float(nullmax.mean()),
                n_fdr=int((T.fdr_q < ALPHA).sum()),
                n_fwer=int((T.fwer_p < ALPHA).sum()), n_tested=F)
    print(f"  {F} quantities tested.  In shuffled ratings the single best quantity reaches"
          f" |rho| {glob['null_max_mean']:.3f} on average (95th pct {glob['null_max_95']:.3f});")
    print(f"  the real best reaches {glob['max_abs_rho']:.3f}.")
    print(f"  Significant after FDR: {glob['n_fdr']}   after strict family-wise: {glob['n_fwer']}\n")
    for ft, r in T.head(25).iterrows():
        print(f"  {ft:16s} rho={r.rho:+.3f}  q={fp(r.fdr_q)}  fwer={fp(r.fwer_p)}"
              f"  {r.verdict:16s} {plain_feature(ft)}")
    return T, glob


# --------------------------------------------------------------------------- #
# Phase 2 : the whole subgroup search, real vs shuffled                       #
# --------------------------------------------------------------------------- #
def build_conditions(X):
    conds, cols = [], []
    for ft in X.columns:
        v = X[ft].to_numpy(float)
        for thr in np.unique(np.quantile(v, [0.33, 0.66])):
            for op in (">", "<="):
                m = v > thr if op == ">" else v <= thr
                if MIN_GRP <= m.sum() <= len(v) - MIN_GRP:
                    conds.append([(ft, op, float(thr))])
                    cols.append(m)
    return conds, np.column_stack(cols)


def search(M, feats, yc, top_k=15):
    """discover.py's search on one centred label vector: WRAcc of every single
    condition, then of every pair among the top_k singles (different quantities)."""
    N = len(yc)
    single = yc @ M / N
    top = np.argsort(-np.abs(single))[:top_k]
    pairs = [(a, b) for a, b in itertools.combinations(top, 2) if feats[a] != feats[b]]
    if not pairs:
        return single, [], np.array([])
    ia, ib = map(np.array, zip(*pairs))
    PM = M[:, ia] & M[:, ib]
    size = PM.sum(0)
    ok = (size >= MIN_GRP) & (size <= N - MIN_GRP)
    pw = yc @ PM / N
    return single, [p for p, k in zip(pairs, ok) if k], pw[ok]


def subgroup_search(X, y):
    print("\n" + "=" * 72)
    print(f"Phase 2 : subgroup search on all chords, judged against {B_SEARCH} shuffled searches")
    print("=" * 72)
    conds, M = build_conditions(X)
    feats = [c[0][0] for c in conds]
    yc = y - y.mean()
    N = len(y)

    nullmax = np.empty(B_SEARCH)
    for b in range(B_SEARCH):
        s, _, pw = search(M, feats, RNG.permutation(yc))
        nullmax[b] = max(np.abs(s).max(), np.abs(pw).max() if len(pw) else 0)

    single, pairs, pw = search(M, feats, yc)
    cands = [(single[i], conds[i], M[:, i]) for i in np.argsort(-np.abs(single))[:25]]
    cands += [(w, conds[a] + conds[b], M[:, a] & M[:, b]) for (a, b), w in zip(pairs, pw)]
    cands.sort(key=lambda t: -abs(t[0]))

    rows, seen = [], {}
    for w, cond, m in cands:
        key = m.tobytes()
        if key in seen:                         # same chords, different wording
            rows[seen[key]]["aliases"].append(cond_str(cond))
            continue
        a, b = y[m], y[~m]
        boot = []
        for _ in range(B_BOOT):
            i = RNG.integers(0, N, N)
            mb = m[i]
            if mb.sum() >= 5 and (~mb).sum() >= 5:
                boot.append(cliffs_delta(y[i][mb], y[i][~mb]))
        seen[key] = len(rows)
        rows.append(dict(pattern=cond_str(cond), n_matching=int(m.sum()),
                         mean_rating=float(a.mean()), mean_rest=float(b.mean()),
                         wracc=float(w), cliffs_delta=cliffs_delta(a, b),
                         ci_lo=float(np.percentile(boot, 2.5)),
                         ci_hi=float(np.percentile(boot, 97.5)),
                         search_p=float((np.sum(nullmax >= abs(w) - 1e-12) + 1)
                                        / (B_SEARCH + 1)),
                         aliases=[]))
        if len(rows) >= 20:
            break
    S = pd.DataFrame(rows)
    S["survives"] = S.search_p < ALPHA
    S.insert(1, "direction", np.where(S.wracc > 0, "toward LIKE", "toward DISLIKE"))
    S["aliases"] = S.aliases.map(lambda v: " | ".join(v))
    glob = dict(real_best=float(abs(cands[0][0])), null_best_mean=float(nullmax.mean()),
                null_best_95=float(np.quantile(nullmax, 0.95)),
                n_conditions=len(conds), n_survive=int(S.survives.sum()))
    print(f"  {len(conds)} single conditions + pairs.  Best subgroup score in shuffled"
          f" ratings: mean {glob['null_best_mean']:.4f}, 95th pct {glob['null_best_95']:.4f};"
          f" real best {glob['real_best']:.4f}\n")
    for _, r in S.iterrows():
        print(f"  WRAcc={r.wracc:+.4f}  mean={r.mean_rating:+.2f} vs {r.mean_rest:+.2f}"
              f"  n={r.n_matching:3d}  p={fp(r.search_p)} {'YES' if r.survives else ' . '}"
              f"  {r.pattern}")
    return S, glob


# --------------------------------------------------------------------------- #
# Phase 3 : which signals are independent? forward selection, Freedman-Lane  #
# --------------------------------------------------------------------------- #
def forward_select(X, y, sig, max_steps=8):
    print("\n" + "=" * 72)
    print("Phase 3 : independent signals (forward selection, Freedman-Lane permutation)")
    print("=" * 72)
    Xz, yz = zrank(X.to_numpy(float)), zrank(y)
    n, F = Xz.shape
    names = list(X.columns)
    sel, steps = [], []

    def resid(A, V):
        return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]

    for step in range(max_steps):
        A = np.column_stack([np.ones(n)] + [Xz[:, j] for j in sel])
        r = resid(A, yz)
        Xr = resid(A, Xz)
        norm = np.linalg.norm(Xr, axis=0)
        live = norm > 1e-8 * np.sqrt(n)
        live[sel] = False
        Xn = np.where(live, Xr / np.where(live, norm, 1), 0.0)
        rn = np.linalg.norm(r)
        corr = Xn.T @ r / rn
        best = int(np.argmax(np.abs(corr)))
        null = np.abs(perm_rows(r, B_FWD) @ Xn / rn).max(1)
        p = (np.sum(null >= abs(corr[best]) - 1e-12) + 1) / (B_FWD + 1)
        accepted = p < ALPHA
        if accepted:
            sel.append(best)
        A2 = np.column_stack([np.ones(n)] + [Xz[:, j] for j in sel])
        r2 = 1 - np.sum(resid(A2, yz) ** 2) / np.sum(yz ** 2)
        # significant quantities this pick is a near-copy of (|rank corr| >= 0.6)
        rc = Xz.T @ Xz[:, best] / n
        near = [(abs(rc[k]), names[k]) for k in range(F)
                if names[k] in sig and k != best and abs(rc[k]) >= 0.6]
        stands = [nm for _, nm in sorted(near, reverse=True)[:4]]
        steps.append(dict(step=step + 1, feature=names[best], partial_rho=float(corr[best]),
                          perm_p=float(p), accepted=bool(accepted), cum_r2=float(r2),
                          stands_for="; ".join(stands)))
        print(f"  step {step + 1}: {names[best]:16s} partial rho={corr[best]:+.3f}"
              f"  p={fp(p)}  {'ADDED' if accepted else 'stop -- adds nothing beyond noise'}"
              f"  (R2 of ranks={r2:.3f})   {plain_feature(names[best])}")
        if not accepted:
            break
    return pd.DataFrame(steps)


# --------------------------------------------------------------------------- #
# Phase 4 : agreement with the split analysis                                 #
# --------------------------------------------------------------------------- #
_FEAT_RE = re.compile(r"(\w+)\s*(?:<=|>)")


def compare_split(T, df, Xfull, y, aliases):
    print("\n" + "=" * 72)
    print("Phase 4 : agreement with the split analysis (discover.py)")
    print("=" * 72)
    sp = pd.read_csv(SPLIT_RESULTS)
    sp["ok"] = sp.survives_fdr.astype(bool) & (sp.bootstrap_stable == True)  # noqa: E712
    canon = {a: k for k, v in aliases.items() for a in v}
    # a quantity is "confirmed" only if it survived ON ITS OWN; appearing inside a
    # surviving pair is weaker -- the pair's effect may be carried by its partner
    alone, paired, failed = set(), set(), set()
    for _, r in sp.iterrows():
        fs = {canon.get(f, f) for f in _FEAT_RE.findall(r.pattern.split("[")[0])}
        if not r.ok:
            failed.update(fs)
        elif len(fs) == 1:
            alone.update(fs)
        else:
            paired.update(fs)
    paired -= alone
    failed -= alone | paired

    # rank each quantity held on discover.py's own discovery half (same seed)
    idx_d, _ = train_test_split(np.arange(len(y)), test_size=0.40,
                                stratify=y, random_state=0)
    Xd = Xfull.iloc[idx_d]
    rho_d = pd.Series({c: stats.spearmanr(Xd[c], y[idx_d]).statistic for c in T.index})

    A = T.copy()
    A["rho_split_discovery"] = rho_d
    A["rank_split_discovery"] = rho_d.abs().rank(ascending=False).astype(int)
    A["split_status"] = ["confirmed" if f in alone else
                         "confirmed only in a pair" if f in paired else
                         "tested, failed" if f in failed else "not shortlisted"
                         for f in A.index]
    full_real = A.fdr_q < ALPHA
    split_real = A.split_status == "confirmed"
    A["agreement"] = np.select(
        [full_real & split_real, full_real & ~split_real, ~full_real & split_real],
        ["both: real", "full-data only", "split only"], "neither")
    A["plain_english"] = [plain_feature(f) for f in A.index]
    A["aliases"] = [", ".join(aliases.get(f, [])) for f in A.index]
    counts = A.agreement.value_counts().to_dict()
    print(f"  {counts}")
    for f, r in A[A.agreement != "neither"].iterrows():
        print(f"  {r.agreement:15s} {f:16s} rho={r.rho:+.3f} q={fp(r.fdr_q)}"
              f"  split: {r.split_status:24s} (ranked #{r.rank_split_discovery} in discovery half)")
    return A, counts


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def write_report(T, gF, S, gS, FS, A, counts, dead, aliases, X, y):
    os.makedirs(OUTDIR, exist_ok=True)
    T.assign(plain_english=[plain_feature(f) for f in T.index]).to_csv(
        f"{OUTDIR}/fulldata_feature_tests.csv", float_format="%.5f", index_label="feature")
    S.to_csv(f"{OUTDIR}/fulldata_subgroups.csv", index=False, float_format="%.4f")
    FS.to_csv(f"{OUTDIR}/fulldata_independent_signals.csv", index=False, float_format="%.4f")
    A.to_csv(f"{OUTDIR}/split_vs_fulldata.csv", float_format="%.5f", index_label="feature")

    L = []
    W = L.append
    real = T[T.fdr_q < ALPHA]
    W("# What makes a chord progression liked — full-data version\n")
    W("_Generated by `Music-Pattern-Analysis/discover_fulldata.py` on all 800 chords in "
      "`Data/tabulated_chords.csv`, with a fixed random seed so the numbers reproduce. "
      "Companion to the split analysis in `results/findings.md`._\n")

    W("## Why a second version\n")
    W("The split analysis finds patterns on 480 chords and tests them on the other 320. That "
      "protects against chance findings, but discovery only sees 60% of the data and only a "
      "shortlist is ever tested, so real but modest signals can be missed. This version uses "
      "**all 800 chords for discovery** and gets its protection a different way: the entire "
      "search is re-run on shuffled ratings, thousands of times, and a finding is believed "
      "only if it beats what the same search finds in pure noise.\n")
    W("**What the two versions each show.** This version shows a pattern is *real in these "
      "800 chords* — not a product of searching. Only the split shows a pattern *predicts "
      "ratings of chords it was not found on*. A pattern backed by both is the strongest "
      "claim available.\n")

    W("## Statistical methods used — and why\n")
    W("**1. Every quantity is tested, not a shortlist.** All "
      f"{gF['n_tested']} musical quantities are rank-correlated with the ordered rating "
      "(dislike −1, not sure 0, like +1). *Why:* the split only carried its top few forward; "
      "anything ranked lower was never tested at all.\n")
    W("**2. Permutation p-values.** Each correlation is compared with the same correlation "
      f"on {B_FEAT:,} shuffles of the ratings. *Why:* no assumption about the shape of the "
      "data, which is bounded and lumpy.\n")
    W("**3. Two levels of correction for testing many quantities.** *False-discovery rate* "
      "(Benjamini–Hochberg) keeps the share of false findings among those declared real "
      "below 5%. *Family-wise* (step-down max-T, Westfall–Young) is stricter: it keeps the "
      "chance of even **one** false finding below 5%, by comparing each quantity with the "
      "*best* quantity found in each shuffle. *Why both:* FDR is the right tool for discovery; "
      "the family-wise tier marks the findings that would survive the harshest sceptic.\n")
    W("**4. Duplicate quantities counted once.** Some quantities are the same numbers under "
      "two names (the top voice is always the top note). *Why:* testing the same thing twice "
      "would inflate the multiple-testing penalty.\n")
    W("**5. The subgroup search is judged against its own performance on noise.** The same "
      "threshold-and-pairs search as the split analysis is run on the real ratings and on "
      f"{B_SEARCH:,} shuffles; each real subgroup is compared with the *best* subgroup each "
      "shuffled search found. *Why:* this is the direct answer to \"a search always finds "
      "something\" — it measures how good the something has to be.\n")
    W("**6. Independent signals by forward selection.** Quantities are added one at a time, "
      "each time choosing the one that best explains what the already-chosen ones do not, "
      "and stopping when the next addition does no better than it would on shuffled "
      "leftovers (Freedman–Lane permutation). *Why:* many quantities are near-copies of each "
      "other (several all measure dissonance); this separates *how many distinct things* "
      "drive preference from *how many ways there are to measure them*.\n")
    W("**7. Effect sizes with bootstrap intervals.** Cliff's delta (−1 to +1) for each "
      "subgroup. *Caveat:* because the subgroups were chosen on this same data, these "
      "intervals are somewhat optimistic; the permutation test in (5) is what guards "
      "against selection.\n")
    W("---\n")

    W("## Summary\n")
    fs_ok = FS[FS.accepted]
    W(f"- **{gF['n_fdr']} of {gF['n_tested']}** quantities are associated with preference "
      f"after the false-discovery-rate correction; **{gF['n_fwer']}** survive the strict "
      "family-wise tier.\n"
      f"- **The search clearly beats noise.** On shuffled ratings the best of all "
      f"{gF['n_tested']} quantities reaches a correlation of about {gF['null_max_mean']:.2f} "
      f"(95th percentile {gF['null_max_95']:.2f}); on the real ratings it reaches "
      f"**{gF['max_abs_rho']:.2f}**.\n"
      f"- **{gS['n_survive']} of the top {len(S)} subgroups** beat the best subgroup found in "
      f"95% of shuffled searches.\n"
      f"- **{len(fs_ok)} independent signals** "
      + (f"— {', '.join(plain_feature(f) for f in fs_ok.feature)} — "
         if len(fs_ok) else "")
      + f"together account for {fs_ok.cum_r2.iloc[-1] * 100 if len(fs_ok) else 0:.0f}% of "
      "the variation in the ranked ratings. Most of the long list above is those few "
      "signals measured in different ways.\n"
      f"- **Agreement with the split:** {counts.get('both: real', 0)} quantities are backed "
      f"by both versions, {counts.get('full-data only', 0)} only by the full-data version, "
      f"{counts.get('split only', 0)} only by the split.\n")

    W("## Every quantity associated with preference\n")
    W("Sorted by strength. *Correlation* is a rank correlation with the rating (−1 to +1; "
      "negative means more of it is disliked). *FDR q* and *family-wise p* are the two "
      "corrected significance levels; below 0.05 counts.\n")
    W("| musical quantity | correlation | FDR q | family-wise p | verdict | in the split analysis |"
      "\n|---|---|---|---|---|---|")
    for f, r in real.iterrows():
        W(f"| {plain_feature(f)} | {r.rho:+.3f} | {fp(r.fdr_q)} | {fp(r.fwer_p)} "
          f"| {r.verdict} | {A.loc[f, 'split_status']} |")
    W("")

    W("## Independent signals\n")
    W("Each step adds the quantity that best explains what the previous ones leave "
      "unexplained. *Partial correlation* is its association with the rating after the "
      "earlier signals are accounted for.\n")
    W("*Also stands for* lists significant quantities that are near-copies of the pick "
      "(rank correlation of 0.6 or more with it) — the pick is the family's representative.\n")
    W("| step | musical quantity | partial correlation | p-value | added? | cumulative share explained "
      "| also stands for |\n|---|---|---|---|---|---|---|")
    for _, r in FS.iterrows():
        also = "; ".join(plain_feature(f) for f in r.stands_for.split("; ") if f) or "—"
        W(f"| {r.step} | {plain_feature(r.feature)} | {r.partial_rho:+.3f} | {fp(r.perm_p)} "
          f"| {'yes' if r.accepted else 'no — stop'} | {r.cum_r2 * 100:.1f}% | {also} |")
    # "most consonant interval" is discover.py's label for the smallest interval class, but
    # class 1 is a semitone -- so spell out what a selected one actually measures
    for f in FS.feature[FS.accepted]:
        if f.endswith("_icmin"):
            v = X[f].to_numpy()
            dbl, rest = y[v == 0], y[v != 0]
            W(f"\n**Reading step \"{plain_feature(f)}\".** This quantity is the smallest interval "
              "class in the chord, where 0 means two notes are the same pitch an octave apart "
              "(a doubled pitch) and 1 a semitone — so despite the label, low values are not "
              f"simply more consonant. In practice it separates chords with a doubled pitch "
              f"({len(dbl)} chords, liked {np.mean(dbl == 1) * 100:.0f}% of the time) from the "
              f"rest ({len(rest)} chords, liked {np.mean(rest == 1) * 100:.0f}%). It rests on "
              f"only {len(dbl)} chords, so treat it as the least secure of the signals.")
    W("\nThe share explained is small, and that is expected for one person's taste: these are "
      "reliable tendencies, not a formula that pins down each rating.\n")

    W("## Subgroups\n")
    W(f"The best subgroup the search finds in shuffled ratings scores "
      f"{gS['null_best_mean']:.4f} on average (95th percentile {gS['null_best_95']:.4f}). "
      f"Subgroups below are real if they beat that — p-value under 0.05.\n")
    W("| pattern | direction | number matching | average rating (vs rest) | effect size "
      "| 95% interval | p-value vs search on noise |\n|---|---|---|---|---|---|---|")
    for _, r in S.iterrows():
        v = "" if r.survives else " ❌"
        W(f"| {humanize(r.pattern)} | {r.direction.replace('toward ', '')} | {r.n_matching} "
          f"| {r.mean_rating:+.2f} ({r.mean_rest:+.2f}) | {r.cliffs_delta:+.2f} "
          f"| [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {fp(r.search_p)}{v} |")
    W("")

    W("## Where the two versions agree and disagree\n")
    W("*Rank in the split's discovery half* shows how the split saw each quantity — the split "
      "only carried roughly its top 8 forward to testing.\n")
    for label, title in [("both: real", "Backed by both versions — the strongest claims"),
                         ("full-data only", "Real in all 800, but not confirmed by the split"),
                         ("split only", "Confirmed by the split, but not significant on all 800")]:
        sub = A[A.agreement == label]
        W(f"### {title} ({len(sub)})\n")
        if not len(sub):
            W("_None._\n")
            continue
        W("| musical quantity | correlation (all 800) | FDR q | split status "
          "| rank in split's discovery half |\n|---|---|---|---|---|")
        for f, r in sub.iterrows():
            W(f"| {r.plain_english} | {r.rho:+.3f} | {fp(r.fdr_q)} | {r.split_status} "
              f"| #{r.rank_split_discovery} |")
        W("")
    W("**How to read the disagreements.** A quantity that is *real in all 800 but not "
      "shortlisted* was usually just outside the split's top 8: a genuine but modest signal "
      "the split never tested. One *confirmed only in a pair* appeared in a two-part pattern "
      "that survived the split, but failed when tested on its own — the pair's effect may be "
      "carried by its partner. One that was *tested and failed* in the split but is real here "
      "most likely has an effect too small for 320 held-out chords to confirm. Neither case "
      "contradicts the other analysis — they differ in power, not in the data. Where the "
      "full-data version is the only support, the claim is \"real in these 800\", not yet "
      "\"predicts new chords\".\n")

    W("## Housekeeping\n")
    W(f"- Quantities that never vary across the 800 chords were dropped: "
      f"{', '.join(plain_feature(d) for d in dead) or 'none'}.\n"
      "- Quantities that are exactly the same numbers under two names were tested once: "
      + ("; ".join(f"{plain_feature(k)} = {', '.join(plain_feature(x) for x in v)}"
                   for k, v in aliases.items()) or "none") + ".\n")

    with open(f"{OUTDIR}/findings_fulldata.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nWritten to {OUTDIR}/: fulldata_feature_tests.csv, fulldata_subgroups.csv,"
          " fulldata_independent_signals.csv, split_vs_fulldata.csv, findings_fulldata.md")


def main():
    df, X, y, dead, aliases = load()
    vc = pd.Series(y).value_counts().sort_index()
    print(f"rows={len(y)}  dislike={int(vc.get(-1, 0))} notsure={int(vc.get(0, 0))}"
          f" like={int(vc.get(1, 0))}   quantities={X.shape[1]}"
          f"  (dropped {len(dead)} constant, {sum(map(len, aliases.values()))} duplicate)\n")
    T, gF = feature_tests(X, y)
    S, gS = subgroup_search(X, y)
    FS = forward_select(X, y, set(T.index[T.fdr_q < ALPHA]))
    A, counts = compare_split(T, df, X, y, aliases)
    write_report(T, gF, S, gS, FS, A, counts, dead, aliases, X, y)


if __name__ == "__main__":
    main()
