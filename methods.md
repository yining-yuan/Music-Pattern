# Candidate patterns, statistical testing, and repeated preferences

This note explains the existing chord and melody analyses and how to extend them to preferences collected on several occasions. Sections 1–2 describe the current pipeline, with qualifications where needed. Section 3 proposes how to handle repeated observations; the current scripts do not fit a repeated-measures model.

## 1. How candidate patterns are generated and listed

A **feature** is a numerical description of a musical item. A **candidate pattern** is a condition on one or more features, such as “melody mean pitch > 70” or “few close note pairs AND small top-note movement.” Each item either matches the condition or does not. A candidate is a hypothesis to test, not yet an established preference.

### Musical features

The chord-only analysis represents each progression as four chords with four notes each. Features describe pitch height, within-chord intervals and spread, shared notes, repetitions, and movement between chords. The chord-and-melody analysis adds the ten melody notes and features describing melodic range, mean pitch, steps, leaps, repetition, net rise or fall, and relationships to the chord pitches.

These definitions need to follow the calculations precisely. For example, `clash_le1` counts note pairs separated by **at most one pitch unit**, including unisons; it is not specifically a count of pairs less than a semitone apart. The melody–harmony fit features compare each melody note with pitches found **anywhere in the progression**. They do not measure its fit to the chord sounding at that moment; that would require timing alignment. These numerical summaries are proxies for musical properties.

### Musical features list — complete inventory

The chord-only script contains **102 feature columns**: 16 raw pitches and 86 calculated features. The chord-and-melody script contains **135 feature columns**: all 102 chord features, ten raw melody pitches, and 23 additional calculated features. These are feature counts, not counts of candidate threshold rules or independent musical effects.

Indices in script names start at zero: `c0`–`c3` mean chords 1–4, `v0`–`v3` mean the four stored note-position tracks, and `m0`–`m9` mean melody notes 1–10. `c0_gap0`–`c3_gap2` includes every combination of four chord indices and three gap indices. Voice tracks follow stored positions; their interpretation as persistent musical voices depends on how the input was encoded.

Pitch differences below are in the input’s numerical pitch units. The scripts use modulo 12 for octave equivalence. A folded interval class is `min(d mod 12, 12 − (d mod 12))` for an absolute pitch difference `d`; its numerical size is not a direct measure of consonance or harshness. Histogram-based pitch classes and melody–harmony fit use integer-converted modulo-12 values, which can discard fractional pitch information.

**A. Chord features — used in both analyses (102 columns)**

| Feature | Script name(s) | Count | Definition |
|---|---|---|---|
| Raw chord pitches | `n0`–`n15` | 16 | Pitch at each of the 16 stored note positions: four successive groups of four notes. |
| Overall register | `mean_all`, `range_all`, `std_all` | 3 | Mean, maximum minus minimum, and population standard deviation of the 16 chord pitches. |
| Within-chord gaps | `c0_gap0`–`c3_gap2` | 12 | For each chord, the three adjacent gaps after sorting its four pitches from low to high. |
| Chord span | `c0_span`–`c3_span` | 4 | Highest minus lowest pitch in each chord. |
| Chord mean | `c0_mean`–`c3_mean` | 4 | Mean pitch of each chord. |
| Chord spread | `c0_std`–`c3_std` | 4 | Population standard deviation of pitches in each chord. |
| Between-chord movement | `move0`, `move1`, `move2` | 3 | Signed change in mean chord pitch for transitions 1→2, 2→3, and 3→4. |
| Overall chord drift | `total_drift` | 1 | Final chord mean minus first chord mean. |
| Total chord movement | `abs_motion` | 1 | Sum of absolute changes in mean chord pitch across the three transitions. |
| Rising chord transitions | `contour_up` | 1 | Number of transitions with an increase in mean chord pitch. |
| Pitch-class span | `c0_pcspread`–`c3_pcspread` | 4 | Maximum minus minimum of pitches modulo 12 within each chord. This is a linear span of remainders, not the shortest circular span. |
| Smallest interval class | `c0_icmin`–`c3_icmin` | 4 | Minimum folded interval class among the six note pairs in each chord. |
| Largest interval class | `c0_icmax`–`c3_icmax` | 4 | Maximum folded interval class among the six note pairs in each chord. |
| Mean interval class | `c0_icmean`–`c3_icmean` | 4 | Mean folded interval class among the six note pairs in each chord. |
| Shared exact pitches | `common0`, `common1`, `common2` | 3 | Number of distinct exact pitches shared by each pair of adjacent chords. |
| Shared pitch classes | `commonpc0`, `commonpc1`, `commonpc2` | 3 | Number of distinct pitches modulo 12 shared by each pair of adjacent chords. |
| Movement per stored voice | `v0_leap`–`v3_leap` | 4 | Sum of absolute pitch changes along each fixed note-position track across the four chords. |
| Drift per stored voice | `v0_drift`–`v3_drift` | 4 | Last minus first pitch along each fixed note-position track. |
| Largest voice jump | `max_voice_leap` | 1 | Largest absolute pitch change among all voices and transitions. |
| Combined voice movement | `voice_leap_total` | 1 | Sum of absolute pitch changes over all four voices and three transitions. |
| Same-direction movement | `parallel_moves` | 1 | Number of transitions where all four stored voices strictly rise or all strictly fall. |
| Opposite-direction movement | `contrary_moves` | 1 | Number of transitions with at least one rising and one falling stored voice. |
| Change in voice ordering | `voice_crossings` | 1 | Number of transitions where the sorted order of stored note positions changes; ties can affect this proxy. |
| Close note pairs, ≤1 | `clash_le1` | 1 | Count of within-chord note pairs with absolute pitch difference ≤1, including unisons, summed over all four chords. |
| Close note pairs, ≤2 | `clash_le2` | 1 | Count of within-chord note pairs with absolute pitch difference ≤2, including unisons, summed over all four chords. |
| Smallest pitch gap | `min_interval` | 1 | Smallest absolute within-chord pairwise pitch difference anywhere in the progression. |
| Interval-class-one pairs | `clash_ic1` | 1 | Count of within-chord note pairs whose folded interval class equals 1. |
| Bass movement | `bass_motion` | 1 | Total absolute movement of the lowest pitch of each chord. |
| Bass drift | `bass_drift` | 1 | Lowest pitch of the final chord minus lowest pitch of the first chord. |
| Bass range | `bass_range` | 1 | Maximum minus minimum of the four chord bass pitches. |
| Chord-top movement | `mel_motion` | 1 | Total absolute movement of the highest pitch of each chord. |
| Chord-top drift | `mel_drift` | 1 | Highest pitch of the final chord minus highest pitch of the first chord. |
| Chord-top range | `mel_range` | 1 | Maximum minus minimum of the four chord-top pitches. |
| Rising chord tops | `mel_up` | 1 | Number of transitions where the highest chord pitch rises. |
| Pitch-class variety | `pc_distinct` | 1 | Number of occupied pitch-class bins across the 16 chord notes. |
| Pitch-class entropy | `pc_entropy` | 1 | Shannon entropy in bits of the chord pitch-class histogram; larger values indicate more evenly distributed occupancy. |
| Most frequent pitch class | `pc_max` | 1 | Number of notes in the most populated chord pitch-class bin. |
| Transposed adjacent chords | `transpositions` | 1 | Number of adjacent chord pairs whose sorted pitches differ by a constant shift, using tolerance 10⁻⁶ on the standard deviation of the shifts. Zero-shift repeats also count. |
| Repeated adjacent chords | `repeats` | 1 | Number of adjacent chord pairs whose sorted pitches have summed absolute difference below 10⁻⁶. |
| First–third chord repetition | `chord0_eq_2` | 1 | Binary indicator that chords 1 and 3 have sorted pitches with summed absolute difference below 10⁻⁶. |
| Odd-valued note count | `odd_notes` | 1 | Count of chord pitches that become odd integers after integer conversion. Musical tuning interpretation depends on the pitch encoding. |

**B. Melody and melody–harmony features — added in the combined analysis (33 columns)**

| Feature | Script name(s) | Count | Definition |
|---|---|---|---|
| Raw melody pitches | `m0`–`m9` | 10 | The ten melody pitches in time order. |
| Total movement | `melo_motion` | 1 | Sum of the absolute differences between consecutive melody pitches. |
| Net drift | `melo_drift` | 1 | Last melody pitch minus first melody pitch. |
| Pitch range | `melo_range` | 1 | Highest minus lowest melody pitch. |
| Largest leap | `melo_maxleap` | 1 | Largest absolute difference between consecutive melody pitches. |
| Mean pitch | `melo_mean` | 1 | Mean of the ten melody pitches. |
| Pitch spread | `melo_std` | 1 | Population standard deviation of the ten melody pitches. |
| Starting pitch | `melo_start` | 1 | First melody pitch; duplicates `m0`. |
| Ending pitch | `melo_end` | 1 | Last melody pitch; duplicates `m9`. |
| Highest pitch | `melo_high` | 1 | Maximum melody pitch. |
| Lowest pitch | `melo_low` | 1 | Minimum melody pitch. |
| Upward movements | `melo_up` | 1 | Number of strictly positive consecutive pitch differences. |
| Small movements | `melo_steps_le2` | 1 | Number of consecutive absolute pitch differences ≤2; repeated notes are included. |
| Large movements | `melo_leaps_ge5` | 1 | Number of consecutive absolute pitch differences ≥5. |
| Repeated notes | `melo_repeat` | 1 | Number of consecutive pitch differences equal to zero. |
| Changes in movement sign | `melo_turns` | 1 | Number of changes between consecutive movement signs. Moving into or out of a repeated note also counts, so this is broader than strict up/down reversals. |
| Pitch-class variety | `melo_distinct_pc` | 1 | Number of occupied pitch-class bins among the melody notes. |
| Pitch-class entropy | `melo_pc_entropy` | 1 | Shannon entropy in bits of the melody pitch-class histogram. |
| Mean distance to harmony | `melo_fit_mean` | 1 | Mean, over melody notes, of the nearest folded interval-class distance to any chord pitch class anywhere in the progression. |
| Largest distance to harmony | `melo_fit_max` | 1 | Maximum of those nearest distances. |
| Notes outside chord pitch classes | `melo_nonchord` | 1 | Number of melody notes with nearest distance ≥1 to the pooled chord pitch classes. |
| Notes at distance one | `melo_clash1` | 1 | Number of melody notes with nearest distance exactly 1 to the pooled chord pitch classes. |
| Clearance above all chords | `melo_above_gap` | 1 | Lowest melody pitch minus the highest pitch anywhere in the chords. |
| Mean height above chord tops | `melo_vs_top_mean` | 1 | Mean melody pitch minus the mean of the four chord-top pitches. |

The `mel_` features describe the highest notes of the four chords; the `melo_` features describe the separate ten-note melody. The feature list includes redundant measurements, such as `melo_start` and `m0`; it should not be interpreted as 135 independent sources of evidence. Constant features or rules with too few items on either side may not appear in the candidate catalogue.

### The candidate catalogue

In [discover_melody.py](Music-Pattern-Analysis/discover_melody.py), `dump_candidates` uses the discovery data to:

1. Calculate each feature’s median.
2. Form two rules: `feature <= median` and `feature > median`.
3. Keep rules with at least 20 discovery items on each side.
4. Record the rule, feature, operator, threshold, number matching, mean preference, overall mean, shift from that mean, category proportions, direction, and Spearman correlation.
5. Sort rules by the absolute difference between the subgroup mean and the overall discovery mean.

The saved [candidate_patterns.csv](Music-Pattern-Analysis/results/candidate_patterns.csv) is therefore a catalogue of eligible **single-feature median rules**. It is not an exhaustive list of every threshold or combination searched. Its LIKE/DISLIKE labels are descriptive: the script uses a mean shift above +0.03 or below −0.03 to assign them, not a significance test.

### Selection for confirmation

The shared `phase1` function in [discover.py](Music-Pattern-Analysis/discover.py) builds a separate shortlist:

- **Feature trends:** take the eight strongest absolute Spearman correlations and form median rules pointing toward higher preference.
- **Subgroups:** search thresholds at the discovery 33rd and 66th percentiles, requiring at least 30 items on each side. Score a rule as `fraction matching × (subgroup mean − overall mean)`. Search two-feature AND rules among the strongest singles and nominate up to eight subgroup rules.
- **Deduplication:** remove repeated condition signatures before confirmation. Different features can still describe equivalent or highly overlapping groups.

The tree models and LIKE-versus-DISLIKE contrasts print exploratory diagnostics; the current shared implementation does not automatically add their rules to the confirmation shortlist.

For a complete audit trail, an expanded catalogue should also record every searched subgroup, exact unrounded thresholds, selection source, and whether each rule was shortlisted, tested, or skipped for insufficient support. Keep unsuccessful tests in the results. The current confirmed-pattern CSV files contain both successful and unsuccessful confirmation tests despite their filenames.

## 2. How to test chord and melody patterns

### Separate discovery from confirmation

The standard scripts split items into 60% discovery and 40% confirmation, stratified by preference. Thresholds and candidate selection use discovery data only. Confirmation applies those frozen rules to the remaining items, requiring at least 15 matching and 15 nonmatching items.

All presentations of the same stimulus must stay in the same partition. A confirmation set must also be unused in earlier decisions about the analysis: an existing split cannot provide fresh confirmation after its results have guided further pattern selection. The reshuffled analysis uses 99 discovery items and the remaining 701 parent items; its confirmation interpretation depends on that same requirement.

### Preference scale and test statistic

The scripts code dislike as −1, not sure as 0, and like as +1. For each pattern, calculate:

`T = mean preference among matching items − mean preference among nonmatching items`.

A positive value indicates higher preference among matching items. This mean-based statistic assigns equal numerical spacing to the three responses. Treating “not sure” as intermediate is a substantive assumption: it may instead express uncertainty. Report category proportions and use an ordinal or categorical sensitivity analysis if that distinction matters. An exploratory projection placing “not sure” between the other groups does not establish the assumption.

### Permutation confirmation

Keep pattern membership fixed, shuffle confirmation preferences across independent items, and recalculate `T`. The standard scripts use 3,000 shuffles and a two-sided p-value:

`p = (1 + number of shuffled |T| values at least as large as observed |T|) / (3,000 + 1)`.

This tests whether the observed association is unusual under the null in which labels are exchangeable across items. It avoids a normality assumption, but still requires a valid shuffling scheme. A p-value is not the probability that the pattern is false. See the [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

### Multiple tests and effect sizes

The scripts apply Benjamini–Hochberg correction at 5% across the tested shortlist. Under its independence or suitable positive-dependence assumptions, this controls the expected proportion of false discoveries among rejected hypotheses. Highly overlapping rules require attention to dependence; Benjamini–Yekutieli offers a more conservative correction for general dependence. See the [statsmodels FDR documentation](https://www.statsmodels.org/v0.12.2/generated/statsmodels.stats.multitest.fdrcorrection.html).

Report the mean shift and **Cliff’s delta**: the probability that a randomly chosen matching item has a higher rating than a nonmatching item, minus the probability it has a lower rating. Ties contribute zero. The standard scripts bootstrap confirmation items 1,000 times to estimate 95% intervals for delta among FDR survivors. An interval excluding zero supports a consistent direction; it does not mean every bootstrap sample has that sign. These are individual intervals, not simultaneous or selection-adjusted intervals.

A results table should include the exact rule, counts on both sides, category proportions, mean shift, delta and interval, raw p-value, adjusted p-value, and verdict. The current scripts report an FDR pass/fail flag rather than adjusted p-values. Failure to pass means insufficient evidence under this procedure, not proof of no effect.

### Chords, melody, and their combination

Run chord-only and chord-and-melody analyses against the preference labels for the corresponding listening condition. A melody feature associated with ratings of the combined stimulus does not isolate a causal melody effect.

To assess whether melody adds predictive information, compare a chord-feature model with a chord-plus-melody model on identical held-out items and labels, with tuning confined to training data. To test whether two features interact, compare a model containing their main effects with one additionally containing their product. A successful AND rule alone does not establish interaction, and a nonsignificant interaction does not prove additivity.

Define the multiple-testing family before testing. The current scripts correct each analysis separately; a single combined claim across chord and melody candidates should account for all relevant tests together. Direct comparisons of listening conditions should preserve pairing when the same underlying progressions are rated in both conditions.

## 3. How to consider preferences collected multiple times

### Preserve each observation

Store repeated ratings in a long table with one row per presentation:

| stimulus_id | condition | listener_id | session_id | presentation_order | preference |
|---|---|---|---|---|---|
| A | chords | L1 | 1 | 12 | like |
| A | chords | L1 | 2 | 37 | like |
| A | chords | L1 | 3 | 8 | not sure |

Retain timestamps and collection provenance when available. Identify a stimulus by its actual musical content and condition, rather than assuming matching row positions imply matching items. Missing responses should remain missing, not become “not sure.”

### When the preferences are currently identical

In [preference_by_chord.csv](Data/preference_by_chord.csv), the five preference columns agree for all 99 items. This is perfect **recorded agreement**, but identical values alone cannot establish whether answers were copied or independently recollected.

- **Copied or unverified repetitions:** use one preference per item for the primary association analysis. Preserve the five columns for auditing, but do not count 495 entries as 495 independent musical items or interpret their agreement as measured test–retest reliability.
- **Verified fresh responses:** retain every observation and report the observed agreement. Even perfectly agreeing repeated responses remain clustered within the same item and listener. They supply information about observed consistency, while the number of distinct musical items remains 99.

Thus, repeated preferences can be represented now even when their values coincide. Their provenance determines what conclusions they support. If all items also share one preference category, there is no between-item preference variation for these association tests.

### Analysis once independent re-ratings are available

A simple primary analysis can average the coded ratings within each item and give each item equal weight. This estimates average recorded preference across its observed sessions; report response counts and category proportions because averaging hides disagreement. Bootstrap whole items, and shuffle whole rating vectors between exchangeable items rather than shuffling individual presentations. Session imbalance or missingness may require a more specific model or restricted shuffling design.

For an analysis that retains all responses, use an **ordinal mixed model**: a model for ordered categories that accounts for repeated observations of each item. Include the candidate pattern, session, and presentation order as predictors, with an item random intercept; add a listener random intercept when there are multiple listeners. With one listener, the conclusions concern that listener. Cumulative link mixed models are available through [the ordinal package](https://stat.ethz.ch/CRAN/web/packages/ordinal/refman/ordinal.html); check model assumptions and fit before interpreting coefficients.

Estimate session-specific pattern effects and a pattern-by-session interaction if the question is whether preferences change over time. Correct across the planned pattern and interaction tests. Report agreement and category transitions separately from musical associations: consistency of repeated answers and predictability from musical features answer different questions. Similarity between different progressions is not a direct measure of test–retest reliability or a validated prediction ceiling.

**Practical rule:** select patterns on discovery items, test frozen rules on unused items, and keep repeated observations together throughout splitting and uncertainty estimation.
