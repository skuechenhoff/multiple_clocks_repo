# CHANGELOG

## 2026-09-16 (b) — I1: ripple-locked cortical HFB. Medial frontal yes, visual null; survives the pad sweep

**New:** `scripts/swr_ripple_locked_hfb.py`. **Results:**
`.../group/swr/ripple_locked_hfb_2026-09-16/`. Runs from `bundle_v2`, no cluster.

61 sessions, 42 subjects, 505 cortical derivations, ~740k ripple-alignments.
He et al. windows: epoch ±750 ms, **peri = ±250 ms**, **non-peri = (−750:−250) ∪
(+250:+750)**. Estimate = peri − non-peri (each ripple against its own flanks),
then **minus a shifted null**. Both windows must be artifact-free in the
hippocampal *and* the cortical derivation.

### Result — different-shaft derivations only

| ROI | derivations | subjects | effect (z) | t | p |
|---|---|---|---|---|---|
| **mOFC** | 100 | 32 | **+0.0083** | 4.71 | **<0.0001** |
| **mPFC** | 51 | 30 | **+0.0040** | 4.38 | **0.0001** |
| Lat. temporal | 129 | 35 | +0.0028 | 3.70 | 0.0008 |
| Auditory | 38 | 18 | +0.0016 | 1.44 | 0.17 |
| Visual | 97 | 20 | +0.0011 | 0.51 | 0.62 |

mOFC > mPFC (+0.0050, p = 0.016) and > lateral temporal (+0.0062, p = 0.015).
Sharp transient peaking within ~100 ms of the ripple peak, back to baseline by
±250 ms. **Replicates He et al.'s medial-frontal-yes / visual-no dissociation**,
with medial OFC the stronger of the two medial frontal regions.

### It passes the pad-stability check that the ripple-RSA effect failed

Same standard as the entry below. 0.75 → 3.0 s takes the usable set from
169,697 to 67,353 alignments (−60%):

| ROI | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---|---|---|---|---|
| mOFC | +0.0079 | +0.0078 | +0.0086 | +0.0079 | +0.0060 |
| mPFC | +0.0040 | +0.0039 | +0.0033 | +0.0025 | +0.0046 |
| Lat. temporal | +0.0029 | +0.0027 | +0.0024 | +0.0025 | +0.0032 |
| Visual | +0.0011 | +0.0008 | −0.0001 | −0.0016 | −0.0015 |

Flat-to-declining, not growing — the healthy direction. **mOFC is robust
(p < 0.02 throughout); mPFC is the wobblier one**, dipping to p = 0.080 at pad
2.0 s before recovering. Auditory drifts up at the largest pads (p = 0.052,
0.072) without ever reaching significance, which at that n reads as noise.

⚠ **The first version of this sweep was vacuous and looked reassuring.** It ran
0.10/0.25/0.50/1.00 and gave a perfectly flat line — because a ±750 ms epoch
must be artifact-free, so **every event nearer than 750 ms to a crossing is
already excluded at any detection pad** (26.7% of the bundle). Three of those
four pads were therefore the identical subset. The sweep now starts at the
epoch half-width, where the comparison is real. A flat sweep is only evidence
if the subsets actually differ.

### Three controls, each of which changed the answer

1. **The reference must be a SHIFTED NULL, not zero.** Against zero *every*
   region is positive, controls included (Visual p = 0.044, Auditory p = 0.044)
   — ripples cluster in states (F1) and cortical HFB tracks state too. Shifting
   ripple times 5–120 s leaves ~0 everywhere. This is what makes the controls
   null.
2. **Same-shaft derivations are volume-conduction contaminated**, and the
   2026-09-13 note recommending them as "the tightest control" was wrong about
   which confound they control. The *entire* Visual effect came from 9
   derivations on the hippocampal electrode (+0.0115, p = 0.039) against
   +0.0011 and null on the other 97. Same-shaft is a good control for recording
   quality, a bad one for anatomical specificity. Primary set is different-shaft.
3. **Figure aggregation must match the test** (found by SK). The time-course
   panel averaged over derivations while the bars averaged over subjects; for
   Visual those differ fourfold (+0.0044 vs +0.0011) because its coverage is
   concentrated — up to 14 derivations in one subject. The figure showed Visual
   responding and mPFC not, the opposite of the statistics beside it. Traces are
   now stored flat with an index and aggregated subject-level.

### Status

**Exploratory.** All three controls were chosen after seeing the naive result,
so they are justified on mechanism rather than by prior declaration. No
confirmatory run on held-out sessions. Effect sizes are small in absolute terms
(0.003–0.008 z) and rest on very large alignment counts — the dissociation and
the time-course shape are the evidence, not the p-values.

⚠ The non-peri flanks are the baseline, so a response broader than ±250 ms
would leak into its own baseline and be under-estimated. The measured responses
return to ~0 by ±250 ms, so this does not bite here.


## 2026-09-16 (b) — Time-resolved, descriptive: what the pad is actually changing

**New:** `scripts/swr_ripple_rsa_timeresolved.py`.
**Results:** `.../group/swr/ripple_rsa_timeresolved_2026-09-16/` (no tests, no
p values anywhere in this entry's figures).

### The pad is a UNIFORM ~30 % thinning

Spearman(lag from uncover press, `dist_to_artifact_s`) = **+0.005, p = 0.69**.
At pad 1.0 s, ~30 % of ripples are removed in EVERY lag bin (0-0.5 s: 29.4 %;
0.5-1 s: 31.4 %; 1-2 s: 29.5 %; 2-4 s: 34.7 %; 4-8 s: 33.5 %). The pad is not
preferentially deleting press-locked events, so "the pad removes the
interesting ripples" is ruled out.

### Therefore the pad-dependence is resampling noise — shown directly

At pad 0.1 (window 0-1 s, 1254 ripples), randomly dropping 33 % of ripples
300 times:

| ROI | full-data rho | range under random thinning | SD |
|---|---|---|---|
| **mPFC D** | **+0.104** | **-0.386 to +0.679** | 0.200 |
| PCC D | +0.056 | -0.486 to +0.641 | 0.172 |
| HC_anterior D | -0.097 | -0.464 to +0.426 | 0.134 |

**2.7 %** of random thinnings reach the pad-1.0 value of +0.563 or more.
Dropping a random third of the data moves rho by up to ±0.4, so the gap between
+0.10 and +0.56 carries essentially no information. Full write-up in
`random_thinning_result.md`.

### Time-resolved view (uncover D, `known_set`, pads 0.25 and 1.0)

CUMULATIVE (all ripples from the press to t) and SLIDING (1.5 s window stepped
through), both capped at each event's own interval end so nothing leaks into
the next reward's condition.

- **Cumulative curves start extreme and decay toward 0 as ripples accumulate.**
  At pad 1.0, mPFC begins at +0.59 with ~50 ripples and falls monotonically to
  ~-0.05 by t = 4 s (~570 ripples). At pad 0.25 the same curve begins at +0.34.
  The published-looking value is the left edge of a decaying curve, i.e. the
  sparsest point.
- **Sliding curves oscillate around 0 in every ROI with no sustained window.**
  If a subset of ripples carried the plan, a sliding window should find a
  stretch where the fit is consistently positive. None exists.
- Ripple counts: cumulative reaches ~870 (pad 0.25) / ~640 (pad 1.0) by 8 s;
  sliding peaks at ~420 / ~300 around 1.5 s.
- **Firing rate inside ripples is genuinely event-locked**: mOFC falls from
  ~4.1 Hz at the press to ~1.8 Hz by 2 s; HC_mid shows the opposite hump. So
  the data do contain real uncover-locked dynamics — they just do not carry
  configuration structure.

### Anatomy: mPFC and mOFC do NOT overlap here

SK asked whether the "mPFC" entry zone might sit in dorsal mOFC. It does not:
mPFC cells sit at MNI z ~ -5 to +13 (mode ~0 to +5), mOFC at z ~ -25 to -12,
with a clean gap between. Caveat: the 65 mPFC cells come from only ~8 distinct
microwire locations (cells cluster on bundles), so this is 8 sites, not 65
independent ones — which is itself part of why the estimate is unstable.

## 2026-09-16 — Artifact-pad sweep on the swr_v2 bundle: the mPFC effect runs backwards

**New:** `scripts/swr_ripple_rsa_pad_sweep.py`; `pad_s` filtering and
`default_bundle_dir()` in `mc/analyse/ripple_rsa.py`.
**Results:** `.../group/swr/ripple_rsa_pad_sweep_2026-09-16/`.

⚠ **Path change:** the old `group/swr/bundle/` is now `bundle_08.09.2026/` and
the new one is `bundle_v2/`. Hardcoded `.../swr/bundle` paths are stale;
`rrsa.default_bundle_dir()` now resolves this.

### The new bundle

`swr_v2` detects at a **0.1 s** artifact pad and carries `dist_to_artifact_s`,
so larger pads are nested subsets. 100,737 events vs 64,760 in swr_v1 (1.56x);
**pad 1.0 s reproduces swr_v1 almost exactly** (64,895 vs 64,760), so that is
the pad the old results were computed at.

`meta.json` warns that re-padding also needs exposure rebuilt from
`artifact_intervals`. **That caveat does not apply here**: this analysis
measures firing DURING ripples and never divides by artifact-free seconds, so
filtering events is sufficient and complete.

Still absent from the export: `t_start_s` / `t_end_s`. The `peak ± duration/2`
approximation therefore stands — worth adding on the next export, it costs
nothing (the columns already exist per-session).

### The sweep: 5 pads x 2 schemes x 5 ROIs x 4 states x 2 models = 332 cells

**The pre-declared cell, mPFC at D, window scheme — it grows as ripples are
REMOVED:**

| pad (s) | ripples at D | pairs missing | rho | p |
|---|---|---|---|---|
| **0.10** (most data) | **311** | 0 | **+0.104** | **0.300** |
| 0.25 | 290 | 0 | +0.266 | 0.101 |
| 0.50 | 257 | 2 | +0.440 | 0.025 |
| 1.00 (old bundle) | 209 | 4 | **+0.563** | 0.017 |
| 2.00 | 139 | — | not estimable | — |

Monotonic, and backwards. A real effect gets clearer with more data; this one
is largest where the data are thinnest and the RDM is least complete. At the
native pad, with 50% more ripples and a complete RDM, it is rho = +0.10,
p = 0.30. The interval scheme shows no mPFC D effect at any pad
(-0.084 to -0.180).

### The whole sweep produces fewer hits than chance

**11 of 332 cells reach p < 0.05 uncorrected = 3.3%**, against the ~5% expected
under the global null. The hits do not concentrate in any ROI, state, model or
pad: mOFC B at pads 0.10/0.25, HC_mid D at 0.25, mPFC D at 0.50/1.00, mPFC B at
0.50/2.00, HC_mid A at 2.00. That is the signature of no effect.

**The pad is not a free parameter.** Choosing 1.0 s because the effect is
biggest there is exactly the post-hoc gate this project avoids; the sweep is a
stability check, and this effect fails it.

### Null width does not improve with more ripples

Median null SD across the sweep: 0.200-0.222, against the 28-pair floor of
1/sqrt(27) = 0.192. Going from 2,610 to 5,866 ripples (interval scheme) moves
it from 0.199 to 0.203 — i.e. not at all. **The binding constraint is the 8 x 8
RDM, not the ripple count.** More ripples cannot fix this design; more
conditions could.

### Where this leaves the analysis

The entry (f) mPFC result is now contradicted from two independent directions:
loosening the ripple-to-condition rule removes it (entry g), and adding ripples
at a smaller artifact pad removes it (here). It should not be carried forward.

## 2026-09-15 (g) — Interval scheme (supervisor's suggestion), and the mPFC D effect does not survive it

**New:** `ripples_in_intervals` + `SCHEMES` in `mc/analyse/ripple_rsa.py`;
`scripts/swr_ripple_rsa_diagnostics.py`.
**Results:** `.../group/swr/ripple_rsa_diagnostics_2026-09-15/`.
The `window` scheme is kept and still runnable — entry (f) is not superseded,
it is contextualised.

### The interval scheme: 4.7x the data, complete RDMs

Every ripple in [uncover_k, uncover_{k+1}) is assigned to state k; state D runs
to the NEXT repeat's t_A. Three advantages: the intervals tile the first
traversal exactly once so **no ripple can enter two conditions**; the knowledge
state is **constant** throughout each interval, which is exactly what the
knowledge-gated model describes; and coverage jumps.

| scheme | ripples | per (config x state) | RDM pairs missing |
|---|---|---|---|
| post 0–1 s window | 885 | 28 | 0–23 depending on ROI |
| inter-uncover interval | 4040 | **126** | **0 everywhere** |

Ripples per state (interval): A 1011, B 1179, C 1132, **D 718** — D is the
thinnest because the interval to the next repeat's t_A is short (median 3.1 s).

### ⚠ The mPFC D effect does not survive

| ROI / state | post 0–1 s window | whole interval |
|---|---|---|
| **mPFC D** | **+0.325** (p = 0.059) | **-0.135** (p = 0.726) |
| PCC D | +0.248 (p = 0.129) | +0.200 (p = 0.162) |
| HC_anterior D | -0.328 | -0.220 |

### The diagnostic that settles it: early vs late within the interval

If the effect were genuinely time-locked to the discovery, it should live in the
first second and be ~0 afterwards. Splitting the SAME intervals:

| ROI / state | early 0–1 s (885) | late 1 s–end (3155) | whole (4040) |
|---|---|---|---|
| **mPFC D** | **+0.325** | **-0.445** | -0.135 |
| PCC D | +0.248 | +0.342 | +0.200 |
| HC_anterior D | -0.328 | -0.274 | -0.220 |

**mPFC flips sign**, and the late half is as extreme negative as the early half
is positive — both ~1.6-2.2 SD of a null whose SD is 0.2. That is what a coin
flip looks like across 20 cells, not a time-locked effect. By contrast **PCC is
the only ROI positive in every subset** (+0.248 / +0.342 / +0.200), though it
reaches p = 0.06 at best.

Conclusion: the entry (f) mPFC result was a window-specific fluctuation. It
should not be carried forward as a finding.

### The two nulls are nearly the same distribution

Plotted as histograms (`null_distributions.pdf`) rather than error bars: the
config-relabel null and the surrogate-window null have almost identical width
and centre. So the surrogate is NOT a tighter test — it answers a different
question (does the window have to be at a ripple) and cannot tell you whether
there is any configuration structure at all. **Keep both**: config relabel as
the statistical null, surrogate as the specificity control.

### Model support (`model_support.pdf`)

Configuration pairs sharing >= 1 reward location, by state: A **0/28**,
B 5/28, C 17/28, D 24/28. Rank offset of shared locations: **offset 0 never
occurs at any state** — which is why a strict order model has literally no data
and `position_locked` is flat. r(known_set, known_seq) = 1.00 / 0.97 / 0.92 at
B / C / D, so the two are not separable over 28 pairs.

### Runs per configuration

A session contributes a median of 3 grids per configuration
(`coverage_sweep.pdf` panel d). Currently all their ripples are pooled into one
pattern. Pooling is NOT biased for the off-diagonal — two different
configurations never share ripples, so their noise is independent — but a
cross-run RDM (odd vs even grid occurrences, as the DSR pipeline does with its
`across_only` scope) would additionally protect against slow drift shared by
configurations recorded close in time. Worth adding now that the interval
scheme makes each cell well populated. NOT yet implemented.

## 2026-09-15 (f) — Controls for the ripple RSA, and why the null is 0.19 wide

`scripts/swr_ripple_rsa_controls.py` + a fast cached path in
`mc/analyse/ripple_rsa.py` (`cache_ripple_rates`, `patterns_from_cache`,
`fit_rho`, `fit_pooled`) — verified to reproduce `collect_spike_patterns`
exactly, 82x faster, which is what makes 2000-draw permutations affordable.
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_controls_2026-09-15/`.

### The null width is arithmetic, not a pipeline quirk

An 8 x 8 RDM has **28 unique pairs**, so a chance Spearman correlation over it
has SD = 1/sqrt(27) = **0.19**. Measured null SDs: config relabel 0.207,
surrogate window 0.185, pooled-over-84-pairs 0.122 (predicted 0.110),
post-minus-pre 0.291 (a difference carries both variances). All four match
their arithmetic prediction, which is the check that the nulls are built
correctly.

Consequence: **with 28 pairs you need |rho| > ~0.37 for p < 0.05.** A rho of
0.3 that would be large in an RDM with hundreds of cells is 1.5 SD here. The
fix is more RDM cells, not more permutations.

### Four controls

| test | draws | what it asks |
|---|---|---|
| C1 config relabel | 2000 | is the fit about configuration identity? |
| C2 surrogate window | 200 | does the window have to be at a ripple? |
| C3 pooled B+C+D | 2000 | same question, 84 pairs instead of 28 |
| C4 post − pre | 1000 | is the fit specific to after the press? |

C2 is the specificity control: each ripple's window is moved to a random time
inside the SAME +-1 s press window of the SAME uncover event, keeping its
duration. Same events, configs, states, cells and window count — the only
thing removed is that the window sat on a ripple.

### Result: mPFC at D, and C2 separates it from PCC

| ROI | state | rho | C1 p | **C2 p (ripple-specific)** | C4 p |
|---|---|---|---|---|---|
| **mPFC** | **D** | **+0.325** | 0.058 | **0.040** | 0.107 |
| mOFC | D | +0.285 | 0.214 | 0.060 | — |
| PCC | D | +0.248 | 0.129 | **0.125** | 0.174 |
| HC_anterior | D | -0.328 | 0.952 | 0.960 | 0.954 |

mPFC at D is the pre-declared primary cell and is the only one to clear
p < 0.05 on any control. **The surrogate-window control is what separates it
from PCC**: mPFC's fit is specific to windows that sat on a ripple (z = +1.95),
PCC's is not (z = +1.11). Neither survives FDR across the 14 cells of the C2
family (mPFC p_fdr = 0.42), and C1 does not reach 0.05 (p = 0.058).

**The effect is D-specific, not an accumulation.** Pooling B+C+D (C3) gives
mPFC rho = +0.111 on 74 pairs, z = +0.93, p = 0.173 — the D effect is diluted
by B (+0.079) and C (-0.094). That argues against "working memory accumulating
with each reward" and for "the plan is assembled once the configuration is
complete", which is the fMRI claim. It also means the pooled test, which would
have had the tighter null, is the wrong test for this effect.

HC_anterior runs NEGATIVE throughout (D: -0.328; pooled -0.198, z = -1.70) —
configurations sharing more locations have LESS similar hippocampal patterns.
No account offered.

 2026-09-15 (e) — Ripple RSA on raw spikes, +-1 s windows, ripple-extent firing

Supersedes (c) and (d). `scripts/swr_ripple_rsa_plan.py` (binned-matrix
version) DELETED; replaced by `scripts/swr_ripple_rsa_spikes.py`. The
binned-data collection path was removed from `mc/analyse/ripple_rsa.py`.

**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_spikes_2026-09-15/`.

### What changed

- **Firing measured INSIDE the ripple**: `t_peak +- duration_s/2`, rate =
  spikes / window width, from raw `abcd_passed.mat` spike times.
- **Press windows widened to +-1 s** (was [-0.35, 0] / [+0.15, +0.70]). The
  narrow pre-window left a median of 8 ripples per (config x state) and no
  estimable RDM at all. Now: **715 pre / 885 post ripples**, median 22 / 28 per
  condition. Cost: the post window no longer sits only on the rate increase.
- **Silent cells dropped**: 145 (pre) / 132 (post) cells fire no spike in ANY
  ripple of that window and are removed — they are an all-zero column.
  **Individual zero counts are KEPT**: "silent in this ripple" is data, and
  dropping those would bias every rate upward.
- ROIs from `neurons_with_ROI_labels.csv` / `atlas_roi`.

### Positive control now PASSES for HC_anterior

Firing inside the ripple vs two equal-width windows offset by 250 ms either
side (equal width, to avoid the count-vs-window-length trap):

| ROI | sessions | units | change | t | p |
|---|---|---|---|---|---|
| **HC_anterior** | 26 | 171 | **+4.4%** | **2.16** | **0.041** |
| HC_mid | 19 | 145 | +2.9% | 1.05 | 0.310 |
| PCC | 6 | 51 | +2.4% | 1.27 | 0.259 |
| mPFC | 15 | 65 | +1.7% | -0.16 | 0.879 |
| mOFC | 10 | 74 | -0.4% | -0.40 | 0.702 |

Measuring the ripple's own extent is more sensitive than the fixed 0-200 ms
post-peak window used in (c), which gave +2.2%, p = 0.104 for pooled HC.

### ⚠ SIGN CORRECTION — this inverts the conclusions of the first version

The first version of this entry reported "negative rho = predicted". **That was
wrong.** Model and data RDMs are both DISSIMILARITIES, so a region that encodes
the model produces a POSITIVE correlation: two configurations sharing few
locations are far apart in the model AND far apart in the data. Verified by
simulation — patterns constructed so each cell fires for its own rewarded
location give **rho = +0.92** against `known_set`. All one-sided tests are
upper-tail. `fit_model` and the figure labels are fixed.

### known_set at D IS full_abcd

Verified `np.allclose` = True: once D is uncovered the subject knows the whole
configuration, so the knowledge-gated model and the fixed full-ABCD model
coincide exactly at state D. That is why their rho is identical there.
`known_seq` at D is NOT the same (rank-weighted, not a plain set).

The ordered "compare all A-rewards, all B-rewards, ..." model is
`position_locked`, and it is **constant at every state** (sd = 0.000) because
within a state all eight configurations have distinct locations. It cannot be
fitted. The testable model is the unordered set, as expected.

### Result: direction is right in mPFC, but nothing is significant

`known_set`, post-press, positive = predicted:

| ROI | state | rho | exact p (1-sided) | pipeline p | z vs pipeline null |
|---|---|---|---|---|---|
| **mPFC** | **D** | **+0.325** | **0.054** | 0.090 | +1.49 |
| PCC | D | +0.248 | 0.124 | 0.090 | +1.48 |
| mOFC | D | +0.285 | 0.204 | 0.155 | +0.84 |
| HC_mid | B | (full_abcd) +0.373 | 0.013 | — | — |
| HC_anterior | D | -0.328 | 0.941 | 0.990 | -1.73 |

mPFC at D is the largest effect in the predicted direction and the pre-declared
primary cell. It does **not** reach significance under either null (0.054
exact, 0.090 pipeline) and does not survive FDR across the three primary ROIs
(0.162). **PCC at D matches it almost exactly (+0.248, p = 0.090)** and PCC is
not a predicted region — which is the main reason not to read the mPFC value as
a result yet.

HC_anterior at D goes the OTHER way (-0.328, z = -1.73): configurations sharing
more locations have LESS similar hippocampal patterns. No account is offered;
at a null SD of 0.2 this is what the tail looks like.

### THE PERMUTATION: how big is chance here?

`pipeline_null` draws a fresh per-session configuration relabelling and re-runs
the WHOLE estimator -- patterns, cell centring, missing-data structure, RDM,
fit -- 100 times. Each session gets its own relabelling, destroying the
cross-session config alignment that pooling cells depends on. That alignment is
the signal, so this is the right null, and it passes through identical code
(CLAUDE.md rule 4).

**The null SD is ~0.20 for every ROI and every state.** That is the number to
carry around: a Spearman rho of |0.2| in this design IS chance, and |0.3| is
1.5 SD. Values of that size appear all over the table, in both directions, in
regions with and without a prediction.

At 100 permutations the p resolution is 0.01, so prefer z to p.

### post - pre contrast: built a proper null for it, and it is not significant

A difference of two Spearman rhos has no standard sampling distribution, so
`pipeline_null_contrast` draws ONE per-session relabelling and applies it to
BOTH windows before differencing -- the same arithmetic as the real data, with
session composition, cell coverage and ripple counts preserved.

| ROI | state | post-pre | null SD | z | p |
|---|---|---|---|---|---|
| mPFC | B | +0.334 | 0.296 | +1.14 | 0.13 |
| mPFC | C | +0.320 | 0.317 | +1.05 | 0.17 |
| mPFC | D | +0.394 | 0.331 | +1.08 | 0.14 |
| PCC | D | +0.309 | 0.358 | +1.12 | 0.13 |
| HC_anterior | D | -0.501 | 0.262 | -1.88 | 0.98 |

mPFC is positive at all three testable states, which is a coherent pattern, but
the contrast null is WIDER than the single-window null (SD 0.26-0.36, because a
difference carries both variances) and nothing reaches significance. Note also
that the mPFC contrast is driven as much by a NEGATIVE pre fit (-0.255, -0.414,
-0.069) as by a positive post fit. Results in `contrast_permutation.csv`.

HC_anterior flips sign between windows in the opposite direction to mPFC
(pre +0.239 / +0.173 at C / D, post -0.180 / -0.328).

### Figures

- `model_RDMs.png/.pdf` — all four models x four states, with sd printed and
  the constant (unfittable) panels outlined in red. Makes the counterbalancing
  visible: `known_set`/`known_seq` are uniform at A, `position_locked` is
  uniform everywhere.
- `fit_timecourse_A_to_D.png/.pdf` — 3 models x {post, pre, post-pre}, all
  ROIs. Panels ~6 cm wide, Arial 9-11 pt, 2.2 pt lines, per CLAUDE.md.
- `overview_all_ROIs.png/.pdf` — (a) plan-model fit A->D for every ROI, post
  solid / pre dashed; (b) observed effect against its re-estimated null;
  (c) how complete each RDM is; (d) split-half reliability, restricted to
  states with >= 10 observed pairs.
- `rdm_reliability.png` — split-half reliability, so a flat timecourse is not
  misread as a null.
- `data_RDMs_<roi>.jpeg` — the measured RDMs.

## 2026-09-15 (d) — Ripple RSA power, done properly on RAW spike times

Redo of (c) after SK pointed out that the analysis must start from
`abcd_passed.mat`, not the 25 ms binned per-grid matrices, and that firing
should be measured DURING the ripple.

**New:** `scripts/swr_ripple_rsa_power.py`; raw-spike loaders in
`mc/analyse/ripple_rsa.py` (`load_spike_times`, `cell_roi_table`,
`spike_counts_in_windows`, `ripples_near_events`).
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_power_2026-09-15/`.

### Sources now used

- **Spikes:** `abcd_passed.mat` -> `abcd_data.neural_data(c).spikeTimes`, in
  seconds on the same session clock as behaviour and ripples
  (`save_iEEG_as_csv.m` bins these from 0 with no offset). Cached per session
  as .npz under `group/swr/spike_cache/`. 564 cells, 4.57 M spikes, 28 sessions.
- **ROIs:** `neurons_with_ROI_labels.csv`, column `atlas_roi`, joined
  positionally on `cell idx`. **Verified: that order matches the mat file's
  `electrodeLabel` for all 28/28 sessions.** This replaces
  `neurons_MNI_latest.csv`, which is not row-aligned for s27/s40/s50/s57/s60.

### Ripple onset/offset — no cluster work needed

`swr_detect.py` already computes `t_start_s`, `t_peak_s` and `t_end_s`, and all
three are present in the per-session `ripple_events.csv`. **Only the bundle
export drops them.** Re-exporting the bundle with those two columns is
sufficient; re-detection is not required. On s38 (the one session held locally,
461 accepted ripples, matching the bundle exactly) the peak sits essentially at
the centre of the event — median 23 ms before, 24 ms after, duration 59 ms — so
`peak +- duration/2` reconstructs the extent well in the meantime.

### THE POWER VERDICT: yes, still a problem, and it is arithmetic

Spikes per neuron per ripple, post-press window, 28 sessions:

| ROI | n cells | ±10 ms | ±duration/2 | ±100 ms | % zero (±dur/2) |
|---|---|---|---|---|---|
| HC_anterior | 171 | 0.043 | 0.185 | 0.521 | 87% |
| HC_mid | 145 | 0.072 | 0.278 | 0.787 | 83% |
| mPFC | 65 | 0.032 | 0.129 | 0.389 | 89% |
| mOFC | 74 | 0.115 | 0.319 | 0.810 | 78% |
| PCC | 51 | 0.039 | 0.148 | 0.471 | 88% |
| EC | 3 | 0.000 | 0.167 | 0.333 | 92% |

Firing rate inside ripples is 1.5-4.6 Hz depending on ROI. A ripple is ~60 ms.
**3 Hz x 60 ms = 0.18 spikes.** That product, not the analysis design, is the
constraint.

- **±10 ms is unusable**: 90-100% of (neuron, ripple) pairs contain no spike at
  all; EC contains none anywhere.
- Pooled over ALL 28 sessions, one (config x state) RDM cell is built from a
  median of **0.15 (mPFC) to 0.35 (HC_mid) spikes per neuron**. The RDM entry
  is therefore a comparison of near-empty count vectors.
- Nothing about this changes with the raw data — the binned matrices gave the
  same firing rates; what the raw data adds is the ability to ASK the question
  at ripple resolution, and the answer is that the question is not affordable
  in spikes.

Reaching ~5 spikes per neuron per condition would need roughly 20-30x more
ripples per condition. Available multipliers: all correct uncovers instead of
discoveries only (~7x, but the reward is then already known), pooling the four
states (4x, destroys the state axis), widening to ±100 ms (~2.8x, no longer
"during the ripple"). No combination preserves the design and the question.

**Conclusion unchanged from (c), now on the right data:** ripple-triggered
single-unit RSA is not affordable in this dataset. HFB remains the route —
`swr_extract_hfb` is continuous, has no spike-count floor, and covers every
derivation rather than only the sessions with microwires.

### ROI table disagreement — needs resolving

`atlas_roi` and `neurons_MNI_latest.csv` disagree substantially on the same 28
sessions: HC 316 (171 anterior + 145 mid) vs 236; **EC 3 vs 51**; PCC 51 vs 41;
mPFC 65 vs ACC 67; mOFC 74 vs OFC 68. The EC discrepancy is a factor of 17 and
changes whether EC is analysable at all. `atlas_roi` was used here because it
carries the project's canonical names and its row order is verifiable against
the mat file.

## 2026-09-15 (c) — Ripple-triggered RSA on human cells: UNDERPOWERED, not a null

The cell analogue of the instruction-phase fMRI RSA. Subjects discover A/B/C/D
rather than being shown them, so "how much of the plan is assembled" is indexed
by WHICH reward was just uncovered instead of by time within a 12 s instruction
period. Ripple-triggered firing at the discovery presses, 8x8 config RDM per
state, cells pooled across the 28 shared-config sessions.

**New:** `mc/analyse/ripple_rsa.py`, `scripts/swr_ripple_rsa_plan.py`.
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_plan_2026-09-15/`.

### What the task design forbids (verified, not assumed)

Within any single state all 8 configs have DISTINCT reward locations
(A: 3 8 1 4 6 9 7 2; B: 7 2 9 8 4 1 3 5; C: 9 6 5 1 2 3 4 7; D: 5 7 8 3 9 4 2 6).
Therefore:

- The position-locked Hamming model used in the fMRI (`rewDSR`) has **sd = 0.000
  at every state** and cannot be fitted in a within-state RDM. The
  generalisation that survives is set overlap of the locations known so far.
- At state A every knowledge-gated model is constant (one known location, all
  eight distinct). A is only testable with the fixed full-ABCD model, where it
  is the knowledge null.
- **Current location is constant within every within-state RDM**, so a
  within-state effect cannot be a place code. The counterbalancing supplies the
  confound control for free — stronger than the pre/post window.
- `known_set` vs `known_seq` correlate r = 1.00 / 0.97 / 0.92 at B / C / D.
  Not separable over 28 pairs; they are NOT run as a horse race. The sequence
  question is asked as a rank-offset test instead.

### Why this is underpowered, with numbers

- 2,665 discovery events (correct, `is_discovery`, explore) in the 28 sessions.
- Ripples in the windows: **265** in pre [-0.35, 0], **570** in post
  [+0.15, +0.70]. That is **0.48 / 0.64 ripples per (session x config x state)**
  — 64% / 56% of condition cells empty. Condition-averaging per session is
  impossible; only cross-session cell pooling makes any RDM estimable at all.
- **The whole `pre` window is unusable**: not one ROI/state yields a complete
  8x8 RDM, so the pre-vs-post control could not be run.
- In `post`, **only HC** gives complete RDMs (median 51-64 cells per pair).
  mPFC/ACC, OFC and EC all have missing pairs at every state, so the exact
  permutation is not defined for them and they are reported as NaN.
- **Split-half reliability of the best-covered RDM (HC, post, D) is
  rho = -0.06.** The RDMs are noise.

### Positive control: WEAK, and it caps everything downstream

Peri (0:+200 ms) vs non-peri (-750:-250 and +250:+750) firing, all ripples,
session-level paired test:

| ROI | sessions | units | mean change | t | p | sessions positive | sign test |
|---|---|---|---|---|---|---|---|
| HC | 25 | 236 | +2.2% | 1.69 | 0.104 | 19/25 | **0.015** |
| ACC (mPFC) | 16 | 67 | +1.0% | 1.08 | 0.296 | 8/16 | 1.00 |
| EC | 9 | 51 | +4.6% | 1.61 | 0.147 | 6/9 | 0.51 |
| OFC | 8 | 68 | +0.7% | 1.13 | 0.297 | 5/8 | 0.73 |

**This is not a clock problem.** A +/-10 s lag scan of HC firing against ripple
peaks maxes at exactly **0.0 s** (`clock_lag_scan.csv`), and the firing matrices
are raw unsmoothed 25 ms spike counts (integers 0-7, lag-1 autocorrelation
0.013). The coupling is genuinely ~2-4%.

**Nor is it probe co-location.** Unit electrode labels are `m` + probe name, so
microwire and ripple-derivation probes can be compared directly. Only **66 of
564 units (10 of 28 sessions)** sit on a probe that also produced ripples — but
restricting to the row-aligned sessions, same-probe units show **+2.9% (n=5
sessions)** and different-probe units **+3.7% (n=13)**. Co-location does not
rescue the coupling; an earlier apparent +6.6% for different-probe was driven
by sessions whose neuron table is not row-aligned (see below).

### Results (all secondary except the one marked)

Pre-declared primary: state D, post window, HC and mPFC, `known_set`.

| ROI | state | model | rho | p (1-sided) | p FDR |
|---|---|---|---|---|---|
| HC | D | known_set | +0.112 | 0.722 | 0.722 | **PRIMARY — null, wrong sign** |
| mPFC | D | known_set | — | — | — | RDM incomplete, not estimable |
| HC | B | known_set | -0.339 | 0.012 | 0.054 |
| HC | B | full_abcd | -0.322 | 0.042 | 0.125 |
| HC | C | known_set | -0.266 | 0.073 | 0.132 |
| HC | A | full_abcd | -0.117 | 0.297 | 0.404 |

The knowledge-accumulation prediction (full-ABCD fit grows A -> D) is **not**
borne out: -0.117, -0.322, -0.095, +0.112 across A/B/C/D. The B and C values sit
in the predicted direction but do not survive FDR, sit in a family of 40-odd
tests, and come from RDMs with ~zero split-half reliability.

`rank_offset` at HC post-D is rho = +0.534, p = 0.007 uncorrected — reported in
`controls.csv` and **not** believed, for the same reliability reason.

### Conclusion

**This is an underpowered design, not a negative result**, exactly as
`POTENTIAL_IDEAS.md` I5 warned ("if it is thin, I5 is not a null result, it is
an underpowered one, and those are different claims"). Do not re-run it as-is.
What would change the answer, in order of expected value:

1. More ripples per condition — the binding constraint. Either loosen detection
   (the `ripple` envelope is stored continuously and is threshold-free) or drop
   to a coarser condition space that keeps config variance.
2. HFB instead of spikes: `swr_extract_hfb` covers every derivation, so the
   feature space is not limited to the sessions with microwires.
3. Accept that mPFC cannot support a pooled-cell RDM here (67 cells, 45%
   coverage per condition) and go per-contact + LME, as He et al. do.

### Incidental finding, needs fixing elsewhere

`neurons_MNI_latest.csv` is **not row-aligned** with
`all_cells_region_labels_sub<N>.txt` for **s27, s40, s50, s57, s60** (38%, 95%,
44%, 81%, 63% of rows agree). Anything that joins the neuron table to firing
matrix rows by position is wrong for those five sessions. This analysis uses
`swu.unit_labels` (the row-aligned file) throughout and is unaffected, but the
mismatch should be resolved at source.

## 2026-09-15 (b) — Direction regressor + order-independent instruction models

Replaces the masking approach (see the correction below) with fitting the
forward/backward cue explicitly, and adds a new model family.

**`scripts/fMRI_run_RSA_instruction.py`**
- `DIRECTION_REGRESSOR = 'direction'` — reserved name, 1 = the cell crosses the
  cue. Built from condition labels (`build_direction_RDM_blocks`), has variance
  in both scopes.
- `UNORDERED_CHANNELS` — `AB/ABC/ABCD_rew_instr_unordered`, with
  `d = 1 - |multiset overlap of the first k instructed locations| / k`. Multiset,
  not set: a location revealed twice counts twice, which keeps the
  0/0.25/0.5/0.75/1 scale of the ordered models. Cut from the same rewDSR
  A_reward chunks the ordered models use.
- `model_family()` — three families (`execution`, `instruction`,
  `instruction_unordered`) plus `direction`, because the set models are
  instruction models that CAN be fitted across task halves. `single_model_scopes`
  now keys on it, accepts an exact model name as an override, and accepts `[]`
  to mean "only ever fit inside a combo".
- Degenerate designs are now DROPPED, not fatal. A regressor with no variance in
  a scope is a property of the subject's task assignment, not a bug; failing the
  run would block all other models for that subject. Recorded in
  `degenerate_maps` in the settings summary. Genuine rank deficiency (collinear
  but non-constant) still raises.

**`condition_files/rsa_instruction_direction_and_unordered.json`** — 37 maps.
Singles: `direction` (within+across) and the three set models (within+across).
Within combos: each ordered `*_rew_instr + direction`, each set model +
direction. Across combos: each set model + its ordered execution counterpart,
with and without direction. No singles for the ordered instruction or execution
models — those are already estimated.

**No k=1 set model.** With one location per condition order cannot matter, so
`A_rew_instr_unordered` is arithmetically identical to `A_rew_instr` — verified,
the whole 20x20 RDM matches at r = 1.0000. Omitted rather than run as a
duplicate. It would also have been constant across halves.

**Why direction is in the across combos.** Across halves `direction` is exactly
orthogonal to every set model (r = 0.0000, because each task appears once
forward and once backward per half). But it correlates about -0.55 with the
execution models there, so once execution is in the design the partial
correlation reappears: +0.029 (AB), **+0.372 (ABC)**, **+0.454 (ABCD)**.
Marginally orthogonal is not conditionally orthogonal; leaving it out would let
execution absorb direction-driven variance.

**k=4 caveat.** The full multiset is reversal-invariant, so
`ABCD_rew_instr_unordered` cannot distinguish instructed from executed ORDER —
read it as a "locations this task uses" model. It is NOT equal to `ABCD_rew`
(r ~ 0.18), which is why every across combo carries its execution counterpart.

**Design pre-flight (sub-02, 01-TR4 proxy):** all fitted designs well
conditioned — within-half condition numbers 1.6-2.3, across-half 1.8-2.9.

**`mc/fmri_analysis/check_RSA_ran.py`** — mirrors the three families and the
exact-name override, so `expected_map_names` resolves the new names.

**`mc/fmri_analysis/submit_RSA_instruction_epochs.sh`** — default config changed
to this one. It had been left pointing at `rsa_instruction_samedirection.json`
by commit c32824a, which would have run the execution model under an `_instr`
name; that config is now marked `_DEPRECATED` in its own body.

## 2026-09-15 — CORRECTION: the same-direction mask cannot test the instruction model

SK spotted this. Verified, and it is exact: **on same-direction cells,
`ABCD_rew_instr` and `ABCD_rew` are bit-identical regressors** — max|diff| =
0.000, r = 1.0000, for the F-F and the B-B cells separately.

Reason: the backward execution sequence is the exact reverse of the forward
one, and reversing BOTH members of a pair preserves position-wise Hamming
distance. So for a B-B pair, `Hamming(reverse(a), reverse(b)) ==
Hamming(a, b)`, which is the instruction value. For an F-F pair the two models
are trivially the same. Masking to same-direction cells therefore does not
produce a cleaner instruction model — it produces the **execution** model under
an `_instr` name, and fitting it returned exactly the same beta (+0.00682) for
both.

### The design entanglement this exposes

The within-half RDM contains only two kinds of cell, and each kills one model:

| cells | n | `ABCD_rew_instr` | `ABCD_rew` | tests |
|---|---:|---|---|---|
| same-direction (F-F, B-B) | 40 | varies | **identical to instr** | reward overlap, not instruction |
| direction-crossing (B-F) | 50 | varies | **constant at 1.0** | instruction, but every pair crosses the cue |

`ABCD_rew` is constant on every crossing cell because a forward and a backward
sequence never match at any of the four positions. So the
instruction-vs-execution contrast is identifiable **only** on direction-crossing
cells — which is exactly where the "please backwards" cue differs.

The confound is therefore not removable by any means tried, and this explains
the whole series: no control regressor works because the confound is collinear
with the hypothesis *by construction*, and masking does not work because it
deletes the contrast entirely. It is a design-level entanglement, not an
analysis choice.

### The one clean fit that remains

On **B-F cross-task cells (40)**, `ABCD_rew` is constant AND direction-crossing
is constant (every pair is one forward with one backward, both orderings
present), so neither can confound anything, and the forced-zero cells are
excluded. Only the instruction model varies there:

    ABCD_rew_instr on B-F cross-task cells = -0.05547

That is an unconfounded test of whether instructed reward-sequence overlap
predicts instruction-period pattern similarity across a direction switch, and
it is clearly negative.

### Status of (d) and (e), written yesterday

`condition_files/rsa_instruction_samedirection.json` and the
`within_same_direction` scope are **not withdrawn but must not be run or
reported under the `_instr` name** — the maps it writes are the execution model.
The scope machinery itself is correct and the runner/audit changes in (e)
(including the smoothing-config collision fix) stand on their own. Decision
pending: drop the config, or rename it to what it actually fits (a
reward-sequence-overlap model on the cells where instruction and execution
coincide).

The 2026-09-14 (c) conclusion — the negative is a forward/backward effect, not
a time effect — is unaffected and still holds.

## 2026-09-14 (e) — Cluster runner + audit adjusted for the same-direction config

**`mc/fmri_analysis/submit_RSA_instruction_epochs.sh`**
- base config is now `${RSA_CONFIG:-rsa_instruction_samedirection.json}`, so the
  same runner drives either instruction config.
- audit dir is `rsa_audit_${configTag}_<date>`, so the two configs do not share
  a todo list.
- default wall time 240 -> 120 min. 240 was sized for 28 models (21 single + 7
  combo) on 90 cells; the same-direction config fits 7 single models on 40
  cells and reuses the searchlight cache. Under-requesting is cheap because the
  RSA resumes. `FSLSUB_T=240` for the cumulative-rew config.

**`mc/fmri_analysis/check_RSA_ran.py`**
- `SCOPE_ALIASES` / `SCOPE_TAGS` gained `within_same_direction` ->
  `within-samedir`, mirroring the runner. Without this the audit KeyErrors on
  the new config.
- **Bug found and fixed while adjusting**: the per-epoch *smoothing* config
  snapshots were named `{smooth_config}_{glm}.json`, with no reference to
  `name_of_RSA` — unlike the RSA snapshots, which are keyed on the base config
  name. Auditing two RSA configs over the same epochs therefore overwrote each
  other's smoothing snapshots, and the surviving one carries a single
  `name_of_RSA`, so the smoothing jobs of one config would have been pointed at
  the other config's results folder. Now keyed on `name_of_RSA` too. Safe
  rename: only this script builds these names and they reach the smoothing job
  via `todo_smooth.txt`.

**Verified locally against the real sub-02 run:**
- `expected_map_names` gives the 7 `*_within-samedir` names, matching the files
  the RSA actually wrote; the cumulative-rew config still resolves to its 41.
- `settings_differences` vs the written `sub-02_settings_summary.json` is empty
  and no expected beta map is missing — the run is correctly marked DONE, so
  there is no permanent RERUN_CHANGED loop.

## 2026-09-14 (d) — New RSA scope `within_same_direction` + masked instruction config

Implements the conclusion of (c). Two changes, no new pipeline script.

**1. `scripts/fMRI_run_RSA_instruction.py`** gains a fourth scope,
`within_same_direction` (aliases `within_samedir`, `same_direction`): the
within-half cells that do NOT cross the forward/backward cue. Added
`same_direction_mask` / `same_direction_mask_2d` next to `within_half_mask`,
registered in `SCOPE_ALIASES` / `SCOPE_TAGS` / `SCOPE_CAPTION`, and wired into
`_scope_cells`, `_mask_2d_for_scope` and `all_scopes_used` so the model
regressor, the cached data RDM and the plotted RDM are all subset identically.

Verified on the real sub-02 condition ordering: 190 -> 90 within-half -> 40
same-direction cells, composition 20 F-F + 20 B-B, zero same-task pairs, zero
cross-half pairs, 2-D display mask agrees with the flat mask.

**2. `condition_files/rsa_instruction_samedirection.json`** — new config,
`name_of_RSA = instr_cumrew_samedir`. The 7 `*_rew_instr` single models only;
`run_combo_models: false`; no execution/planning models, which are unaffected
by the direction cue and stay with `rsa_instruction_cumulative_rew.json`.

Because the searchlight cache is keyed on `regression_version + TR +
searchlight_mask` and NOT on `name_of_RSA`, this reuses
`data_RDMs_glmbase_01-TR4_grey_matter` and recomputes **no searchlights**
(cache mtime confirmed unchanged after the run). Results go to
`RSA_instr_cumrew_samedir_glmbase_01-TR4/`, every map suffixed
`_within-samedir`, so nothing collides with earlier runs.

Run: `python scripts/fMRI_run_RSA_instruction.py <subj_no> rsa_instruction_samedirection.json`

**Validation.** Test run on sub-02 exited 0 with `n_cells_per_searchlight: 40`
and wrote beta/t/p for all 7 models. The pipeline's
`ABCD_rew_instr_within-samedir_beta` reproduces the independent audit
implementation bit-for-bit: mean +0.006784 vs +0.006784, r = 1.00000000,
max |diff| = 5.6e-16.

Still to check when this is run on the group: the F-F vs B-B split-half
reliability per subject (section 10) — in sub-02 it was ~0, and if that holds
across subjects the mask is clean but the data are thin.

## 2026-09-14 (c) — Why `ABCD_rew_instr` is negative: NOT time

Follow-up to the event-locked instruction RSA audit, which found a consistent
negative whole-brain shift for the instruction/memory model across subjects.
Hypothesis tested: the negative reflects **temporal** structure — the two
directions of a task being acquired far apart, so drift makes them look
dissimilar where the instruction model forces them identical.

Audit script has since been deleted (one-off diagnostic, not part of the
pipeline). Its outputs are kept at
`data/derivatives/group/instruction_instr_negative_time_audit_2026-09-14/`.
The conclusion is implemented as the `within_same_direction` RSA scope.

**Scope caveat.** The only locally cached full neural RDM is the older
`glmbase_01-TR4` instruction GLM, sub-02 only — not the 2026-09 event-locked
GLMs. The model side is identical to the event-locked pipeline
(`corr(ABCD_rew_instr, ABCD_rew) = 0.2311` reproduces the stored design audit
exactly), so only the neural side is a proxy. Needs rerunning where the
event-locked caches live.

### Reconstruction is exact

Recomputed sub-02 betas vs the saved `ABCD_rew_instr_within_beta.nii.gz`:
r = 1.0000000, max |diff| = 3.9e-16 over 126,321 searchlights. The cell-level
attribution below is therefore the pipeline's own beta, not an approximation.

### 1. The data DO have a strong time structure

Instruction onsets taken from the `ev_*_instruction_onset.txt` files the
first-level GLM actually used (one block per task, 5 contiguous repeats).
Mean beta over searchlights for a pure time RDM:

| time RDM | mean beta |
|---|---:|
| log lag between onsets | **+0.1439** |
| linear lag | +0.1401 |
| block order gap | +0.1381 |

That is ~3x larger than any reward model effect here. Temporally distant
blocks are reliably more dissimilar. So the confound is real and large.

### 2. But it does NOT explain the negative — it is the wrong sign

The instruction model is *positively*, weakly correlated with lag
(r = 0.048 log / 0.090 linear), so partialling time out makes the negative
**slightly worse**, never better:

| fit | ABCD_rew_instr beta |
|---|---:|
| alone | -0.0498 |
| + log lag | -0.0568 |
| + linear lag | -0.0629 |
| + block order gap | -0.0616 |
| + ABCD_rew + log lag | -0.0568 |

Same for the execution model (-0.0071 → -0.0131). **Time hypothesis rejected.**
Mean lag at the 10 model-forced-identical cells (533 s) is indistinguishable
from all other cells (547 s), and within those 10 cells the contribution is
unrelated to lag (the two longest-lag pairs contribute ~0/positive; the
shortest-lag pair E1 contributes strongly negative).

### 3. The actual mechanism: leverage on the low-dissimilarity cells

The beta is additive, `beta = sum_c z(model_c) * z(data_c) / 90`. Attribution
by model level:

| model dissimilarity | n cells | mean z(neural) | contribution to beta |
|---:|---:|---:|---:|
| 0.00 (same task, forced identical) | 10 | **+0.130** | **-0.0316** |
| 0.25 | 8 | **+0.178** | **-0.0224** |
| 0.75 | 40 | -0.091 | -0.0048 |
| 1.00 | 32 | +0.029 | +0.0091 |

64% of the negative comes from the 10 forced-zero cells and a further 45% from
the 8 cells at 0.25 — i.e. **the cells the model calls most similar are the
cells the data call most dissimilar**. Because the model is skewed (18 of 90
cells at the low end), z(model) there is -2.19, so those few cells carry
enormous leverage.

The same inversion holds for the execution model (its 4 cells at d=0.25 have
mean z(neural) = +0.132, contributing -0.0220) — it just has 4 such cells
instead of 18, so its total stays near zero. **This predicts the across-subject
negative shift scales with how many low-dissimilarity cells a model has**,
which is consistent with the audit: memory-within (many) is the most negative
model, plan-within (few) is near zero.

### 4. It IS the forward/backward comparisons — confirmed structurally

The 10 model-forced-identical cells are **exactly** the 10 same-task
forward/backward pairs (verified as an identical set, not merely overlapping).
No two different tasks share an instructed reward sequence, so dissimilarity =
0 is enforced *only* on direction-crossing comparisons. All the model's
low-end leverage therefore sits on the forward/backward contrast.

Premise check (SK): A1_forw and A1_backw really are instructed with the same
reward sequence — but the screen additionally shows a "please backwards" text,
or nothing, implying forwards. So the instruction screens are **not** identical
across directions, and `ABCD_rew_instr` is mis-specified exactly where it has
the most leverage.

### 5. A forward/backward co-regressor does NOT fix it

| fit | ABCD_rew_instr beta |
|---|---:|
| alone | -0.0498 |
| + direction_diff (binary F/B crossing) | **-0.0578** |
| + direction_diff + log lag | -0.0624 |
| + different_task (binary) | -0.0356 |

`direction_diff` is itself negative alone (-0.0188) and correlates -0.245 with
the instruction model, so partialling it out pushes the instruction beta
*further* negative. A single additive direction term cannot work, because the
direction effect is not additive — it reverses sign with task identity:

| cells | n | mean z(neural) |
|---|---:|---:|
| B-F, same task (the forced-zero cells) | 10 | **+0.130** |
| B-F, cross task | 40 | -0.054 |
| F-F, cross task | 20 | +0.120 |
| B-B, cross task | 20 | -0.078 |

### 6. Where the negative actually lives — model-matched subset refits

Refitting the instruction model on cell subsets (all 126,321 searchlights):

| subset | n cells | mean beta |
|---|---:|---:|
| all cells | 90 | -0.0498 |
| drop the 10 forced-identical cells | 80 | -0.0237 |
| **cross-task, same-direction** | 40 | **+0.0068** |
| **cross-task, different-direction** | 40 | **-0.0554** |
| different-direction only (B-F) | 50 | -0.0893 |

The last two rows are the clean test: identical model-value distributions, no
forced-zero cells in either, differing *only* in whether the pair crosses the
forward/backward boundary. **The negative is entirely on the crossing side;
within same-direction cells the model is flat (~0), not negative.**

So the answer is: not time, and not a removable additive direction confound —
the reward-sequence model simply fails to predict neural similarity whenever a
pair crosses the forward/backward boundary, and the instruction model is built
so that its most influential cells are all such crossings.

### 7. No control regressor can soak up the negative — and here is why

Tested ladder (sub-02, 126,321 searchlights), all as `instr + control`:

| control | cols | mean instr beta |
|---|---:|---:|
| none | 0 | -0.0498 |
| n_backward (0/1/2) | 1 | **-0.0498 (exactly unchanged)** |
| direction_diff (binary crossing) | 1 | -0.0578 |
| direction levels (F-F / B-F / B-B) | 2 | -0.0578 |
| different_task | 1 | -0.0356 |
| condition identity (20 per-condition dummies) | 20 | -0.0704 |
| condition identity + different_task | 21 | -0.1076 |
| condition identity + different_task + log lag | 22 | -0.1213 |

**Every control leaves it unchanged or makes it worse.** The structural reason
is a dilemma, visible in the design correlations:

- `corr(instr, n_backward) = +0.000000` — exactly orthogonal by design (the
  same-task cells sit exactly at the mean of n_backward, and among cross-task
  cells direction is independent of reward overlap). A regressor orthogonal to
  the model cannot move its coefficient at all, and this one moves it by
  literally zero.
- `corr(instr, direction_diff) = -0.245`, and direction_diff is itself negative
  (-0.0188 alone), so partialling it out pushes instr *further* negative.
- `corr(instr, different_task) = +0.774` — this is the only control that
  shrinks the negative, and it does so precisely because it **is** the
  instruction model's own main contrast (same task vs not). Absorbing it
  deletes the hypothesis rather than controlling a confound.

So: anything correlated enough with the instruction model to move its beta *is*
the hypothesis; anything that is a genuine nuisance is uncorrelated and does
nothing. Controls that remove the positively-contributing cells (the d = 1.00
level, +0.0091) make the negative worse, which is exactly what the
condition-identity rows show.

### 8. Respecifying the model to match the screen also does not rescue it

Since the instruction screen carries a direction cue, the instructed stimulus
is 4 reward locations *plus* that cue:
`d = (1-alpha) * reward_hamming + alpha * direction_mismatch`.

| alpha (direction cue weight) | d for same-task F/B | mean beta |
|---:|---:|---:|
| 0.00 (current model) | 0.00 | -0.0498 |
| 0.20 (one of five screen features) | 0.20 | -0.0582 |
| 0.50 | 0.50 | -0.0488 |
| 1.00 (pure direction model) | 1.00 | -0.0188 |

Negative at every weight, including a pure direction model. The reward-sequence
content never earns a positive coefficient in these data.

### 9. Conclusion: this is a cell-selection question, not a regressor question

The only unconfounded test of "do overlapping instructed reward sequences
produce similar instruction-period patterns" is the **cross-task,
same-direction** cells (40 cells, no forced-zero cells, no direction crossing):
**beta = +0.0068**, i.e. null, but not dragged negative.

Restricting the fitted cells is implementable in the existing framework (the
RSA runner already selects cell subsets via `data_rdm_scope`). It is legitimate
here *because the direction cue is a known stimulus property, established from
the screen content independently of any fit* — not because it improves the
number. Choosing among the controls above by which one flips the sign would be
post-hoc and is explicitly rejected.

### 10. What the same-direction mask actually leaves (sub-02)

Figure: `sub-02_same_direction_mask.png` in the output dir.

The mask keeps 40 of 90 cells (20 F-F + 20 B-B), all cross-task. But:

- **The 40 cells are only 20 unique comparisons.** Within a half each task has
  exactly one forward and one backward condition, and the instruction model
  gives the backward condition the forward vector, so
  `d(X_F, Y_F) == d(X_B, Y_B)` identically. Each task pair is therefore
  measured twice with the *same* model value. The model has 20 independent
  predictions, not 40. This is structural and holds for every subject.
- **The d = 0 level disappears entirely** (all 10 zero cells were B-F). Model
  levels retained: 0.25 / 0.75 / 1.00; model sd falls from 0.325 to 0.218 (67%).
- **The "similar" end rests on 2 task pairs** — C1-E1 and C2-E2, the only
  comparisons at d = 0.25 (4 cells). This part is subject-specific: it depends
  on the particular reward sequences.

**The duplication is a free split-half reliability check, and in this subject
it fails:**

| comparison | r |
|---|---:|
| task-pair level, F-F estimate vs B-B estimate (20 pairs) | **+0.020** |
| searchlight beta maps, F-F-only vs B-B-only fit | **-0.080** |

with beta = +0.0130 (F-F only) and +0.0011 (B-B only). So under the mask there
is little reliable neural structure for the model to explain in sub-02 — the
two measurements of the same prediction do not agree.

Caveat in the other direction: single-subject whole-brain RSA reliability is
routinely near zero and effects normally emerge only at group level, so this is
a warning rather than proof the mask is useless. It does mean the mask is not
rescuing anything here, and that this reliability check should be run per
subject on the event-locked caches before the mask is adopted.

### 11. Searchlight NIfTIs exported

`.../instruction_instr_negative_time_audit_2026-09-14/sub-02_searchlight_maps/`

| map | cells | mean beta |
|---|---:|---:|
| `ABCD_rew_instr_all90` | 90 | -0.0498 |
| `ABCD_rew_instr_same-direction` | 40 | +0.0068 |
| `ABCD_rew_instr_forward-only` | 20 | +0.0130 |
| `ABCD_rew_instr_backward-only` | 20 | +0.0011 |
| `ABCD_rew_instr_direction-crossing` | 50 | -0.0893 |

Each as `_beta` and `_t_val`, written on the subject's functional grid using the
pipeline's own saved map as geometry reference (shape 108x108x64, affine
verified identical), naming per `mc.analyse.handle_MRI_files`.

Two caveats attached to these files:

- **Subject functional space, not MNI.** They are not comparable to the group
  maps without applying the subject's registration.
- **The `_t_val` maps are first-level parametric t and must not be used for
  inference.** RDM cells are not independent observations, and under the
  same-direction mask they are doubly non-independent because each task pair
  appears twice (F-F and B-B) with an identical model value, so the nominal
  df = 38 is roughly double the truth. They are included for thresholding and
  visual inspection only; group inference stays a random-effects test over
  subject beta maps.

### What this does not yet settle

- Single subject, proxy GLM (see scope caveat). Must be rerun on the
  event-locked caches before any of this is asserted for the group.
- The F-F (+0.120) vs B-B (-0.078) asymmetry among cross-task cells is
  unexplained and may just be noise at n=20 cells in one subject.
- Whether an *interaction*-style regressor (direction crossing x same task),
  or simply excluding the direction-crossing cells, gives a defensible
  instruction model — not yet tested. Excluding them is a scope change to the
  hypothesis (it removes the only cells where instruction and execution models
  diverge), so this needs a decision rather than a fit.

## 2026-09-14 (b) — Pad to 100 ms; subfield moderator; He et al. methods obtained

### 1. `PAD_S` 0.25 → 0.1 s

**0.1 s is He et al. 2026's own figure** — their Methods (now in hand) discard
"any ripple occurring within 100 ms of IED" and likewise within 100 ms of a
pathological HFO. Recovers ~56% more artifact-free seconds than the 1 s pad.

Chosen as a **lower bound**, not as optimal. Because every event carries
`dist_to_artifact_s` and the bundle carries the unpadded crossings, any
stricter pad can be re-imposed downstream for free — so the cheap mistake is
padding too little here, not too much. Supported by the 2026-09-14 measurement:
ripple properties are flat across distance-to-artifact bins (peak frequency
98.1–98.5 Hz, duration 65.8–67.1 ms from 0.25 s out to ∞).

### 2. Correction — the `swr_v1` bundle CANNOT be re-padded

Stated carelessly yesterday. To be exact about what was measured where:

- the **pad sweep** and the **event-property measurement** were computed on the
  laptop from `continuous.npy` (present locally for 21 sessions), NOT from a
  bundle;
- the **re-padding demonstration** used the *new* s29 `swr_v2` proof-run bundle,
  built by the new code.

The downloaded `swr_v1` bundle (2026-09-08, 61 sessions, 212 derivations) has
no `dist_to_artifact_s` and no `artifact_intervals`, and was built at a 1 s pad.
Events outside that pad were never detected and do not exist in it at any price.
Re-padding only ever goes **stricter**, and only for bundles built by the new
code. This is precisely why the new pad is set at the lower bound.

### 3. `scripts/swr_subfield_moderator.py` — runs on the EXISTING bundle

Is the F5 first-D ripple increase stronger on CA-weighted derivations, as
Sakon et al. report for CA1 vs DG? Subfield is a pure MNI lookup and `pairs`
already carries coordinates, so **no cluster run and no LFP are needed**.

Result (212 derivations, 61 sessions, 42 subjects):

| condition | moderator `delta ~ ca_dg_index + (1\|subject)` |
|---|---|
| first_D | β = −0.078, p = 0.285 |
| later_D | β = +0.008, p = 0.711 |

No CA gradient, and the first-D point estimate runs *opposite* to Sakon.

**But the sample cannot test that contrast.** Only 2 of 199 derivations have
P(DG) > P(CA); the lower half of the index still has P(CA) > P(DG) in 95 of 97.
That is `select_hpc_contacts` **working as intended** — it takes the deepest
hippocampal contact per probe, which mostly sees cornu ammonis — so the montage
is already concentrated where Sakon put the effect, which helps detect F5 at
all. The narrow cost is that the CA-vs-DG contrast has no DG arm: a null on the
moderator is uninformative about the gradient and says nothing about F5 itself.
Testing it properly needs the runner-up contacts per probe (56 of 128 clear the
25% threshold, `methods.md` §4.2), which this montage deliberately excludes.

The script prints this as a precondition rather than leaving it to be inferred.
Its Eq. 2 figure is a derivation-level sanity check, not the canonical F5 —
`fit_eq2` expects trial-level rows and drops subjects below its 10-row floor
(13 of 42 survive). The moderator uses all derivations and is the actual test.

### 4. He et al. 2026 Methods — what they actually did

Obtained, resolving what was previously flagged as unverifiable:

- **Hippocampal contacts are referenced against a nearby WHITE MATTER contact
  on the same probe**, not against an adjacent contact. This project rejected
  white-matter referencing for hippocampus deliberately (`methods.md` §4.2:
  it doubles inter-contact distance and enlarges the lead field). A real
  divergence, and a pre-existing decision — not changed here.
- All other contacts: adjacent-bipolar on the same probe. **Confirms the
  cortical montage choice**: every in-ROI contact, not one per probe.
- Ripple band **70–180 Hz**, 4 SD threshold extended to 2 SD, 20–200 ms,
  30 ms merge. This pipeline follows Chen (80–120 Hz, dual 1.5/3.0 SD,
  38–500 ms) — a deliberate, documented difference.
- HFB **80–140 Hz**, *overlapping* 10 Hz sub-bands at 5 Hz steps, each divided
  by its own channel-and-band mean, summed, averaged, rescaled by the grand
  mean, then 10·log10 to dB. Ours: 70–150 Hz, contiguous 10 Hz sub-bands,
  log-then-z-score per band. Both are standard normalisations; ours follows
  `criterion_broadband_power` already in this codebase.
- They recorded at 50 Hz line frequency, so **100 Hz sits inside their
  80–140 Hz HFB band and they simply notch through it.** Our notch-aware
  sub-band dropping has no counterpart in their pipeline.
- Also: contacts >3 mm from the cortical ribbon excluded; a common-average
  control removes ripples coincident across contacts.


## 2026-09-14 (b) — Exploratory mistake sweep (all null), and the final figures rebuilt

### Exploratory: do ripples predict mistakes? No.

`scripts/swr_explore_ripples_and_mistakes.py`, results in
`swr/explore_ripples_mistakes_2026-09-14/`. **EXPLORATORY** — .png only, nothing
confirmatory. Window +0.35 to +0.75 s (where the stillness-matched cluster
peaked), two measures: `delta` (window minus the same trial's baseline) and
`raw` (window alone). Paired t-tests at session level: within each session the
mean over grids WITH the outcome against the mean over grids WITHOUT.

| question | measure | horizon | no mistake | mistake | Δ Hz | p_perm | n |
|---|---|---|---|---|---|---|---|
| first D | delta | all later repeats | +0.1009 | +0.1012 | +0.0003 | 0.996 | 44 |
| first D | delta | next 3 repeats | +0.0449 | +0.0847 | +0.0398 | 0.685 | 29 |
| first D | raw | all later repeats | +0.2840 | +0.2427 | −0.0413 | 0.307 | 46 |
| first D | raw | next 3 repeats | +0.2857 | +0.2379 | −0.0478 | 0.351 | 36 |
| all first uncovers | delta | all later | +0.0634 | +0.0717 | +0.0083 | 0.796 | 53 |
| all first uncovers | delta | next 3 | +0.0693 | +0.0677 | −0.0015 | 0.968 | 51 |
| all first uncovers | raw | all later | +0.2575 | +0.2464 | −0.0112 | 0.607 | 53 |
| all first uncovers | raw | next 3 | +0.2496 | +0.2511 | +0.0015 | 0.958 | 51 |

Nothing, in either measure or either horizon. The `raw` first-D tests lean the
predicted way (more ripples → fewer later D errors, ≈ −0.04 Hz) but at p ≈ 0.3.

### Descriptive: mistakes have strong structure

27,538 errors, 100% with a location.

- **Immediate repetition is AVOIDED.** In the transition matrix of consecutive
  errors while seeking the same reward, the diagonal is **4.5%** against ~12.5%
  chance. Having just tried a tile, subjects do not retry it.
- **The heavy off-diagonal cells are grid-adjacent** (1→4, 4→7, 2→5): the
  search walks to neighbouring tiles.
- **But errors do recur across repeats:** **23.6%** are eventually repeated at
  the same reward *and* the same location — roughly double chance — evenly
  across A–D (21.9, 22.5, 25.3, 25.1%).
- **Ripples at an error do not prevent repeating it**, and the non-significant
  trend runs the wrong way: repeated errors had *higher* rates
  (delta +0.0450 vs +0.0022, p = 0.151; raw +0.2075 vs +0.1871, p = 0.288).

### Final figures rebuilt

The 3.5 cm eight-panel grid and the raster are retired — the raster showed
nothing at 0.2 Hz over 4 s (under one ripple per row). Replaced by two figures
in the house 16 cm × 4 cm format:

- **`ripple_main_figure`** — three rows through `rip.plot_rows`: positive
  feedback and negative feedback against their own baselines (the
  `feedback_stage` panel), then the same headline cell against its
  stillness-matched control press. One figure, both references.
- **`ripple_methods_figure`** — the controls at the same scale (2 × 2, 16 cm).

**Display smoothing added.** `rip.triangle_smooth` implements Sakon & Kahana's
5-bin triangle, and `plot_rows(smooth_bins=…)` applies it to the LEFT panel
only. The middle and right panels, and every statistic, still read the
unsmoothed profiles — this is why their PVTHs look smooth and ours did not.

**Two `plot_rows` parameters added.** `colours=` overrides `condition_colour`
for labels the project's valence/stage naming cannot colour (the control press
is neither, and was falling through to a tab10 orange — the hue reserved for
state A); `legend_cm=` sizes the legend strip, which a three-row figure with
eight entries overflowed into the bottom row.

On the current bundle the `feedback_stage` rows give: correct/first uncovers
increase +0.25…+0.65 s, p = 0.005; error/while learning decrease +1.15…+1.45 s,
p = 0.046; the other four none. Row 3 (matched) gives +0.35…+0.75 s, p = 0.004.


## 2026-09-14 — Hippocampal subfield, a matched temporal control, and post-hoc re-padding

Follow-up to 2026-09-13 (f), on the same branch `swr-mpfc-hfb`.

### 1. The pad now has evidence, not just a rationale

The 0.25 s choice was a reasoned prior, **not a measurement** — stated as such
at the time. The measurement has now been run: 15,036 accepted events on 29
derivations across 11 sessions, comparing events a 1 s pad would have kept
(`dist_to_artifact_s >= 1`) against those the 0.25 s pad newly admits.

| | kept by 1 s pad | newly admitted | diff |
|---|---|---|---|
| peak frequency (Hz) | 98.12 | 98.38 | +0.3% |
| duration (ms) | 66.3 | 67.0 | +1.1% |
| RMS peak (z) | 3.847 | 3.918 | +1.8% |
| amplitude (µV) | 1.553 | 1.680 | **+8.2%** |

29.5% of events are newly admitted. By distance bin the profile is **flat and
monotonic, with no discontinuity near the artifact**: peak frequency
98.1–98.5 Hz and duration 65.8–67.1 ms across every bin from 0.25 s to ∞.

**Caveats, both real.** The +8.2% amplitude difference is a genuine monotonic
gradient (rms z 3.92 nearest → 3.80 farthest); events near crossings are
slightly larger. That is consistent with contamination *and* with epileptogenic
tissue genuinely producing larger ripples, and this measurement cannot separate
the two. Separately, `spectral_passed_strict` reads 1.000 in every bin only
because the comparison is over events that already passed — it carries no
information here and should not be quoted as if it did.

### 2. Any pad can now be re-imposed on the laptop

Previously impossible, and **`dist_to_artifact_s` was being silently dropped by
the bundle's column filter** — so the per-event distance stored on 2026-09-13
would never have reached the laptop at all.

- detection writes `artifact_intervals.csv`, the **unpadded** criterion union
- `swr_artifact.clean_intervals_at_pad` rebuilds the exposure denominator at
  any pad from it. Verified EXACT against the cluster-side mask at 0.1 / 0.25 /
  0.5 / 1.0 s (dilation distributes over union, so this is not an approximation)
- the bundle carries both, plus `dist_to_artifact_s` per event

⚠ Both halves are needed. Filtering events without shrinking the denominator
inflates the rate; the bundle `meta` says so.

### 3. Hippocampal subfield — as probabilities, not a label

Every contact and hippocampal derivation now carries Jülich `p_CA`, `p_DG`,
`p_SUB` and `ca_dg_index` = (P(CA) − P(DG)) / (P(CA) + P(DG)).

**Three limits, all load-bearing:**

1. **This is not CA1 vs CA3.** Jülich's "cornu ammonis" is one volume covering
   all CA fields. Nothing in this project's atlas stack resolves CA1 from CA3.
2. **A hard label would be a volume artefact** — max-prob gives 88 CA / 5 DG /
   7 SUB of 103 derivations. Probabilities overlap heavily on the same contact
   (median P(CA) 78, P(DG) 38) because the subfields interdigitate below this
   resolution. Use `ca_dg_index` as a continuous moderator only.
3. **ASHS was the right tool and is not available** — populated for 3 of 103
   selected contacts. Obtaining ASHS segmentations is the real fix.

Also surfaced: **12 of 103 selected hippocampal contacts are called *amygdala*
by Jülich** while the Harvard–Oxford rule that selected them says hippocampus.
Worth a sensitivity analysis (`bRAMY2-bRAMY3` in s29 is one such derivation).

### 4. A matched temporal control (SK's suggestion, and it is the better one)

A temporal depth electrode is inserted laterally, so its outer contacts sit in
lateral temporal cortex **on the same shaft as the hippocampal contact** — same
amplifier, reference chain, noise environment and trajectory.

Measured: **353 TemporalLateral derivations over 31 sessions, 158 of them on a
probe that also carries the selected hippocampal contact**, plus 54 Auditory
(Heschl / planum). Flagged per derivation as `same_probe_as_hpc`.

This is strictly better than Visual, which differs from mPFC in every one of
those respects at once. Visual is kept anyway for like-for-like comparison
with He et al.

Labels come from the Harvard–Oxford **cortical** atlas via the new
`contact_anatomy.add_swr_roi`, writing a new `roi_swr` column. It only fills in
contacts the shared ROI ladder had already returned `leftover` for, so
`anatomy_atlas.assign_atlas_roi` — which is byte-for-byte shared with the cell
pipeline — is untouched.

`CORTICAL_ROIS` is now `mPFC, mOFC, TemporalLateral, Auditory, Visual`. Insula
(172 derivations) and PCC (27) were dropped from the default: neither target
nor matched control, and every extra derivation is extra channels on the one
I/O-bound stage.

### 5. Two bugs found by the proof-run

- **Ripple detection was running on the cortical derivations.** They share the
  pairs table so extraction reads every channel in one pass, but Chen's band,
  thresholds and duration gate are hippocampal. This would have emitted a table
  that looks exactly like a ripple table and means nothing. Detection now
  filters on `role == 'ripple'` and says how many rows it skipped.
- **Stale `bipolar_pairs_*.csv` files** from before the cortical montage (s13,
  s24 locally) carry no `role` column. They would extract hippocampus-only and
  look complete. `swr_make_conditions.py` now detects and reports them.

### 6. Montage figure

`ripple_figures.montage_figure`, drawn automatically by `swr_build_contacts.py`
to `group/swr/figures/montage_coverage.pdf/.jpg`. Derivation midpoints — where
a bipolar derivation is actually sensitive — coloured by ROI, with same-shaft
controls ringed in black. Targets take the project's Showgirl2 colours; controls
are grey, so target and control are distinguishable without reading the key.

### 7. Proof-run on the laptop (s29, full chain)

`build_contacts → extract_continuous → detect_session → extract_hfb`, all green:

- 15 derivations: 3 hippocampal, 3 mPFC, 5 mOFC, 3 TemporalLateral, 1 Visual
- 30 unique channels in one raw pass; residual line noise 0.00 at all harmonics
- detection: 3 hippocampal rows, 12 cortical correctly skipped; rate 0.178 Hz
  (Chen 0.17–0.24); 22% of events within 1 s of a crossing
- **120 Hz was notched on 8 of 15 derivations**, so 8 dropped to 6 HFB
  sub-bands. Without the notch-aware band, more than half this session's HFB
  would have been measured through a notch
- **positive control: all 3 hippocampal derivations positive** (+0.099, +0.120,
  +0.085 z peri-ripple vs flanks)
- `hfb.npz` 43 MB for 15 derivations × 0.87 h ≈ 3.3 MB/derivation/hour,
  projecting to ~1.9 GB for the cohort


## 2026-09-14 — Figures split, and two framing corrections

Follow-up to (e), after SK's questions.

**Two figures instead of one.** `ripple_results_figure` shows the data: a
Sakon-style raster (one row per trial × derivation, one tick per ripple), the
peri-event rate for uncovering and its matched control, the sliding *t* curve
with the surviving cluster, and the baseline-vs-test window means.
`ripple_methods_figure` shows the controls: the stillness imbalance that
motivates the design, the matching balance, control neutrality across the four
arrow keys, and the matched-vs-unmatched selection check. All panels measured
to 3.51 × 3.51 cm.

**Correction 1 — "raw / no baseline" was the wrong name.** The matched control
*is* a baseline; it is a different EVENT rather than a different TIME. Renamed
throughout to "own-baseline reference" and "matched-event reference", with a
table in the methods stating what each does and does not control for.

**Correction 2 — the primary/secondary ordering was backwards.** Sakon &
Kahana's Eq. 2 (own-baseline) is the field's statistic and is now the headline:
+0.059 Hz [+0.011, +0.106], t(60) = +2.46, p = 0.018, cluster +0.25…+0.75 s,
p = 0.008. The matched-event contrast (+0.035 Hz, cluster +0.35…+0.75 s,
p = 0.004) is now presented as the control analysis showing the effect is not
stillness. Both readings were always reported; only which leads has changed.

**New observation from the window table.** The effect is a CROSSOVER, not an
offset: at the −1.6…−1.1 s baseline the uncovering sits *below* its matched
control (0.180 vs 0.204 Hz) and by 0–0.5 s it sits above it (0.225 vs
0.190 Hz). The matched-event difference crosses zero about 1 s before the press
and is already positive in the pre-event window (+0.039 Hz, p = 0.074;
own-baseline +0.063 Hz, p = 0.027). A rise beginning before the press resembles
Sakon's PRE effect, though the pre-event portion does not survive cluster
correction alone. Worth pursuing.

**On whether the controls are too strict.** Sakon & Kahana never match on a
behavioural covariate: their PRE and BASE windows both sit in the same silent
pre-vocalisation stretch, so motor state is constant across the comparison by
construction. Our window is post-press and therefore downstream of what the
participant does next, which is exactly what differs between conditions — so
the control is warranted here and would be redundant there. Their 1st-vs-≥2nd
recall comparison probably does carry an analogous timing confound that they
address only with the 2 s exclusion.

## 2026-09-13 (f) — Artifact pad 1.0 s → 0.25 s, and a cortical (HFB) branch

Two changes on branch `swr-mpfc-hfb`, both additive. Motivated by He et al.
2026 (*Nat Neurosci* 29:1711), who show hippocampal ripples update mPFC
representations — the "how is the action plan loaded" step this project has not
yet tested. Ideas the new data makes testable are listed in
`data/final_results/ripple_analysis/POTENTIAL_IDEAS.md`; none has been run.

### 1. Artifact pad (`mc/analyse/swr_artifact.py`)

`PAD_S` 1.0 → **0.25 s**. Measured across 27 derivations in 11 sessions, by
re-dilating the *same* per-criterion masks:

| pad | mean clean fraction | derivations >2/3 contaminated | exposure vs 1.0 s |
|---|---|---|---|
| 1.0 s (old) | 0.569 | 3/27 | — |
| 0.5 s | 0.723 | 0 | +27% |
| **0.25 s (new)** | **0.819** | 0 | **+44%** |
| 0.1 s | 0.885 | 0 | +56% |

The five criteria flag ~2.6% of samples between them but
`frac_combined_after_pad` was 47.6% across the cluster bundle — the pad, not
the criteria, was doing essentially all of the rejection.

The binding reason is cross-regional: a peri-ripple window needs ±750 ms clean
in the hippocampal **and** the cortical derivation. Measured on the bundle, only
56.7% of accepted ripples had a clean ±750 ms in their own derivation, which
against an independent cortical mask leaves ~32% usable.

Detector metrics are stable across the change (s38): rate 0.222 → 0.213 Hz
(Chen 0.17–0.24), spectral rejection 30.3% → 30.1%. Only exposure moves.

Three guards, because this is a parameter change made after seeing the data:

- **`EXCLUSION_PAD_S = 1.0`, fixed.** Contamination is judged at the old pad so
  the set of included derivations cannot move with `PAD_S`. Otherwise "more
  exposure" and "dirtier contacts added" are confounded.
- **`dist_to_artifact_s` is stored per event** — distance to the nearest
  *unpadded* criterion crossing. Any larger pad can be re-imposed by filtering
  that column, so a 0.25 s run reduces exactly to the 1 s result with no
  re-detection. The pad is no longer a decision that must precede the analysis.
- **Rejected metric, recorded so it is not retried:** "fraction of accepted
  events within 100 ms of a crossing" is identically **zero** for any pad above
  100 ms, because detection only ever runs on clean samples. It cannot
  discriminate between pads. The distance distribution can.

`swr_detect_session.py` gained `--pad_s` and `--clean_name` (which extraction to
*read*), so a pad sweep costs no raw I/O. Refactor verified: `criteria_masks` +
`combine_criteria` reproduce the stored pad-1.0 numbers to 5 decimal places.

### 2. Cortical branch — HFB (`mc/analyse/swr_hfb.py`, `scripts/swr_extract_hfb.py`)

`bipolar_pairs_{XX}.csv` now holds both montages, keyed by `role`:

- `ripple` — hippocampal, one derivation per probe. **Unchanged**, verified
  identical to the committed `bipolar_pairs_38.csv`.
- `hfb` — cortical (mPFC, mOFC, Visual, Insula, PCC), non-overlapping adjacent
  pairs anchored on each in-ROI contact. Verified invariant across 6 sessions:
  **no contact appears in two derivations.** One-per-probe was not used here —
  He et al. ran on 47 dmPFC contacts, which that rule cannot reach — but the
  independence it protects is kept.

Cortical pairs ride in the same table so stage 2 reads them in the **same
raw-file pass**; extraction is the only I/O-bound stage and running it twice
would cost hours for nothing. Resources bumped 64G/4h → 96G/8h.

`swr_extract_hfb.py` is a **sibling of detection**, not a successor — both read
`continuous.npy`. Per derivation it writes HFB (70–150 Hz, per-sub-band log +
z-score, averaged) plus `ripple`/`theta`/`beta` envelopes and `theta_phase`, all
continuous at 100 Hz, float16.

- **Notch-aware.** 120 Hz sits inside any HFB band and this pipeline's notch is
  adaptive *per derivation*, so a fixed band would mean different things on
  different contacts of the same session. Sub-bands on a notched harmonic are
  dropped and `n_sub_bands` is carried per derivation.
- **z-scored on artifact-free samples only**, else the flagged samples — by
  construction the largest excursions — set the SD.
- **Nothing is epoched.** Windows, baselines and phase splits stay free choices
  on the laptop; that was the point of the stage.
- **Positive control built in:** peri-ripple HFB at hippocampal derivations is
  computed and printed. On s38: **+0.115 and +0.147** z against flanks. If this
  is not positive the clocks disagree and no cortical number means anything.

Coverage (local 32-session subset; the cluster has ~46): mPFC 80 contacts / 23
sessions / 17 subjects, mOFC 225 / 26 / 18, Visual 179 / 19 / 13. Every session
with an mPFC contact also has a hippocampal one. He et al. worked with 47 dmPFC
+ 12 vmPFC. ⚠ s26/27/28 are one subject contributing 10 mPFC contacts each —
subject-level inference stays mandatory.

### 3. Bundle — still the final stage

`export_bundle(with_hfb=True)` copies the HFB stores to `bundle/hfb/s{NN}_hfb.npz`,
one per session, read back with `swr_bundle.load_hfb` (casts to float32; float16
accumulates error over the long sums that averaging epochs is). Kept beside the
pickle, not inside it, so `swr_bundle.pkl` stays a few MB and loadable alone;
the HFB half is ~1–2 GB. `--hfb_arrays` narrows the transfer.

**Bug found and fixed while testing:** `export_bundle` read `pairs.csv` from
`LFP-clean/{analysis_name}`, so for any run using `--clean_name` the pairs table
came back **empty** — and the `FileNotFoundError` then skipped that session's
behaviour and uncover tables too, silently. It now reads `clean_name` from the
detection settings. This would have hit every pad-sweep bundle.


## 2026-09-13 (e) — Final consolidated ripple analysis, figure and manuscript drafts

The analyses SK settled on are now in one runnable script rather than spread
across five exploratory ones.

**`scripts/swr_final_ripple_analysis.py`** — self-contained (imports only
`mc.analyse.ripples`, `swr_io`, `swr_behaviour`), one command, no hard-coded
bundle. It builds `swr/press_categories.csv` from the raw 25 ms button series
if that cache is missing or older than the bundle, so pointing `--bundle` at
new data is the only change needed when more sessions arrive. The cache build
is the only slow step (~15 min, I/O bound); everything after it is minutes.

It runs, in order: control validity (four arrow keys compared), matching
balance, selection check (matched vs unmatched targets), per-cell contrasts
against stillness-matched controls in two readings (raw = primary, own-baseline
= secondary), and the five between-cell contrasts including the valence × stage
interaction.

**Outputs** in `swr/ripple_final_<date>/`:
- `ripple_final_figure.png` / `.pdf` — 8 panels, each measured to
  3.51 × 3.51 cm (the layout iterates until the axes hit the requested 3.5 cm,
  because `tight_layout` otherwise gives panels whatever the text leaves over).
  Figure 20.1 × 10.7 cm overall. Type steps to 8/7.5/7 pt: the house 9 pt fills
  a third of a 3.5 cm panel.
- `ripple_statistics.csv` — 72 rows, one per test: analysis, contrast, reading,
  window, n_units, mean, SEM, 95% CI, t, df, p, p_perm, Cohen's d, n_events.
- `ripple_statistics.json` — the same plus every setting, the descriptives and
  the full time courses, for provenance.

**Manuscript drafts**, in `data/final_results/ripple_analysis/manuscript/`:
- `statistics_methods_DRAFT.md`
- `statistics_results_DRAFT.md`

Both quote only numbers present in the statistics files. The results draft leads
with the raw reading, states the own-baseline reading as secondary and says why
it is inflated, reports the interaction as underpowered rather than null with
its CI, and carries four caveats into the discussion (exploratory status, button
stillness only, the control not being a blank, and the 57% matched subset).

All values reproduce the exploratory runs of entries (b)–(d) exactly.

## 2026-09-13 (d) — CORRECTION: with 1:1 stillness matching the first-traversal effect SURVIVES

This supersedes the conclusion of entries (b) and (c) for one cell. SK proposed
matching each uncover press against a press of a category carrying no
hypothesis — an arrow key — with the same following stillness, rather than
against a pooled or coarsely binned control. Done properly, the effect that
binning had washed out comes back.

New scripts: `scripts/swr_build_press_categories.py` (one slow read of the
25 ms button series, cached as `swr/press_categories.csv`) and
`scripts/swr_matched_control_presses.py`. Results in
`.../swr/matched_control_presses_2026-09-13/`.

**Press categories.** 248,737 presses, 61 sessions: Return 97,661;
RightArrow 40,554; LeftArrow 37,751; DownArrow 37,173; UpArrow 35,598.
97,660 of 97,661 uncover presses carry a valence/stage label.

**Matching.** Per session, 1:1 nearest neighbour on `still_next_s`, WITHOUT
replacement, caliper max(0.15 s, 10%), targets served in seeded random order
(seed 42), both sides de-duplicated at 2 s before matching. Achieved balance is
essentially exact — target and control median stillness agree to 3 decimals in
every category (e.g. 0.500 vs 0.500 s; 1.175 vs 1.175 s).

**The control is neutral.** Rate after each arrow key, gaps 1–2.5 s:
Left 0.1879, Right 0.2069, Up 0.2041, Down 0.1957 Hz — all within 0.02 Hz with
overlapping SEMs, so "an arrow key" is one category and direction need not
enter the matching.

### Results, 0–0.5 s, uncover minus its matched arrow key

| target | matched | vs own baseline | raw (no baseline) | raw cluster |
|---|---|---|---|---|
| **correct, first uncovers** | 3,074 (57%) | **+0.0586, p = 0.018** | **+0.0349, p = 0.055** | **+0.35…+0.75 s, p = 0.004** |
| correct, while learning | 4,061 (90%) | +0.0409, p = 0.024 | +0.0198, p = 0.090 | none |
| correct, once known | 10,412 (80%) | +0.0343, p = 0.118 | +0.0062, p = 0.611 | none |
| error, first uncovers | 5,671 (94%) | +0.0001, p = 0.996 | +0.0054, p = 0.617 | none |
| error, later | 3,891 (91%) | +0.0231, p = 0.287 | +0.0026, p = 0.875 | none |
| first D (F5) | 1,085 (76%) | +0.0302, p = 0.503 | +0.0036, p = 0.918 | none |

n = 61 sessions for all but error/later (57) and first D (54).

### What this changes

1. **The first-traversal correct uncovering is NOT explained by stillness.**
   Against a press matched on stillness to three decimals, with no baseline
   window anywhere in the statistic, it carries +0.035 Hz more and produces a
   cluster surviving correction over all window positions (p = 0.004; still
   p < 0.05 after Bonferroni over the six targets). Entries (b) and (c)
   concluded otherwise from coarse binning, which compared a target at 1.6 s
   with a control at 2.4 s inside the same bin and lost most of the sample
   (n fell to 32); that was a power and matching failure, not an absence.

2. **The stage gradient is real in the matched data.** Raw: first uncovers
   +0.035 > while learning +0.020 > once known +0.006, with errors flat at
   every stage (+0.005, +0.003). This is the pattern SK saw in
   `feedback_stage`, now with stillness removed by construction.

3. **F5's D-specific claim does not survive; the general first-traversal
   claim does.** `first D` gives +0.0036 Hz (p = 0.92) while all first-traversal
   correct uncoverings give +0.035 Hz. The effect is about uncovering a reward
   for the first time, not about D. Since F5 was the one pre-declared claim,
   this needs stating plainly in any write-up.

4. **The own-baseline reading is inflated and should not be the headline.**
   It exceeds the raw reading in every one of the six cells (+0.059 vs +0.035,
   +0.041 vs +0.020, +0.034 vs +0.006, …). That is the pre-event stillness
   imbalance predicted in entry (b), Limitation 2, showing up exactly as
   expected: controls sit in already-quiet stretches, so their baselines are
   ripple-richer and their (window − baseline) is depressed. The raw reading is
   the one to quote.

### The unmatched 43% are not a favourable selection

The obvious worry about a 57% match rate is that matching kept the events with
the smaller effect. It did not — it kept the ones with the SMALLER effect in the
opposite sense, i.e. the drop is conservative. Each subset against its own
baseline, 0-0.5 s:

| target | matched | still | unmatched | still | paired Δ |
|---|---|---|---|---|---|
| **correct, first uncovers** | **+0.0446** (n=61) | 1.05 s | **+0.0647** (n=55) | 2.19 s | −0.019, t = −0.82, p = 0.42 |
| correct, while learning | +0.0209 (n=61) | 0.52 s | −0.0235 (n=13) | 3.32 s | +0.051, p = 0.38 |
| correct, once known | +0.0167 (n=61) | 0.50 s | −0.0124 (n=45) | 0.53 s | +0.043, p = 0.075 |
| correct, later | +0.0162 (n=61) | 0.52 s | — | 0.57 s | — |
| error, first uncovers | −0.0108 (n=61) | 0.52 s | — | 0.64 s | — |
| first D (F5) | +0.0385 (n=54) | 1.17 s | — | 3.03 s | — |

The unmatched first-traversal events are the LONG-stillness ones (median 2.19 s
vs 1.05 s) and they carry a BIGGER own-baseline effect (+0.065 vs +0.045), with
no significant difference between subsets (p = 0.42). So matching did not
cherry-pick the events that show the effect; it discarded the ones that show it
most, precisely because their stillness had no arrow-key partner. How much of
that extra +0.02 Hz is signal and how much is their doubled stillness cannot be
separated — which is exactly why they are excluded rather than included.

### The valence x stage interaction does NOT hold under matching

Each cell is already `uncover − its own stillness-matched arrow key`, so
differencing two cells is a stillness-adjusted comparison. Raw reading, 0-0.5 s:

| contrast | n | Δ Hz | t | p_perm | 95% CI | cluster |
|---|---|---|---|---|---|---|
| stage within correct (first − later) | 61 | +0.0205 | +1.07 | 0.291 | [−0.018, +0.059] | none |
| stage within error (first − later) | 57 | +0.0008 | +0.05 | 0.957 | — | none |
| valence at first (correct − error) | 61 | +0.0295 | +1.45 | 0.152 | [−0.011, +0.070] | none |
| valence later (correct − error) | 57 | +0.0118 | +0.77 | 0.449 | — | none |
| **INTERACTION valence × stage** | 57 | **+0.0211** | +0.80 | **0.439** | **[−0.032, +0.074]** | none |

Every contrast points the predicted way — the stage effect lives in the correct
cells (+0.021) and not in the error cells (+0.001), and valence separates more
at first (+0.030) than later (+0.012) — but not one of them reaches
significance, and no cluster survives.

This is the same dissociation the 2026-09-11 entry found and it is now
confirmed under the cleanest control available: **the simple effect at
correct/first-uncovers holds against its matched control; the between-cell
contrasts, including the interaction, do not.** The interaction CI spans
[−0.032, +0.074] Hz, which is consistent with anything from a small negative to
a substantial positive interaction — this is underpowered, not clearly null,
and should be reported as such rather than as an absence.

Arithmetically the reason is plain: a between-cell contrast is the difference
of two control-adjusted quantities, so its variance roughly doubles while its
expected size shrinks relative to the simple effect.

### Limitations

- **57% match rate for correct/first-uncovers.** Long-stillness targets are the
  hardest to match (the arrow-key pool thins out past 2 s), so the matched
  subset skews shorter than the full set (median 1.050 s vs 1.375 s). The
  contrast is stillness-balanced by construction, but it is established on that
  subset, not on all first-traversal uncoverings. See the subset comparison
  above: the omission is conservative.
- Matching without replacement discards targets that find no partner; counts
  are reported per category, never silently dropped.
- `first D` has the lowest match rate (76%) and the smallest n (54 sessions);
  its null is weakly powered and should not be read as evidence against F5, only
  as a failure to confirm it under this control.


## 2026-09-13 (c) — What stillness is; ripples are flat inside a pause; task-free rest is the highest rate in the dataset

SK's four objections to the stillness control, each answered.
`scripts/swr_stillness_anatomy.py`; results in
`data/ephys_humans/derivatives/group/swr/stillness_anatomy_2026-09-13/`.
61 sessions, unit = session, min 8 events, rates are RAW (no baseline
subtraction anywhere in this entry).

### Q1 — what "stillness" actually is, and what it was conflating

`still_next_s` = time from the event to the NEXT key press of any kind
(movement or uncover), from the 25 ms button series. It CLASSIFIES the event;
it is not a comparison with baseline, and "the window is more still than
baseline" was never the claim.

It bundles two different problems, which the earlier entries did not separate:

1. **Window contamination.** `correct, first uncovers` has a median gap of
   1.375 s, so the 0–0.5 s test window contains nothing else. `correct, later`
   has 0.425 s, so the same window usually CONTAINS THE NEXT BUTTON PRESS. That
   is a mechanical difference in window content, no theory of brain state
   needed.
2. **Brain state.** Longer quiet → more LIA → more ripples.

Q2 below separates them.

**Limitation of the measure itself:** this is BUTTON stillness. No eye
tracking, no motion capture, no accelerometry. A subject sitting motionless and
thinking hard is indistinguishable from one resting.

### Q2 — inside a long pause, ripples are FLAT

Raw rate in 0.5 s slices from the press that opens each gap ≥ 2.5 s:

| source | 0–0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 |
|---|---|---|---|---|---|
| correct uncovering (n=61, 3,734 ev) | 0.230 | 0.219 | 0.188 | 0.186 | 0.199 |
| movement press (n=43, 1,522 ev) | 0.244 | 0.239 | 0.148 | 0.218 | 0.180 |
| grid end, task-free (n=61, 945 ev) | 0.176 | 0.216 | 0.195 | 0.230 | 0.201 |

**No ramp, no decay — roughly 0.18–0.24 Hz throughout.** Aligning to the END of
the gap gives the same flat picture.

This matters for interpretation. The stillness "dose-response" reported on
2026-09-13 (−0.004 Hz for gaps < 0.5 s rising to +0.075 Hz for gaps ≥ 2.5 s)
is therefore **not** ripples accumulating as a pause lengthens. It is a
difference between two behavioural regimes — quiet stretches sit at ~0.20 Hz
throughout, busy stretches lower — plus, for short gaps, the next press landing
inside the measurement window. Problem 1 above, not problem 2.

### Q3 — task-free rest has the HIGHEST rate in the dataset

SK's idea: after the last D of a grid the subject has finished and the next
grid has not started, so that gap is stillness with no task on it. It works —
1,431 grid endings, 945 with a gap of 2.5–60 s, median 5.92 s, all 61 sessions
contribute (~16 usable per session).

Raw rate 1.5–2.5 s into the gap, by what opened it:

| source | n | rate |
|---|---|---|
| correct uncovering | 61 | 0.1923 Hz |
| movement press | 43 | 0.1988 Hz |
| **grid end (task-free)** | 61 | **0.2157 Hz** |

Paired:

| contrast | Δ Hz | t | p_perm |
|---|---|---|---|
| correct uncovering − grid end | **−0.0234** | t(60) = −2.08 | **0.046** |
| movement press − grid end | +0.0034 | t(42) = +0.14 | 0.894 |
| correct uncovering − movement press | −0.0183 | t(42) = −1.06 | 0.298 |

**The task-free period carries MORE ripples than in-task stillness**, and it is
the highest rate of the three. This is the opposite of "task events drive
ripples" and it is what the classical view predicts: ripples are maximal in
quiet rest with no task demand. It also gives the project a clean reference
rate for "still, nothing to do" — 0.216 Hz — which no task-locked condition in
this dataset exceeds.

Caveat: a grid-end gap still BEGINS with a D uncovering, so it is the aftermath
of a task event rather than a truly neutral epoch; reading it at 1.5–2.5 s is
what keeps any D transient out. The next grid's onset is a median 4.5 s away,
so it is outside the read window for most grids.

### Q4 — event vs stillness-matched control, with no baseline window at all

SK's suggestion, and it is the better design: let the matched pause BE the
reference instead of measuring both against their own pre-event baselines. That
also removes the pre-event stillness imbalance flagged as Limitation 2 in the
previous entry, since no pre-event window enters the statistic.

Restricted to gaps ≥ 2.5 s so both sides are stillness-matched, raw Hz,
sliding cluster test over all positions:

| feedback cell | n | 0–0.5 s Δ Hz | t | p | cluster |
|---|---|---|---|---|---|
| correct, first uncovers | 32 | +0.0202 | +0.58 | 0.56 | none |
| correct, later | 43 | −0.0264 | −0.91 | 0.37 | none |
| error, first uncovers | 7 | −0.0482 | −1.17 | 0.29 | none |
| error, later | 14 | −0.0673 | −2.17 | 0.050 | decrease +0.25…+0.55 s, p = 0.011 |

`correct, first uncovers` — the cell the whole question is about — is +0.020 Hz
and not significant, with no surviving cluster, under the cleanest design
available. The `error, later` cluster rests on 14 sessions and is one of four
tests; not interpreted.

Cost of the design: requiring a ≥ 2.5 s gap on BOTH sides drops n hard
(61 → 32 for correct/first, → 7 for error/first). This is the least biased and
least powered of the three controls run; it agrees with the other two.

### Where the three controls now stand

Stratification (62% of the stage effect attributable to stillness), matched
movement presses (feedback − pause ≈ +0.01 Hz, ns), and now the no-baseline
matched contrast (+0.020 Hz, ns) all point the same way. Against that,
task-free rest carries MORE ripples than any feedback moment. F5 (first-D) has
still not been put through any of these and remains the outstanding job.


## 2026-09-13 (b) — Feedback moments are not distinguishable from pauses of the same length

SK's proposed control, and it is the decisive one: compare feedback moments
against **randomly occurring moments of matched stillness**. Implemented as
`scripts/swr_feedback_vs_matched_stillness.py`. Results in
`data/ephys_humans/derivatives/group/swr/feedback_vs_matched_stillness_2026-09-13/`.

**Control events.** Every MOVEMENT press — a button press that uncovers
nothing, so no information arrives — with its own time-to-next-press.
151,076 of them. Chosen over random time points because a random time point
has no motor onset and would differ from an uncovering in two ways at once;
N1 already showed movement presses carry no ripple modulation on their own.

**Design.** Stratify both feedback events and movement presses by time to the
next key press of any kind, then contrast feedback − pause WITHIN each bin,
paired by session. Same session, same stillness, same motor act; the only
remaining difference is whether information arrived. Stratification rather
than 1:1 matching, so nothing is sampled or discarded (the ≥2.5 s bin holds
only 1,608 movement presses, so 1:1 matching would need replacement there).

### 1. How much does stillness alone do? A lot.

Movement presses vs their own baseline, 0–0.5 s, by how long the subject then
sat still — no feedback anywhere in this analysis:

| stillness after the press | Δ rate 0–0.5 s | t | p_perm |
|---|---|---|---|
| 0–0.5 s | −0.0043 | −0.78 | 0.44 |
| 0.5–1 s | +0.0062 | +1.11 | 0.28 |
| 1–1.5 s | +0.0183 | +1.39 | 0.17 |
| 1.5–2.5 s | **+0.0607** | **+3.09** | **0.0023** |
| ≥ 2.5 s | **+0.0751** | **+3.03** | **0.0026** |

**A pause with nothing in it produces +0.06 to +0.075 Hz — larger than the
entire correct/first-uncovers effect (+0.056 Hz).** Any event class enriched
for long pauses inherits a rise of this size for free.

### 2. Is feedback more than that? No.

Feedback − matched pause, stillness-standardised across bins:

| feedback cell | n | Δ Hz | t | p_perm | cluster |
|---|---|---|---|---|---|
| correct, first uncovers | 58 | +0.0105 | +0.58 | 0.56 | none |
| correct, later | 61 | −0.0020 | −0.19 | 0.85 | none |
| error, first uncovers | 58 | −0.0057 | −0.61 | 0.54 | none |
| error, later | 40 | +0.0016 | +0.13 | 0.90 | none |

**None of the four cells differs from a movement press followed by the same
amount of stillness**, at any bin-standardised estimate or in any surviving
cluster. Per-bin, `correct, first uncovers` runs +0.016, +0.021, +0.015,
+0.007, +0.010 Hz — consistently positive but never significant and an order of
magnitude smaller than the stillness effect it sits on.

Two of twenty bin-level tests reach p < 0.05 — `error, first uncovers` at
1.5–2.5 s (+0.114, p = 0.006, n = 17) and `error, later` at ≥2.5 s (−0.116,
p = 0.015, n = 12). One is expected by chance, they have opposite signs, both
have the smallest n in the table, and neither survives standardisation. They
are noise.

### What this means, and what it does not

The feedback-locked ripple rise in this dataset is, as far as this control can
tell, the pause. Three independent lines now agree: stratification removes it,
direct standardisation attributes 62% of the stage difference to stillness, and
a matched non-feedback press reproduces it entirely.

**This does not by itself overturn F5** (ripples after the first uncovering of
D), which was the one claim stated before any analysis. F5 was tested on a
different contrast — first-D against its own baseline with a sliding cluster
test — and has NOT been put through this control. It should be, and that is
the obvious next job: first-D is exactly the kind of event followed by a long
pause, so it is the most exposed claim in the project, not the least.

**Limitation 1 — the control is not neutral either.** A movement press followed
by 2 s of stillness is not a blank moment: the subject may have arrived
somewhere and stopped to think. If deliberation pauses carry ripples, the
control absorbs genuine signal and the test is conservative. Separating pause
TYPES rather than pause LENGTHS is the way past this: VTE-like deliberation
pauses are theta-rich and ripple-poor, quiet-rest pauses the reverse. The LFP
needed for that split exists (`continuous.npy` per session) but no theta
measure is in the bundle yet.

**Limitation 2 — matching on following stillness leaves PRECEDING stillness
unbalanced.** Median time since the previous press, within each
following-stillness bin:

| bin (time to next press) | correct, first | correct, later | movement press |
|---|---|---|---|
| 0–0.5 s | 0.275 | 0.400 | 0.425 |
| 0.5–1 s | 0.312 | 0.500 | 0.550 |
| 1–1.5 s | 0.350 | 0.625 | 0.800 |
| 1.5–2.5 s | 0.350 | 0.738 | **1.025** |
| ≥ 2.5 s | 0.400 | 0.500 | **1.200** |

A movement press with a long pause after it tends to have a long pause before
it too — it is an isolated press in an already-idle stretch — whereas a
feedback event with a long pause after it was reached by active walking. The
baseline window sits at −1.6 to −1.1 s, so the control's baseline should be the
more ripple-rich one, which would DEFLATE the control's (window − baseline) and
therefore INFLATE feedback − pause. The observed feedback − pause is ~+0.01 Hz
and non-significant even with that bias pushing in its favour, so the null
conclusion is the safe direction. This is a reasoned direction, not a measured
one — the rigorous fix is to stratify on preceding AND following stillness
jointly, which is the next thing to build if this control is to be quoted.


## 2026-09-13 — Positive feedback early vs late: 62% of it is stillness

Follow-up to 2026-09-11. SK pointed out that the contrast she is actually
looking at is not the interaction I had tested: it is **positive feedback
early vs positive feedback late** (correct only, across stages), not
`(correct − error) × stage`. Tested directly, with the stillness controls the
project's own F1 demands.

New script: `scripts/swr_stillness_control.py`. Results in
`data/ephys_humans/derivatives/group/swr/stillness_control_correct_2026-09-11/`.
Bundle `swr/bundle` (2026-09-08), 61 sessions, unit = session, min 10 events
per session per cell per bin, 0–0.5 s vs the same trial's baseline.

### Which number is which

Worth stating because the two are easy to conflate, and the `feedback_stage`
figure shows the first while the question is about the second:

| quantity | value | verdict |
|---|---|---|
| `correct, first uncovers` vs **its own baseline** (the `**` in the figure) | +0.056 Hz, t = +2.76, p_perm = 0.006; cluster +0.25…+0.75 s, p = 0.006 | **holds**, survives cluster correction |
| `correct, first` − `correct, once known` | +0.0396 Hz, t = +2.14, p_perm = 0.034 | one of 3 windows, no cluster |
| `correct, first` − `correct, while learning` | +0.0249 Hz, t = +1.16, p_perm = 0.257 | no |
| `correct, first` − `correct, later` (pooled) | +0.0360 Hz, t = +2.03, p_perm = 0.044 | one of 3 windows, no cluster |

A condition can beat its own baseline strongly and still not beat another
condition: the second test is paired and has to clear the other condition's
rise as well.

### Stillness is a five-fold effect on this very window

Measured on the correct uncoverings themselves, pooled over stages so it
carries no stage information, binned by time to the next key press of any kind:

| stillness after the event | Δ rate 0–0.5 s |
|---|---|
| 0–0.5 s | +0.010 Hz |
| 0.5–1 s | +0.015 Hz |
| 1–1.5 s | +0.029 Hz |
| 1.5–2.5 s | +0.052 Hz |
| ≥ 2.5 s | +0.039 Hz |

And the two stages sit in completely different places on that curve:

| stillness bin | % of `first uncovers` | % of `later` |
|---|---|---|
| 0–0.5 s | 16.9 | 58.2 |
| 0.5–1 s | 18.4 | 28.0 |
| 1–1.5 s | 17.8 | 6.4 |
| 1.5–2.5 s | 23.0 | 3.5 |
| ≥ 2.5 s | 23.9 | 3.8 |

### Stratifying removes most of it

Same contrast computed INSIDE each stillness bin, so like is compared with
like:

| stratum | n | Δ Hz | t | p_perm |
|---|---|---|---|---|
| unstratified | 61 | +0.0360 | +2.03 | 0.044 |
| 0–0.5 s | 32 | +0.0281 | +0.74 | 0.489 |
| 0.5–1 s | 35 | +0.0059 | +0.17 | 0.874 |
| 1–1.5 s | 38 | +0.0214 | +0.73 | 0.472 |
| 1.5–2.5 s | 33 | +0.0242 | +0.71 | 0.496 |
| ≥ 2.5 s | 38 | −0.0281 | −0.50 | 0.629 |
| **stillness-standardised** | **58** | **+0.0169** | **+0.72** | **0.488** |

No surviving cluster for the standardised contrast at either width.

**Direct standardisation.** Taking the rate-vs-stillness curve above (measured
with both stages pooled, so it is stage-blind) and applying each stage's own
stillness distribution to it:

- observed: +0.0360 Hz (t = +2.03)
- predicted by the stillness imbalance alone: **+0.0223 Hz (t = +2.70)**
- residual: +0.0136 Hz, t = +0.89, **p = 0.376**
- **stillness accounts for 62% of the observed difference**, and the remaining
  38% is not distinguishable from zero.

Caveat, stated because it cuts the other way: stratifying costs sessions
(61 → 32–38 per bin), so "not significant within bins" is partly power. The
informative part is that the effect SIZE drops with it (+0.036 → +0.017
standardised), which power loss alone would not do.

### Interpretation, and why this is not automatically a confound

Stillness may be a MEDIATOR rather than a confounder: uncovering a reward for
the first time may cause the pause, and the pause carries the ripples. That is
a real possibility and controlling for stillness then removes part of the
causal path. But the claim that survives in that case is "discovering a reward
makes people pause, and pauses carry ripples", not "the hippocampus signals
discovery". Distinguishing them needs a comparison against pauses of the same
length that contain no feedback — see the next entry.


## 2026-09-11 — Valence × stage is a stillness effect; ripples do not predict later errors

Two questions from SK about the `feedback_stage` panel at `--unit=session
--min_events=15`: (1) the six cells look crossed — positive feedback raises the
rate early in a grid, negative feedback late — can that be tested as an
interaction? (2) does a higher ripple rate early predict fewer mistakes later?

Two new scripts, both reusing the existing test functions rather than
re-implementing them:
`scripts/swr_valence_stage_interaction.py`, `scripts/swr_ripples_predict_errors.py`.
Two new helpers in `mc/analyse/ripples.py`: `contrast_profiles` and
`hotelling_signflip`.

Results: `data/ephys_humans/derivatives/group/swr/valence_x_stage_2026-09-11/`
and `.../ripples_predict_errors_2026-09-11/`. Bundle `swr/bundle` (2026-09-08),
61 sessions, 41 subjects, 180 derivations, 64,760 ripples.

### How the interaction is built

Baseline subtraction is linear, so a weighted combination of RAW per-session
rate profiles, baselined afterwards, is identical to combining already-baselined
profiles (checked: max abs difference 2.2e-16). An interaction contrast
therefore passes through `rip.baseline_subtract`, `rip.window_test` and
`rip.sliding_window_test` unchanged — same sign-flip null, same cluster
correction, no second code path.

### 1. The interaction exists in the right direction and does not survive

0–0.5 s vs the same trial's baseline, session as the unit, min 15 events/cell:

| contrast | n | Δ Hz | t | p_perm |
|---|---|---|---|---|
| correct − error, first uncovers | 61 | +0.0555 | +3.03 | **0.0031** |
| correct − error, while learning | 33 | +0.0326 | +1.27 | 0.224 |
| correct − error, once known | 42 | −0.0042 | −0.16 | 0.866 |
| valence × stage, linear (first − once known) | 42 | +0.0503 | +1.41 | 0.161 |
| valence × stage, quadratic | 27 | +0.0284 | +0.36 | 0.731 |
| valence × stage, pooled (first − later) | 56 | +0.0486 | +1.83 | 0.073 |

Omnibus 2 × 3 (linear + quadratic jointly, Hotelling T² with sign-flip null,
sessions supplying all six cells): **F(2,25) = 0.08, p_perm = 0.924, n = 27**.

Sliding cluster test, both widths: the simple effect at *first uncovers*
survives (+0.25 to +0.75 s, p = 0.006 at 0.3 s; +0.15 to +0.75 s, p = 0.017 at
0.5 s). **No interaction contrast produces any surviving cluster.**

The pooled interaction reaches +0.0685 Hz, t = +2.34, p_perm = 0.021 in the
0.5–1.0 s window, but that is one of three windows inspected (Holm-adjusted
p = 0.063) and no cluster survives, so it is not a result.

Why the interaction is so much weaker than the eye suggests: it is paired, and
pairing costs sessions. `error, while learning` is estimable in 33 sessions and
`error, once known` in 42, so the 2 × 3 crossing has n = 27. Pooling the two
later stages at the EVENT level (before the min-events filter) recovers n = 56,
and that is the best-powered version available.

### 2. The interaction is a stillness artefact

F1 says ripple rate rises with stillness. Stillness — time from an uncovering to
the next key press of ANY kind, movement included, taken from the 25 ms button
series via `swr_probes.press_times` — is wildly unbalanced across these cells:

| cell | median time to next press |
|---|---|
| correct, first uncovers | **1.375 s** |
| correct, while learning | 0.525 s |
| correct, once known | 0.425 s |
| error, first uncovers | 0.450 s |
| error, while learning | 0.575 s |
| error, once known | 0.625 s |

The same interaction contrast on the BEHAVIOUR: **+1.310 s, t(60) = +11.34,
p_perm < 1e-4** (2-stage) and +1.575 s, t = +8.11 (3-stage). The behavioural
interaction is an order of magnitude more significant than the neural one and
has the identical shape. After a correct uncovering on the first traversal the
subject stops and looks; in every other cell they press on within half a second.

Sweeping a common stillness criterion applied identically to all cells
(0–0.5 s window, pooled 2-stage crossing):

| min stillness | valence effect, first uncovers | interaction (first − later) |
|---|---|---|
| none | +0.0555 (n=61, p=0.003) | +0.0486 (n=56, p=0.073) |
| ≥ 0.5 s | +0.0513 (n=59, p=0.006) | +0.0410 (n=47, p=0.170) |
| ≥ 1.0 s | +0.0514 (n=39, p=0.079) | **−0.0066 (n=30, p=0.879)** |
| ≥ 1.5 s | +0.0188 (n=16, p=0.634) | −0.0240 (n=12, p=0.564) |
| ≥ 2.0 s | +0.0223 (n=6, p=0.778) | −0.0153 (n=5, p=0.935) |

The dissociation matters. The **simple effect at first uncovers keeps its
effect size** through ≥ 1.0 s (0.0555 → 0.0513 → 0.0514; only n and therefore p
degrade), so it is not obviously stillness. The **interaction does not** — it
falls to zero and changes sign while n is still 30. Reading the interaction as
neural is not supported.

Caveat on the sweep: thresholding changes cell composition as well as
equalising stillness, and n falls steeply because only 11% of `error, first
uncovers` events are followed by ≥ 1 s of stillness (vs 65% of `correct, first
uncovers`). Distribution matching within session × stage would preserve more n
and is the obvious next step if this is to be pushed further.

### 3. Ripple rate does not predict later errors; stillness does

Unit: the grid (1,381–1,429 grids, 61 sessions, median 24 grids/session).
Spearman within session across grids → Fisher z → t across sessions; null =
1,000 shuffles of the outcome across grids WITHIN session, run through the same
`group_statistic`. Predictor = ripple rate 0–0.5 s minus the same trial's
baseline, averaged over the correct uncoverings of that grid in that stage.

Outcome `errors_total` (every wrong uncovering in the grid):

| predictor | ripples ρ | p_perm | stillness ρ | p_perm |
|---|---|---|---|---|
| all correct uncoverings | +0.019 | 0.422 | **+0.245** | **0.001** |
| first uncovers | −0.000 | 0.988 | **+0.126** | **0.001** |
| while learning | +0.021 | 0.562 | **+0.350** | **0.001** |
| once known | −0.001 | 0.946 | **+0.147** | **0.001** |

Outcome `errors_after` (wrong uncoverings strictly after the last predictor
event — the clean predictive version for `first uncovers`): ripples +0.016,
p = 0.598. First traversal vs the mean of the two later stages: Δz = −0.010,
t(60) = −0.25, p_perm = 0.806 (errors_total); Δz = −0.037, p_perm = 0.592
(errors_after). Ripple rate with stillness partialled out within session:
unchanged, all |ρ| ≤ 0.024.

So the answer is no, and it is a well-powered no: 1,381 grids, and the same
design detects a ρ of +0.13 to +0.35 for stillness at p = 0.001. The sign of
the stillness effect is worth noting — **more pausing goes with MORE errors**,
i.e. it indexes how hard the grid was, not how well it was learned.

`errors_after` is near-degenerate for the `all` and `once known` predictors by
construction (almost no errors follow a grid's last correct uncovering), which
is why their n drops to 26 and 23 sessions; those two cells should be ignored.

### Status

Nothing here was predicted in advance; all of it is exploratory. What survives:
the **simple** valence effect at the first traversal, which was already visible
in `feedback_stage`. What does not: the valence × stage interaction, and any
ripple–behaviour prediction.


## 2026-09-10 — RSA jobs were being killed on walltime; raised -T and made them resumable

**Cause, confirmed from the job log:** `JOB 2348457 ... CANCELLED AT
2026-09-09T15:17:18 DUE TO TIME LIMIT`. `submit_RSA_instruction_epochs.sh`
passed `fsl_sub -T 30`, but one RSA job fits 28 whole-brain searchlight OLS
models (21 single + 7 combo) over ~126k searchlights. Jobs died about two
thirds of the way through, leaving `results/` folders that looked plausible:
sub-01 `instr_see-A-first` held 20 of 41 beta maps -- every single model in
config order up to `D_rew`, then nothing, and no combo maps at all. The
`.pdf`/`.png` panels are drawn before any OLS runs, so the folder looked full.
`check_RSA_ran.py` was right to list those runs; the diff against
`expected_map_names` showed 21 missing and 0 unexpected, i.e. what was on disk
was a strict prefix of what should be.

**Fix 1 -- time.** `-T 30` -> `jobTime="${FSLSUB_T:-240}"`, overridable per
submission.

**Fix 2 -- resume.** `fMRI_run_RSA_instruction.py` now skips any map whose
`_beta` / `_t_val` / `_p_val` volumes are all present and non-empty, so a job
that still runs out of time picks up where it stopped instead of starting over.
Without this a resubmission recomputes all 28 fits and dies at the same
wall-clock point, so it never converges.

Skipping is gated on settings, because otherwise a rerun silently mixes two
analyses in one folder:
  * the settings summary is only written at the END of a run, so a killed job
    never leaves one -- the script now writes `{sub}_run_settings.json` at the
    START and resumes only when it matches;
  * a folder left by a run that FINISHED has no such file but does have a
    settings summary, which is compared against instead (same settings under
    their own key names);
  * a folder with maps and neither file cannot be verified and is NOT resumed,
    unless `"resume_unverified": true` is set in the config. That flag is
    deliberately narrow: it only applies when there is no evidence at all, so
    it cannot trust maps from a different completed analysis.
  * `"resume": false` disables it entirely.
Combos are skipped only when ALL of a combo's maps exist (one OLS produces them
all); the regressor-correlation record is computed either way, so the settings
summary keeps its collinearity entry for skipped combos. The summary records
`resume_enabled` and `resumed_maps`, so a folder completed across several jobs
says so.

**Audit location.** Both audits now write to `analysis/logs_mid_sept/` rather
than `derivatives/group/` -- they are logs, not results. On a laptop
`$analysisDir` is the repo itself, so the shell wrapper redirects there to the
data tree instead; nothing generated ever lands in the repo.

Verified by executing the real resume block out of the script against eight
folder states: fresh, matching rerun, partial map (beta only, no t_val),
changed settings, no run-settings file, `resume_unverified`, `resume: false`,
and an old completed run with different settings plus `resume_unverified` (the
one that must still refuse). All behaved as intended.

## 2026-09-09 — RSA audit emits a per-wrapper resubmission plan

`check_RSA_ran.py` now tracks all five pipeline stages and writes
`resubmit_plan.sh`: the exact commands that would fill every gap, per wrapper,
in pipeline order.

    results/                fMRI_run_RSA_instruction.py   per subject x epoch
    smoothed/               smooth_subject_space.py       per subject x epoch
    standard-space-smooth/  applywarp wrapper             per subject x epoch
    group_..._glmbase_{epoch}/          merge_subj_to_group.sh        per epoch
    group_..._glmbase_{epoch}_cropped/  mask_subj_by_missingvoxels.sh per epoch

Two gating rules keep the plan honest rather than merely complete:
  * a stage only lists work whose PREDECESSOR is complete, so nothing is
    submitted that would read half-written inputs;
  * a run queued for an RSA rerun is excluded from the downstream stages
    entirely -- its maps are about to be rewritten, so smoothing them now is
    wasted queue time. (Caught in testing: a subject whose maps were all
    present but whose settings summary was missing was being offered for
    smoothing and RSA resubmission at once.)
Group stages additionally require the epoch to have NO subject-level gap left,
because merging a partial set gives the wrong volume count and
mask_subj_by_missingvoxels.sh then rejects it on `required_n`.

Smoothing is submitted per (subject, epoch) through
wrapper_python_fMRI_RSA_clean_config.sh rather than through
wrapper_smooth_stat_maps_subj.sh: smooth_subject_space.py has no
skip-if-exists, so the wrapper's loop would re-smooth every subject of the
epoch. That needs a per-epoch smoothing config, which the audit now writes
(`smooth5_config_{epoch}.json`) so concurrent jobs cannot rewrite each other's
`regression_version` in the shared file.

The stages are dependent, so the plan is not a script to run top to bottom: run
one stage, wait for the queue, re-run the audit for the next plan.

Verified on a synthetic tree through the whole chain: mixed subject states ->
RSA + smoothing + standard-space plan; all subjects complete -> merge offered
and crop still gated; merged present (as gunzipped .nii, which the checker
accepts) -> crop offered and merge gone.

## 2026-09-09 — RSA audit now checks all three stages per model

`check_RSA_ran.py` only asked whether the RSA had finished, per (subject,
epoch). It now walks the whole per-subject pipeline, per MODEL, at each stage:

    results/                {map}_beta.nii.gz          fMRI_run_RSA_instruction.py
    smoothed/               smooth_fwhm5_{map}_beta.nii.gz      smooth_subject_space.py
    standard-space-smooth/  smooth_fwhm5_{map}_beta_std.nii.gz  applywarp wrapper

Only the beta map is tracked: it is what the group merge collects
(`*beta_std.nii.gz`) and what `loso.py` reads, so a missing beta is what
actually breaks the pipeline; t_val / p_val are written in the same call.

`expected_map_names` derives the full set of maps from the config, mirroring
the output naming of fMRI_run_RSA_instruction.py: `{model}`, or
`{model}_within` / `_across` where `single_model_scopes` entitles a model to
several scopes, plus `{REGRESSOR}-{combo}` per combo regressor (with the same
scope suffix on the combo name, and the `block` nuisance when
`add_block_nuisance` is set). For rsa_instruction_cumulative_rew.json that is
**41 maps** per (subject, epoch) — 21 single + 20 combo.

New statuses distinguish what actually needs rerunning: `NOT_STARTED` /
`RESULTS_INCOMPLETE` / `RERUN_CHANGED` need the RSA resubmitted and go on
todo_rsa.txt, while `SMOOTHED_INCOMPLETE` and `STANDARD_INCOMPLETE` mean the
RSA is fine and only the cheap downstream wrapper has to be rerun — those are
reported separately and deliberately kept OFF the submission list.
`RESULTS_INCOMPLETE` also covers "all maps present but no settings summary",
i.e. the run died before its last action.

Outputs gain a per-stage table (complete / partial / none / maps missing, with
the script that produces each stage) and `missing_maps.txt`, one line per
individual map absent from disk. settings.json records the expected map names
and the stage tally.

Verified on a synthetic tree covering seven states — all stages complete,
smoothed partial, standard partial, results partial, nothing at all, results
without a summary, and settings changed. Each classified correctly, the stage
tallies reconcile exactly (46 = 5 + 41, 125 = 2 + 3x41, 165 = 1 + 4x41), and
the two downstream-only gaps stayed off todo_rsa.txt.

## 2026-09-09 — group inference runs per TR or per condition (`--axis`)

`per_TR_loso.py` assumed its third axis was the ordered seconds of the
instruction period: `--trs` was parsed with `int()` and substituted into
`dir_pattern.format(tr=tr)`, so the instruction-epoch folders
(`group_RSA_instr_cumrew_glmbase_instr_see-A-first_cropped`) could not be read
at all.

**`--axis {tr,condition}`.** `tr` (default) is unchanged: levels from `--trs`,
peaks reported as TR numbers, figure drawn as a timecourse with the reward
schedule. `condition` takes levels from `--conditions` (the epoch GLM names)
and substitutes them as `{condition}` / `{tr}` / `{level}` in `--dir-pattern`.

**The statistics are identical in both modes** — the same max-t sign-flip
permutation over voxels x levels, one family per mask, through the same
`tstat` / `null_max_t` (CLAUDE.md rule 4). Only reporting differs: `peak_TR`
fields hold the condition name, and `settings.json` records `axis`, `levels`
and an `axis_note` stating that a condition axis must not be read as a
timecourse.

**Plotting is deliberately different per axis.** A condition axis is drawn as
points with error bars, rotated condition labels, no joining line and no reward
schedule — a line between unordered conditions would assert a progression the
design does not contain. A non-numeric axis is forced to that style whatever is
requested, so a mislabelled call cannot draw a timecourse through condition
names. Models are dodged horizontally (x-offset only, no value altered) so
overlapping points stay readable. `--mode plot` reads the axis kind back from
the run's `settings.json`, so old per-TR folders still plot as timecourses.

**Two bugs found while doing this:**
- `CHANNEL_COLOURS` / `CHANNEL_LABELS` still used the pre-rename model names
  (`curr_rew`...), while the RSA now writes `A_rew`...`D_rew`. Every map would
  have plotted grey and the default "four reward channels" selection would have
  matched nothing, silently falling back to plotting all 14 models. Rekeyed to
  the current names with the project state colours; `LEGACY_CHANNEL_ALIASES`
  keeps pre-rename result folders colouring correctly. The cumulative channels
  (AB/ABC/ABCD_rew) stay grey rather than inventing a colour convention.
- `summary_row` reported `loso_peak_TR` as the POSITION along the axis, not the
  level: with `--trs 4,5,6` it printed 0/1/2. Now the level.

Verified end to end on synthetic 32-subject volumes with a planted blob: TR mode
recovers the planted TR, condition mode recovers the planted condition, both
figures render, and `--mode plot` picks the right style from settings.json.

## 2026-09-08 — UCLA's own anatomical labels vs our atlas ROIs (`ucla_label_vs_atlas_roi.py`)

UCLA report that their electrode localisation is very precise, so the labels
they ship were used as an independent check on the ROIs
`cell_to_roi_july26.py` assigns. That script reads only the MNI coordinate out
of `sub-{NNN}_localizations.xlsx` and discards every label column; this
compares those discarded labels against `atlas_roi` / `alt_final_roi`. **Read
only — the ROI table is unchanged.**

140 UCLA cells on 24 microwire bundles, joined on
`(Subject Label, source_electrode)` — the same bundle the coordinate came
from, so the two verdicts describe the same contact. Seven UCLA columns were
mapped onto our ROI vocabulary: `ASHS_ABC` (subject's own T2, MTL subfields),
`aparc+aseg` / `aparc.DKTatlas+aseg` (subject's own T1, FreeSurfer native
space), `Anat` and `AnatMacro_1` (SPM Anatomy toolbox), `NMM`
(Neuromorphometrics), and `region` (the implantation-target code). Labels that
make no regional claim (`Unknown`, `*Cerebral-White-Matter`, `(extra-axial)`)
are scored as "no verdict", not as disagreements. `HC_anterior`/`HC_mid` is our
own y = -21 split and no UCLA column encodes it, so agreement is scored on the
collapsed `HC`.

**Headline: 114/140 cells (81.4 %) and 20/24 bundles agree** with at least one
UCLA column. Per column, over the contacts where that column has a verdict:
NMM 84 %, AnatMacro_1 65 %, aparc+aseg 65 %, DKT 61 %, Anat 54 %,
implantation-target code 48 %. ASHS agrees 11/12 where it is defined.

**The disagreements are not spread out — they are exactly the neighbourhood
fallback.** Splitting by `atlas_reason`:

| assignment mode | cells with a UCLA verdict | agree | % |
|---|---|---|---|
| exact voxel | 104 | 103 | **99.0** |
| `neighbor@1/2/3mm` probe | 28 | 11 | **39.3** |

The single exact-voxel disagreement is a `leftover` cell (Heschl's gyrus,
dropped anyway). So wherever the cell's own voxel lands in an atlas region,
UCLA's independent labelling confirms us essentially always; the risk is
concentrated in rule 12, where the voxel is white matter and we probe outwards.

**What this costs the analysis ROIs.** 93 UCLA cells survive into analyses on
13 bundles; 12 bundles are confirmed, one is not:

* **`RPv-micro` (UC3-0582, 17 cells, currently `HC_mid`) is almost certainly
  pulvinar.** Coordinate (12.2, -31.6, 3.9), assigned via
  `hippocampal_subfield_split_y-21.0_neighbor@2mm`. All four UCLA columns say
  thalamus (`Right-Thalamus`, `Right Thalamus Proper`, `Thal: Temporal`,
  `R Thalamus`) and the electrode is named `RP` = right pulvinar. This is
  **half of the UCLA `HC_mid` cells** (17 of 35).
* `RPHG_micro-1` (UC3-0573, 1 cell, `HC_mid`) — ASHS `Right PHC`, both aparc
  columns `parahippocampal`, AnatMacro `R ParaHippocampal Gyrus`. Also a
  `neighbor@1mm` assignment. Probably PHC.

Per analysis ROI, cells confirmed by at least one UCLA column: PCC 22/22,
mPFC 4/4, EC 3/3, HC_anterior 29/29, **HC_mid 18/35**.

**EC survives the strictest test available.** The 3 EC cells (`LEC-micro`,
UC2-0578) are confirmed by ASHS (`Left_ERC`, segmented on that subject's own
T2) and by SPM Anatomy (`Entorhinal Cortex`) — the two most anatomically
specific columns UCLA ship.

**Caveat: `aparc+aseg` is not uniformly trustworthy, so its 65 % is a floor.**
Per-subject concordance between `aparc+aseg` and `NMM` on the same contacts is
100 % (UC3-0559), 100 % (UC3-0577), 86 % (UC3-0582), 50 % (UC2-0576, UC3-0573)
but **20 % for UC2-0578**, whose column labels an entorhinal contact at
z = -38 as `ctx-lh-insula`, an amygdala contact as `Left-Putamen` and a
hippocampal one as `Left-Pallidum` — anatomically impossible, all displaced the
same way. That subject's FreeSurfer columns look mis-registered; its ASHS and
Anat columns behave normally. So "UCLA's method is precise" holds for their
MTL segmentation, less so for the whole-brain FreeSurfer columns in every
subject.

**Mechanism: there are two neighbourhood probes and only one is dangerous.**

* **Path A — the HC probe inside rule 4** (`hippocampal_subfield_..._neighbor@Nmm`).
  It fires whenever the exact voxel is not hippocampus, and returns *before
  rules 5-11 are ever consulted*, so it can override a perfectly good
  exact-voxel verdict from a lower-priority rule. Its design comment scopes it
  against rule 5 only ("otherwise leaves subicular voxels to be captured by the
  coarser parahippocampal-gyrus rule despite being anatomically hippocampal") —
  but it sits above rules 5-11 and therefore beats *all* of them, including
  Thalamus and Amygdala. That is the RPv bug: at (12.2, -31.6, 3.9) the exact
  voxel is HO-subcortical `Right Thalamus`, rule 10 would have caught it, but
  rule 4's probe finds Juelich cornu ammonis 2 mm away first.
* **Path B — rule 12 proper** (`atlas_neighbor@Nmm`) is reached only when *no*
  rule matched at the exact voxel, so it can only promote `leftover`. It cannot
  contradict an exact-voxel label and is structurally safe.

**Path A is confined to HC by construction** — it lives inside rule 4 and can
only return `HC_anterior`/`HC_mid`. So no other ROI is exposed to this failure
mode. Probe cells in analysis ROIs, whole table: HC_mid 56 (all path A),
HC_anterior 11 (all path A), mOFC 23, mPFC 13, PCC 4 (all path B), **EC 0**.

Re-querying the exact voxel of all 67 path-A cells across all three sites:

| overridden exact-voxel label | HC_ant | HC_mid | sites |
|---|---|---|---|
| PHC — **intended** override | 4 | 39 | Baylor 41, UCLA 1, Utah 1 |
| Thalamus — **not** intended | 0 | 17 | UCLA (RPv) |
| Amygdala — **not** intended | 5 | 0 | Utah |
| white matter — probe justified | 2 | 0 | Utah |

The 43 PHC cells are the documented purpose of rule 4c (the subiculum sits in
what HO calls parahippocampal gyrus) and are **not** errors. The 17 Thalamus
and 5 Amygdala overrides are side effects of the probe outranking rules 10 and
9, which its own rationale never claims.

**Correction to the note above on `RPHG_micro-1`:** it is *not* clearly wrong.
Its exact voxel is HO `Parahippocampal Gyrus, posterior division` and the probe
found Juelich subiculum at 1 mm — the intended rule-4c behaviour — and UCLA's
own SPM Anatomy column independently says `Subiculum` there. ASHS says
`Right PHC`; the two disagree with each other, not with us. **`RPv` (17 cells)
is the only UCLA assignment contradicted by every available source.**

**Not done / open:** the 17 pulvinar cells were left in `HC_mid` — changing the
ROI table is a separate decision, not a side effect of an audit. The 5 Utah
amygdala overrides share the failure mode but have no independent labels to
check against. A minimal fix would be to make rule 4c yield to HO-subcortical
Thalamus/Amygdala at the exact voxel while keeping its PHC precedence; that
would move 17 cells (UCLA) out of HC_mid and 5 (Utah) out of HC_anterior, and
touch nothing else. Baylor and Utah ship no label columns, so the UCLA
agreement check itself covers 140 of 984 cells.

Outputs -> `ephys_humans/derivatives/ROI_assignment/ucla_label_agreement_2026-09-08/`
(`per_cell_labels.csv`, `per_bundle_labels.csv`, `agreement_summary.csv`,
`agreement_by_assignment_mode.csv`, `per_analysis_roi_verdict.csv`,
`analysis_bundles_verdict.csv`, `freesurfer_column_sanity.csv`,
`confusion_*.csv`, `settings.json`).

## 2026-09-07 — instruction RSA submitted only where it can run (`check_RSA_ran.py`)

`submit_RSA_instruction_epochs.sh` used to submit 33 subjects x 11 epochs
unconditionally. Most of those jobs either cannot succeed or need not run: the
RSA opens `glm_instr_<epoch>_pt0{1,2}.feat/stats/pe*.nii.gz` directly, so a
subject whose epoch GLM never finished produces a job that dies on a missing
PE, and a subject already analysed just recomputes the same searchlight maps.

**`check_RSA_ran.py`** decides the list first, per (subject, epoch):
`READY` (submit), `RERUN_CHANGED` (a result exists but with different settings —
submitted, overwriting the old maps, unless `--skip-changed`), `DONE` (skip),
`GLM_NOT_READY`, `MISSING_INPUT` (modelled EVs / example_func / searchlight
mask). GLM readiness calls `check_GLMs_ran.check_one`, so "complete" means the
same thing in the FEAT audit and here; `PROMOTE_TWIN` deliberately does **not**
count as ready, because the finished run is then in a `+` twin while the base
directory the RSA reads is broken.

**"Already done" means done with THESE settings.** The marker is
`{sub}_settings_summary.json`, which the RSA writes as its very last action, so
its presence means the whole run finished rather than some of the maps. Its
contents are compared field by field (`COMPARED_KEYS`: EV_string,
regression_version(+_full), TR, RDM_version, smoothing, fwhm, searchlight_mask,
data_rdm_scope, models_evaluated, run_single/combo_models, combo_models). A
field missing from an older summary counts as a difference — the settings
cannot be shown to match, so it is rerun rather than assumed done.

Writes `todo_rsa.txt`, `report.txt` and `settings.json` to
`derivatives/group/rsa_audit_<name>_<date>/`, plus one per-epoch config
snapshot (`<base>_<epoch>.json`, regression_version = the epoch GLM, TR = null)
next to the base config.

**Bug found and fixed:** `wrapper_python_fMRI_RSA_clean_config.sh` hardcoded
`fMRI_run_RSA_without_rsatoolbox_clean.py` while
`submit_RSA_instruction_epochs.sh` was already passing the script name as `$3`.
Every instruction-epoch RSA job submitted through it therefore ran the WRONG
python script. The wrapper now takes `$3`, defaulting to the old name so
existing callers are unaffected, and the submit script calls the repo copy by
path rather than the unversioned copy sitting loose in `$analysisDir`.

Jobs run in conda env `spyder-env` via `fsl_sub -T 30`. `DRYRUN=1` prints
without submitting; a todo list can be passed explicitly.

## 2026-09-07 — instruction-phase GLM audit: '+' twins handled, reruns gated

FEAT never overwrites an output directory: it appends `+` and leaves the old
one in place. Repeated submissions of the instruction-epoch GLMs (partly after
quota failures) left up to three generations of the same run side by side
(`glm_instr_see-A-first_pt01{,+,++}.feat`) in any mix of complete and broken,
while `my_RSA.py:405` only ever reads the plain `..._pt01.feat`. So a finished
GLM could sit in a `++` twin while the RSA read a truncated base, and
`check_GLMs_ran.py` reported the run as missing.

**`check_GLMs_ran.py` now audits the whole set of `+` generations per run**
instead of the base directory alone, and sorts the grid into: `OK`,
`DUPLICATES` (base complete, twins redundant), `PROMOTE_TWIN` (base
missing/broken but a twin is complete — a rename, *not* a rerun), a failure
status (no complete copy anywhere → rerun), or `NO_EV_FOLDER`/`NO_DRAFT_FSF`
(inputs missing, cannot be submitted at all). The per-folder completeness test
is unchanged (design.mat `/NumWaves` → all PEs non-empty → FILM end-markers →
`task-to-EV.txt` agrees with the design). It writes `report.txt`,
`cleanup_feat_dirs.sh`, `todo_submit.txt` and `settings.json` to
`derivatives/group/glm_audit_<version>_<date>/`.

**No data is deleted by the audit.** `cleanup_feat_dirs.sh` is a dry run unless
given `--apply`, and re-validates every path at the moment of deletion against
`*/derivatives/sub-*/func/glm_*.feat` containing a `stats/` or `design.fsf`,
refusing anything else (verified against injected paths for the derivatives
root, a `func/` dir, an `EVs_*` folder, a non-FEAT `glm_*.feat` and a `..`
traversal — all refused under `--apply`). Promotion moves the broken base aside
first and removes it only once the twin is in place, so an interrupted run
cannot leave the good data deleted.

**`subject_GLM_instruction_epochs.sh` no longer loops over the full grid.** It
runs the audit, submits only runs with no complete copy anywhere, and refuses
to start while cleanup is outstanding (a leftover base directory would send the
new run into yet another `+` twin). `ALL=1` restores the old full-grid
behaviour, `DRYRUN=1` prints without submitting, and an explicit
`todo_submit.txt` can be passed instead. A completed GLM with unchanged
settings is never resubmitted.

Verified end to end on a synthetic derivatives tree covering all cases (clean /
duplicates / promote / incomplete-only / no feat / no EVs / no draft fsf /
base-gone-but-`++`-good): audit → cleanup → re-audit converges, and only the
genuinely missing runs reach the todo list.

## 2026-09-07 — FEAT directories pruned to free scratch space (`prune_feat_dirs.py`)

Scratch ran out of space during the instruction-epoch GLMs. `mc/fmri_analysis/prune_feat_dirs.py`
strips first-level `.feat` directories down to what is actually read.

**What is deleted, and only in old GLMs.** Two levels, as intended:

- *Level 1, never touched:* GLM names matching `01`, `01-TR*`, `instr_*`,
  `all-paths-fixed_stickrews_split-buttons`, plus anything written in the last
  7 days (`--protect-days`). The age rule also covers FEAT jobs still running.
- *Level 2, pruned:* everything else under `sub-*/func/glm_*.feat`. The
  directory stays; `stats/pe*.nii.gz`, `stats/{dof,smoothness}`, the `design.*`
  text files, `absbrainthresh.txt` and `custom_timing_files/` survive.
  Removed: `confoundEV*`/`InputconfoundEV*`, `stats/{cope,varcope,tstat,zstat,
  res4d,sigmasquareds,threshac1}*`, `thresh_zstat*`, `rendered_thresh_zstat*`,
  `cluster_*`, `tsplot/`, `logs/`, `mask/mean_func/example_func`.

**No data is affected.** Every `glm_*.feat` reference in the repo resolves to
`.feat/stats/pe*.nii.gz` (grepped before writing the script); nothing reads the
deleted files. Nothing outside `sub-*/func/glm_*.feat` is considered, so
`preproc_clean_*.feat`, `EVs_*` and `motion/` are out of scope entirely.

**Reproducibility.** Scan writes `settings.json`, `report.txt`, `scan.json` and
`to_delete.txt` to `derivatives/group/feat_cleanup_<stamp>/`; deletion works
only from that manifest, re-validates every path independently, and appends
`deleted.log`. Each pruned directory gets a `PRUNED.json` recording what was
kept and when. `check_GLMs_ran.py` reads that marker so pruned GLMs are not
reported as `FILM_UNFINISHED` (two of its FILM completion markers are among the
deleted files).

## 2026-09-06 — why the HC current-location effect disappeared: `l2_norm`

**Symptom.** In the joint RSA GLM the `location` regressor used to be
significant (uncorrected) in both hippocampal ROIs even with DSR in the model;
in the 2026-08-27 and 2026-08-31 runs it was not.

**Cause: `l2_norm` was added to the shared control stack, not anything about
the data.** Combos across runs (all `split_halves_z`, all 1000 perms,
`shift_and_swap`):

All p below are the one-sided permutation p, UNCORRECTED.

| run | combo | l2_norm in stack | HC ant `location` | HC mid `location` |
|-----|-------|------------------|-------------------|-------------------|
| 2026-07-30 | `ctrl_dsrFULL` | no | b=.0403, p=.009 | b=.0330, p=.025 |
| 2026-08-27/31 | `ctrl_fMRI-state_dsrFULL` | yes | b=.0330, p=.052 | b=.0153, p=.215 |
| 2026-08-31 | `ctrl_fMRI-state_dsrFULL-noL2` | no | b=.0371, p=.012 | b=.0341, p=.021 |

`l2_norm` (graded negative distance from the current location to each of the 9
grid nodes) and the categorical `location` RDM correlate r = 0.588 — two
parameterisations of the same variable. Fitting both splits the spatial
variance: in `ctrl_fMRI-state_dsrFULL` the HC mid spatial signal moves onto
`l2_norm` (b=.0347, p=.057) and off `location` (p=.215).

**Not the cause.** (a) The ROI-table rebuild (`alt_final_roi`, 2026-08-27:
mOFC 85->74, HC ant 162->171, HC mid 143->145) — comparing the two no-L2 combos
across the tables gives HC ant .0403 -> .0371 and HC mid .0330 -> .0341, i.e.
nothing. (b) `bttn_next`, also added in August: it is present in the `-noL2`
combo, which restores the uncorrected effect.

**Consequence — and what may and may not be claimed.** Report
`ctrl_fMRI-state_dsrFULL-noL2` for any location claim. Dropping `l2_norm` does
not touch the DSR conclusions: mPFC dsr_fmri b=.0445 -> .0441 (p=.011 either
way), HC mid b=.0828 -> .0836 (p=.001 either way).

But `-noL2` does NOT make location significant in the joint model. Per-combo
BH-FDR across the 5 ROIs puts both HC effects at **q = .052**, which is why the
`location` column of
`pub_figures_v2/heatmap_roi_x_regressor_FDR_ctrl_fMRI-state_dsrFULL-noL2_split_halves_z.pdf`
carries no stars. The correct wording is therefore: current location is
significant in HC when the model is fitted ON ITS OWN (single-model RSA, see
the next entry: q_FDR = .0025 for both HC ROIs), and a TREND when it has to
compete with DSR and the other controls (HC ant b=.037, t=2.26, p=.012; HC mid
b=.034, t=2.09, p=.021; q=.052 for both). Do not call the joint-model effect
significant.

## 2026-09-06 — Fig 2g: single-model current-location RSA (no DSR control)

New `scripts/RSA_DSR_location_only_figure.py`. Pure plotting of the
`location`-alone rows of `results_summary.csv` (run 2026-08-31_17-57-30,
`split_halves_z`); nothing refitted. Only BH-FDR across the 5 ROIs shown is
computed here.

| ROI | n cells | t | beta | p_perm | q_FDR |
|-----|---------|---|------|--------|-------|
| mPFC | 65 | 0.25 | .0037 | .410 | .410 |
| mOFC | 74 | 0.61 | .0091 | .293 | .366 |
| PCC | 51 | 1.28 | .0190 | .127 | .211 |
| HC ant | 171 | 3.19 | .0472 | .001 | .0025 |
| HC mid | 145 | 4.35 | .0643 | .001 | .0025 |

Middle and anterior hippocampus are the only ROIs that fit the current-location
population geometry. Outputs (heatmap, sagittal + lyrz glass brains, combined
panel, values CSV, config) in the run's `location_only_figure_2026-09-06/`.
`mc.plotting.cell_results.plot_roi_beta_glassbrain` gained optional
`figure`/`axes`/`draw_colorbar`/`draw_footer` args so the brain can be drawn
into a shared panel, and `title=''` now means "no title".

## 2026-09-06 — `CELL_SET` now works on the reload path; non-RSA cohort re-run

**The gap.** `spatial_peaks_simple.CELL_SET` ('rsa' | 'not_in_rsa' |
'all_in_roi_table') was only read on the FULL-COMPUTE path — line 1383's
`cs.load_cells(cell_set=CELL_SET, ...)`. Under `RELOAD_FROM` the flag was
silently ignored, so re-testing a cohort from cached results was impossible
without recomputing 1,000 permutations per cell. `per_lag_encoding.py` had no
cohort flag at all: `_load_cells` hardcoded `cell_set='all_in_roi_table'`.

**The fix.** `CELL_SET` added to `per_lag_encoding.py` with the same three
values, and both scripts now apply it on BOTH paths. On reload each subsets its
cached per-cell table via a new `_subset_cell_set`, which resolves the cohort
from `cs.load_rsa_subjects()` — the same session list `cs.load_cells` uses, so
the compute path and the reload path cannot drift apart. The cohort is recorded
in the run tag and in config/settings JSON. Cached CV r and cached per-cell
permutation p are reused; nothing is recomputed. Verified: the reload run
reproduces a standalone filtered re-run bit-for-bit across
`per_roi_stats.csv`, `per_roi_lagwise_by_unit.csv` and
`per_roi_predicted_window_by_unit.csv`.

**Why it was needed.** The manuscript claims the future-tuning effect replicates
in cells that could not enter the population-RSA pseudopopulation. That claim
carried stale numbers (n = 434, mPFC n = 90) from an older ROI labelling.

**Cohort (current labels, `neurons_with_ROI_labels.csv` / `alt_final_roi`).**
419 of 955 cells, 35 of 63 sessions — the sessions absent from
`all_sessions_dsrRSA_grouping_summary.json`, so overlap with the RSA sample is
exactly zero. Per ROI: mPFC 93 (18 sessions), HC_anterior 124 (26), HC_mid 88
(17), mOFC 68 (17), EC 32 (7), PCC 10 (4). **Supersedes "n = 434, mPFC n = 90".**
Runs:
`per_lag_encoding/2026-09-06_20-56-52_reload_from_2026-08-28_10-18-21_..._not_in_rsa`
and
`spatial_peaks_simple/2026-09-06_20-58-05_reload_from_2026-08-28_10-23-56_..._not_in_rsa`.

**mPFC — replicates under BOTH estimators.** Main (pooled-training) estimator,
a-priori 30+60 deg window: across cells vs zero t(92) = 2.62, p = .0051,
q = .015; vs the same cells' other ten lags t(92) = 2.78, p = .0033, q = .0099.
Across the 18 sessions the specificity contrast survives (t(17) = 2.17,
p = .022, q = .034) but the vs-zero test does not (t(17) = 1.42, p = .087,
q = .131) — expected with 18 rather than 33 sessions. Peak: 30 deg across cells
(r = .072, t(92) = 2.77, q over 12 lags = .041; sign-flip max-t FWE within ROI
p = .038), 60 deg across sessions (r = .070, t(17) = 1.54, p = .071).
Permutation null: 25/93 cells (26.9%) at 30 deg (binomial q = 1.9e-11), 17/93
(18.3%) at 60 deg (q = 2.0e-05).
Paired-grid-group estimator, same cohort: window vs zero t(92) = 3.03,
q = .0048; vs other lags t(92) = 2.81, q = .0090; subject level t(17) = 1.85,
p = .041 and t(17) = 1.90, p = .037 (q = .12 after FDR over 3 ROIs). Peak
30 deg across cells (r = .054, q over 12 lags = .030), 60 deg across sessions
(r = .085, t(17) = 2.24, p = .019). 13/93 cells beat their permutation null
(14.0%, binomial q = .0044).

**HC_mid — partly replicates.** Main estimator, subject level: window vs other
lags t(16) = 2.32, p = .017, q = .034; vs zero t(16) = 2.02, p = .030, q = .091.
0 deg holds (subject r = .093, t(16) = 1.97, p = .033; cell r = .067,
t(87) = 2.26, p = .013), but 330 deg does NOT carry over (subject r = -.003).
18/88 cells (20.5%) beat the null at 0 deg. Under the paired-grid estimator the
window is only a trend (cell vs zero t(87) = 1.77, q = .060).

**HC_anterior — does NOT replicate.** Window null at both units and under both
estimators (main, subject: vs zero t(25) = -0.31, vs other lags t(25) = -0.12);
the cohort's peak drifts to 90 deg (cell r = .036, t(123) = 1.65, FWE n.s.).
Per-cell permutation fractions stay above chance (16/124 at 0 deg, 21/124 at
330 deg) — a statement about spatial structure, not about the predicted window.

**Consequence for the manuscript.** "All ROI-level effects were further
reproduced" is too strong and must be narrowed: mPFC reproduces under both
estimators, HC_mid only its present-lag half and mainly on the specificity
contrast, HC_anterior not at all. The mPFC subject-level vs-zero test being at
trend with 18 rather than 33 sessions should be stated, not omitted.

## 2026-09-06 — Cluster-forming threshold in `get_subj_gradients.py` was inactive

**The bug.** `extract_clusters` took `CLUSTER_THRESHOLD = 90` and applied
`np.percentile(subj_data, 90)` over the WHOLE volume. The input maps are
masked: 4,179 non-zero voxels out of 902,629, i.e. 99.5% exact zeros. The
90th percentile of that volume is therefore exactly 0.0 — verified in 33/33
subjects and in every 4-condition dataset — so the branch reduced to
`subj_data > 0` and the nominal "top 10%" threshold never engaged. Anything
describing this analysis as a 90th-percentile threshold is wrong; it is a
threshold at zero.

Two smaller defects in the same function: threshold values in (0, 20] left
`binary` undefined (NameError), and the `"z"` branch z-scored across the full
volume including the 99.5% zeros rather than within the mask.

**What the analysis actually computes.** Per subject and condition: threshold
the beta map at zero within the ROI mask, label 6-connected components
(scipy default connectivity), keep the component with the largest summed beta,
take its beta-weighted centroid, convert to MNI, read the z-coordinate. All
weights are positive because the cluster is defined as > 0.

The connectivity step is not cosmetic. Positive voxels are 54.6% of the mask
on average (range 2-98%); the retained component holds a mean 93% of them but
as little as 16% (20/132 subject x condition cells below 90%, 5/132 below 50%),
and the resulting z can differ from an all-positive-voxel COM by up to 24 mm.
Dropping connectivity gives t(32) = 2.05, p = 0.048 instead of
t(32) = 2.75, p = 0.0098 — so "centre of mass of the positive effects" is NOT
an accurate description of this analysis.

The ROI mask support is byte-identical (4,179 voxels) across all 33 subjects
and all four quarters, so the Q0->Q3 shift cannot be a mask-geometry artifact.

**The fix.** `CLUSTER_THRESHOLD` replaced by `THRESHOLD_MODE`
("zero" | "percentile" | "z"), `THRESHOLD_VALUE`, and a new
`MIN_CLUSTER_VOXELS` extent filter; all thresholds now computed within the
mask via `cluster_forming_threshold()`. Exposed as CLI flags
(`--threshold-mode`, `--threshold-value`, `--min-cluster-voxels`, `--axis`,
`--n-clusters`, `--peak-modes`, `--out-dir`) and recorded in the run JSON/MD.
Defaults are mode="zero", extent=1, which reproduce the published numbers
exactly: mean z = 17.95, 19.50, 21.89, 22.67; t(32) = 2.748, p = 0.0098.
Verified identical to the stored 2026-08-03 run (t = 2.7484 / 2.7371).

**Threshold sweep (6 thresholds x 3 extents x 3 datasets), reported in full,
no cell selected.** Engaging a real threshold WEAKENS the test at every
setting: in-mask p50 t=2.27, p75 t=1.25, p90 t=1.42, p95 t=1.79, z>1 t=1.64,
against t=2.75 at zero. The slopes barely move (1.0-2.2 mm/step across the
whole grid vs 1.65 at zero) — it is the between-subject SD of the slopes that
inflates, from ~3.45 mm at zero to ~6.75 at p90. Thresholding shrinks the
surviving cluster to a handful of voxels and each subject centroid becomes a
noisier estimate; the effect is not disappearing, the precision is.

FAILED DIRECTION — do not repeat: choosing a threshold because it maximises t
is threshold-shopping, and the currently used setting happens to be the best
cell in the grid. The sweep is a robustness check, not a menu. On that reading
it is reassuring: the slope is positive in 18/18 cells for both real datasets,
while the rotated control flips sign across cells and never approaches
significance (18/18 cells p > 0.25, |t| <= 1.16). Direction is
threshold-independent and specific to the true quarter ordering; only power
depends on the threshold. Sweep saved to
`data/derivatives/group/..._cropped_masked/gradient_threshold_sweep_2026-09-06_13-46-28/`.

**Follow-up: does the gradient-mask PC1 axis fit better than MNI z?** No — it
fits worse. Reproduced the axis exactly as `cell_gradient_master_table.gradient_axis`
defines it (PC1 of the 2,765 `gradient_thr_1.5.nii.gz` voxels, x folded to |x|,
oriented +z): PC1 = [-0.038, -0.409, 0.912], matching the [-0.04, -0.41, 0.91]
in the methods text. Projecting the same subject-wise cluster COMs onto it
(quarters_button_state, cluster_com, defaults):

  MNI z   slope=1.652 mm/step  SD=3.45  t(32)=2.748  p=0.0098
  PC1     slope=1.535 mm/step  SD=3.65  t(32)=2.417  p=0.0215
  MNI y   slope=-0.089         SD=2.71  t(32)=-0.188 p=0.852
  MNI |x| slope=0.200          SD=0.74  t(32)=1.558  p=0.129

The loss decomposes exactly: 0.912*1.652 + (-0.409)*(-0.089) + (-0.038)*0.200
= 1.535. PC1's posterior tilt mixes in the y-axis, which carries no trend at
all, so it dilutes the z signal (slope down 7%) while adding between-subject
variance (SD up 6%). Rotated control stays null on both axes (z p=0.431,
PC1 p=0.316).

Also: PC1 is derived from the group t-maps of these same four quarters in these
same 33 subjects, so projecting their COMs onto it re-uses the data the trend
is being tested on. MNI z is data-independent. z is therefore both the better-
performing and the cleaner choice for the fMRI trend. PC1 remains the right
axis for the CELL projection, where the electrophysiology is an independent
dataset.

Number to correct in the methods text: "r = 0.98 with MNI z" is the correlation
across all 158 mPFC cells, most of which lie OUTSIDE the gradient mask. Among
the 87 cells that actually enter the analysis it is r = 0.789; across the mask
voxels themselves it is r = 0.967. Quoting 0.98 overstates how interchangeable
the two axes are for the cells the analysis uses.

**Follow-up 2: the two axes are NOT interchangeable for the cell median split.**
r = 0.967 is the PC1-vs-z correlation across the 2,765 mask VOXELS (mask
geometry, cell-independent). r = 0.789 is across the 87 in-mask CELLS; Spearman
is 0.675. The drop is range restriction plus clumping: the 87 cells occupy only
19 distinct coordinates (cells on one microwire bundle share a location) and
span just SD = 2.19 mm along either axis, against a mask that is tens of mm long.

Consequence for the median split: 5 of the 19 sites change side when the axis is
swapped, carrying 33 of 87 cells (38%). Group sizes barely move (48/39 on PC1 vs
49/38 on z) but the membership does.

  median fMRI preferred angle:  PC1  ventral 36.3 deg | dorsal 83.5 deg (diff 47.2)
                                z    ventral 27.9 deg | dorsal 93.5 deg (diff 65.6)
  pooled cell profile argmax:   PC1  ventral 30 deg   | dorsal 60 deg
                                z    ventral 30 deg   | dorsal 30 deg

The fMRI-angle contrast survives and is LARGER on z. The pooled cell tuning
result does not: the ventral/dorsal peak difference exists on PC1 and vanishes
on z. It is a one-bin shift on a 30-deg grid either way -- the PC1 dorsal peak
(60 deg) leads its runner-up (30 deg) by only 0.032 in mean r, and under z the
two are 0.005 apart. The dorsal peak is a near-tie between adjacent bins and the
axis choice tips it. PC1 was chosen on stated a priori grounds ("more precise"),
independently of this outcome, but the cell-side result is load-bearing on that
choice and should be reported as such.

**Unrelated issue spotted in the 8-condition set.** The "now" condition
(`LOCATION-...-mask_reward-path_beta_std.nii.gz`) has 146,417 non-zero voxels
against 4,179 for the other seven, i.e. a different and much larger support.
Its COM is therefore not comparable to the mPFC-ROI COMs it is plotted beside,
which affects the eighths/circular analysis. The stored 2026-08-03 run had
pruned this file as missing and ran 7 conditions; it is present now, so the
eighths result changed (t=0.99 -> t=1.67). Not yet addressed.

## 2026-09-06 — State on a 0-1 scale, and panel g2/g3 show only the regressed cells

**State scale.** "Different state" rendered lighter than "different location"
because the two models were on different scales. ``my_RSA.compute_crosscorr``
demeans each row before the cosine, so on a 4-element one-hot two different
states come out at 1 - (-1/3) = 1.333 while the Hamming models top out at 1.0;
on a shared colour scale 1.0 then sits at ~75% of the range. The figure now
scores the state regressor by Hamming on its A-D LABEL (`_state_label_EV`,
`STATE_AS_HAMMING`), putting all three models on 0..1. This is the same binary
same/different matrix, only rescaled, and the RSA GLM z-scores every regressor,
so it cannot change a result — but it is a rescaling, not the pipeline's own
numbers, and the flag documents that. All three models now reproduce their
all-tasks block exactly (previously position-in-sequence did not).

**Regressed cells.** ``used_cell_mask`` reproduces the two restrictions the RSA
config applies (``diagonal_included: false``, ``masked_conds: true``): only the
strict upper triangle is regressed, and only path-path / reward-reward cells
(``make_category_masks(..., mask_only_path_rew_combos=True)`` keeps the
``same``-type cells). Panels g2 and g3 now blank everything else, and g2 gains
a leading "cells regressed" panel showing the mask itself — 12 of 28
upper-triangle cells per 8-bin task block.

Consequence worth knowing: within ONE task block every surviving cell is a
different-state comparison, because the four path bins are states A-D and so
are the four reward bins. The position-in-sequence regressor is therefore
CONSTANT within a task block under the mask and only varies across
configurations (the blue diagonal stripes in g3). That is the same reason
`fMRI_run_RSA_without_rsatoolbox_clean.py` drops `path_rew` and `A-state` from
the path-path and reward-reward subsets.

Output folder rolled over with the date: ``method_schematic_06-09-2026/``.
``method_schematic_05-09-2026/`` holds the previous, unmasked run.

## 2026-09-05 (later) — Panel g split into within-half / across-half / all-tasks

Reverted the example subject to **sub-02** (the previous route: 6-5-8-9-8-7-4-1-2-3),
whose model RDMs carry a range of values rather than only 0 and 1. Its two task
halves solved 5-9-4-3 by different routes at B_path — instead of avoiding that,
panel g now uses it as the teaching point:

- **g1  within one run** (`within_half_rdms`) — bin i vs bin j of task half 1.
  This is the arithmetic panel f counts out, and it is only didactic: the RSA
  never compares a run with itself.
- **g2  across the two runs** (sliced from the all-tasks RDMs) — rows = task
  half 1, columns = task half 2, so no cell can be inflated by shared noise.
  Above it, `draw_route_strips` shows both halves' routes with the differing
  bin framed in red, making visible why the B_path diagonal is 1.00 here while
  it is 0.00 in g1, and why new off-diagonals appear (A_path/B_path 0.50,
  B_path/C_path 0.75).
- **g3  all task configurations** — unchanged, and it now fixes the colour
  limits (`_rdm_limits`, 2nd–98th percentile of the all-tasks matrix) that
  every other RDM panel and panel f reuse, so one dissimilarity has one colour
  throughout the figure.

g2 is sliced from the all-tasks matrices rather than recomputed, so it is that
block by construction whatever metric the pipeline scored a model with.
`across_half_rdms` reproduces it for the two Hamming models (verified at build
time) and symmetrises as `(M + M.T)/2` exactly as `compute_hamming_distance`
does; it differs for position-in-sequence only because the RSA scores that one
with `compute_crosscorr`.

**Panel f overlap bars now use the RDM colormap** (`rdm_colour`): an element
that matches contributes 0 to the Hamming distance and gets the RDM's
"similar" colour, one that differs contributes 1 and gets its "dissimilar"
colour, both under the g3 limits. Each bar now ends in a single square — the
one RDM cell that comparison produces — and the numbers read
"6/96 match = 6.2% → d 0.94", tying panel f directly to panel g.

Numbers are back to the sub-02 values: comparison 1 (A_reward vs B_reward)
concurrent 6/96 = 6.2%, location d 1.00; comparison 2 (B_path vs C_path)
concurrent 6/96 = 6.2%, location d 0.50.

## 2026-09-05 — Panel g single-task RDM now IS the block of the all-tasks RDM

The within-task RDM in panel g did not match the corresponding diagonal block
of the across-tasks RDM. Cause: the two matrices were computed differently.
``single_task_rdms`` compared task half 1 against itself, whereas the RSA (and
therefore the all-tasks matrix) compares task half 1 against task half 2. For
sub-02 the subject walked a different route through B_path in the second half,
so the block had a non-zero diagonal at B_path (1.00) and off-diagonals of
0.50/0.75 where the within-task version had 0 and 0.50.

Two changes, both needed:

1. ``single_task_rdms_from_across`` slices the example task's diagonal block
   straight out of the across-task RDMs, and panel g1 now renders that block
   with the same colour limits as g2 (``_rdm_row_figure`` returns the limits it
   used; g2 is drawn first). The two rows of panel g can no longer drift apart
   whatever the data do. ``across_task_rdms_*`` now also carry ``task_keys``
   and ``block_labels`` so the block can be located.
2. ``_pick_fmri_subject`` chooses the example subject rather than hard-coding
   sub-02: it requires the two task halves to have walked the SAME modal route
   through the example configuration (17 of 33 subjects qualify for 5-9-4-3),
   then prefers the route visiting the most distinct grid locations, no
   non-adjacent steps, the most commonly walked route, lowest subject number.
   This lands on **sub-01** (9 distinct locations, 0 non-adjacent steps, route
   shared by 9 subjects). ``FMRI_SUB`` overrides it.

Verified at build time: the sliced block reproduces the within-task matrix
exactly for the concurrent and current-location models. It differs for
position-in-sequence only because the RSA scores that model with
``compute_crosscorr`` while the within-task helper used a 0/1 categorical —
the figure shows the RSA's own values.

Numbers changed with the subject. Route 3→6→5(A)→6→9(B)→8→7→4(C)→1→2→3.
Panel f now reads 12/96 = 12.5% concurrent for both comparisons, with current
location 0% in both — sub-02's 50% at B_path/C_path was specific to its route.
The informative location cell for sub-01 sits at A_path/B_path instead (both
pass through location 6, dissimilarity 0), visible as the off-diagonal blue in
the current-location RDM. Swap ``FMRI_PAIRS`` to ``((1, 3), (0, 2))`` to put
that pair in panel f.

Output folder: ``data/derivatives/group/method_schematic_05-09-2026/``.

## 2026-09-04 — Methods schematic panel f reworked

Panel f of the RSA methods figure now shows TWO within-task comparisons
instead of three mixed ones — one per comparison type the RSA actually uses,
since across-phase (path–reward) cells are excluded from the regression:

- 1  reward–reward: 5-9-4-3 A_reward vs B_reward — concurrent 6/96 = 6.2%,
     current location 0%, position in sequence 0%
- 2  path–path: 5-9-4-3 B_path vs C_path — concurrent 6/96 = 6.2%,
     current location 50% (both bins pass through location 8),
     position in sequence 0%

Both are the same plan rolled by two bins, so the concurrent overlap is
identical; the two rows differ in what the unfolding code says. The two
across-task comparisons (same place now / same future lags) are dropped from
the default but still reachable via ``build_pairs(..., cross_task=True)`` and
``pick_pair``.

The panel now makes the two time axes explicit, which was the point of the
rework: the rainbow future-lag bar with its degree labels sits on top of EVERY
comparison (across a row = how far into the future), while each row is one
encoded time bin. Every strip carries its own A–D annotation underneath —
"at A" / "→B" per block, with the reward blocks framed in that state's colour —
so it is visible that the same lag position holds a different part of the task
depending on which bin the code is read out from. The four cells of the
position-in-sequence block are labelled A–D inside the boxes.

Two helpers added: ``_is_reward_bin`` (a bin where the subject is AT the
reward: ``*_reward`` for fMRI, ``*_early`` for sEEG) and ``_ink``, which
darkens the pale state colours (C = #C7C6E2) until they are legible as text.
Block labels fall back to one letter per state RUN when there is less than
0.70 cm per block (the 12-bin sEEG case); the runs are found from the labels
rather than assumed, so they stay correct whatever bin the strip starts at.

## 2026-09-04 — Methods schematic for the gradient (harmonic angle) analysis

New `mc/plotting/gradient_schematic.py` + `scripts/build_gradient_schematic_figure.py`
explain `scripts/harmonic_angle_maps.py` in seven panels: the concurrent code
cut into four quarters that enter one searchlight GLM (a); the four β's every
voxel therefore has (b); the angle assigned to each quarter — the bin CENTRE,
45/135/225/315° — and the cos/sin projection (c); those two components as one
vector whose angle is the preferred future step and whose length is the effect
size (d); the per-subject vectors, their group mean and the Hotelling T² test
(e); the same subjects on the unit circle with R̄ and Rayleigh (f, the
`USE_UNIT_VECTOR_MAPS` branch); and the resulting preferred-angle map (g).
Same cm-based layout, 9/11 pt fonts and cyclic rainbow as
`method_schematic.py`. Output: `data/derivatives/group/gradient_schematic_<date>/`.

Everything plotted is real. β profiles, per-subject (cos, sin), Hotelling and
Rayleigh statistics and the angle map are read from the analysis outputs; the
sagittal slice is the one carrying the most suprathreshold mPFC voxels
(x = +10 mm, 39 voxels). The two example voxels are picked by a fixed rule —
among the 480 mPFC voxels with Hotelling p < 0.05, the ventral and dorsal
quintiles along MNI z, and within each the largest amplitude:

- ventral MNI [2, 36, −8]: β = [0.0132, 0.0074, −0.0063, −0.0060],
  angle 79.4°, amplitude 0.0236, Hotelling F = 6.90 p = 0.0033,
  Rayleigh R̄ = 0.31 p = 0.75
- dorsal  MNI [12, 46, +14]: β = [0.0080, 0.0124, −0.0020, −0.0054],
  angle 105.7°, amplitude 0.0204, Hotelling F = 4.30 p = 0.0225,
  Rayleigh R̄ = 0.32 p = 0.71

Both voxels are Hotelling-significant but Rayleigh-null, i.e. a reliable
group-mean vector without per-subject angle agreement. Panel f states this
outcome rather than implying the voxel passed both tests.

**Colour-lookup bug caught while building panel g.** Rendering the angle map
with `vmin=-180, vmax=180` on the cyclic colormap puts 0° in the MIDDLE of the
map (blue) instead of at yellow; the mPFC cluster came out green/teal rather
than red/purple. The angle is now wrapped into [0, 360) before lookup. Only
affects this figure — `harmonic_angle_maps.py` writes signed degrees and the
fsleyes recipes in its README use `hsv`, which has the same wrap issue if the
display range is set to −180..180; worth checking there too.

## 2026-09-04 — Methods schematic: fixed physical sizes, rainbow future scale

`mc/plotting/method_schematic.py` now lays every panel out in absolute cm
(`_add_ax_cm`, `_cm_canvas`, and cm-valued data coordinates for the code
strips) and saves WITHOUT `bbox_inches='tight'`, so the plotted boxes come out
at exactly the declared size on an A4 page and the type stays at its nominal
point size. Sizes: RDM 4x4 cm each, panel c block 4x4 cm, and the code row
7 cm (concurrent) + 2 cm (current location) + 3 cm (position in sequence).
Panels d and e are now one row sharing those three columns, matching panel f.
Fonts: 11 pt headings, 9 pt everywhere else — `FONT_TICK` was 8 pt and is now
9 pt, the A4 floor. `_thin` blanks tick labels that would sit closer than
0.32 cm, and block labels collapse to one letter per state when a loop has
more than one bin per state (the 12-bin sEEG case).

Future lag / task angle now uses the cyclic rainbow of the mPFC gradient
figures (`FUTURE_CMAP`, `lag_colors`): 0 deg yellow, 90 red, 180 blue,
270 green, wrapping back to yellow — replacing the yellow-to-brown ramp.

Panel a places each state letter on the wedge where the subject is AT that
reward (derived from the bin labels) instead of at a fixed 90 deg spacing.

Numbers, comparisons and the data pipeline are unchanged.

## 2026-09-04 — Methods schematic: unfolding vs concurrent code

New `mc/plotting/method_schematic.py` + `scripts/build_method_schematic_figures.py`
build the figure that explains the two competing model geometries (panels a-g:
bins as future angles; example configuration and executed trajectory; the
per-bin encodings; the concurrent code read out from two bins; the unfolding
code; similarity by counting overlapping elements; the resulting RDMs for the
single task and across all tasks). Output goes to
`data/derivatives/group/method_schematic_<date>/` — one assembled overview plus
every panel as its own PDF/PNG, with a settings JSON per figure.

Two example tasks, each in the framework it belongs to:
`5-9-4-3` (fMRI, sub-02 E1_forw, 8 bins x 12 resampled steps -> 96-element DSR,
mirroring `create_fMRI_model_RDMs_on_clean_beh.py`) and `3-7-9-5` (sEEG, modal
path over 870 correct trials, 12 bins x 1 location, mirroring
`RSA_DSR_ROIs_simple.py`). Across-task RDMs come from the real pipeline:
`my_RSA.build_across_halves_model_RDM` on the saved EV pickle for fMRI,
`compute_hamming_distance` over the 8 configurations for sEEG.

**Alignment bug found in the sEEG schematics.** The raw 360-bin sEEG loops for
`3-7-9-5` are anchored on reward D, not reward A: the modal 12-bin trajectory
reads 5-6-3-3-1-7-7-8-9-9-6-5, so labelling bin 0 as 'A' put the wrong
locations under A-D — the mismatch visible in the earlier Fig-1e simulation
panel. `align_to_state_A` now picks the rotation that lines the four
state-onset bins up with the four rewarded locations (here: shift 2, 4/4 onsets
matched) and the shift is logged. The fMRI side needs no rotation: its bins are
labelled from behaviour.

Also logged (`non_adjacent_steps`): downsampling the sEEG loop to 12 bins drops
intermediate locations, so `3-7-9-5` shows two non-adjacent grid steps
(3->1, 1->7). Reported rather than smoothed. The fMRI trajectory
6-5-8-9-8-7-4-1-2-3 is fully adjacent.

Comparisons selected automatically for panel f (fMRI example, Hamming
similarity of the 96-element codes):
- same task, A_reward vs B_reward: concurrent 6.2%, location 0%, position 0%
- 5-9-4-3 A_path vs 1-7-5-3 D_path — same place now: concurrent 12.5%,
  location 100%, position 0%
- 5-9-4-3 B_reward vs 3-5-7-1 C_reward — different place now, same locations at
  the same future lags: concurrent 56.2%, location 0%, position 0%
The last two are the dissociation the RSA rests on and are picked by
`pick_pair`, not hand-chosen.

## 2026-09-03 — One consolidated lag-wise results table

`scripts/overlay_double_dissociation.py` now writes ONE lag-wise table,
`overlay_lagwise_tests.csv` (144 rows), replacing `overlay_per_lag_table.csv`
and `overlay_subject_clustered_lagwise_ttests.csv` (both deleted on re-run,
along with the older `WEIGHTING_RESULTS.md`). `_subject_clustered_lagwise_tests`
and the inline per-lag block in `make_overlay` are gone, replaced by
`_lagwise_tests`.

Grid: unit_of_analysis (cell, subject) x metric (raw_r, fisher_z) x 3 stats
ROIs x 12 lags. Columns: `p_one_sided`, `q_one_sided_across_12_lags`,
`q_one_sided_across_rois`, `q_one_sided_across_rois_and_lags`, plus
`mean_tested_value` (what the t ran on), `mean_raw_r`, `mean_fisher_z`,
`r_from_fisher_z`, `t`, `df`, n's. One-sided only, as requested. Every BH
correction runs INSIDE one unit x metric block -- the four blocks are versions
of the same test, not additional tests. Fisher-z numbers reproduce the two old
tables exactly. The peak-lag permutation now also covers the 2 x 2 grid.

**Raw r is uniformly slightly stronger than Fisher z** (cell mPFC 30 deg
t = 3.09 vs 3.02; subject mPFC 60 deg t = 2.22 vs 2.16), as expected since the
z transform inflates the variance of extreme cell r's. The difference never
changes a conclusion except at one knife-edge, below.

**What survives, by correction scope (one-sided):**
- across 12 lags: ONLY cell mPFC 30 deg (q = .014 raw_r / .018 fisher_z).
  Nothing at subject level, nothing in HC.
- across 3 ROIs at one lag: cell mPFC 30 deg (.0035/.0044), cell HC_mid 0 deg
  (.016/.014), cell HC_anterior 330 deg (.034/.037), cell HC_mid 330 deg
  (.039/.049), subject HC_mid 0 deg (.038/.039).
- across ROIs AND lags (36 tests): cell mPFC 30 deg is q = .042 with raw r but
  q = .053 with Fisher z -- the single place the metric flips a threshold.
  Nothing else survives. Flagged so the choice is never made by outcome; the
  pre-registered metric (Fisher z) stands.


## 2026-09-03 (instruction-phase RSA) — cumulative reward-prefix models + per-combo RDM scope

`scripts/fMRI_run_RSA_instruction.py` gained a second family of models sliced
out of `rewDSR` at the `A_reward` anchor, and combo models can now declare the
task-half scope they are fitted in.

**Cumulative "first k rewards" models.** The `rewDSR` A_reward vector is four
equal 12-element chunks, one per reward step (A, B, C, D), each holding the raw
location value repeated. Alongside the existing per-step SPLIT channels
(`curr_rew` = A, `next_rew` = B, `two_next_rew` = C, `three_next_rew` = D) the
script now also builds the CUMULATIVE prefixes:

| model | chunks | rewards | width |
|---|---|---|---|
| `A_rew` | 0 | A | 12 |
| `AB_rew` | 0–1 | A, B | 24 |
| `ABC_rew` | 0–2 | A, B, C | 36 |
| `ABCD_rew` | 0–3 | A, B, C, D | 48 |

**Models renamed** so the name states which rewards the model knows about. The
per-step SPLIT channels became `A_rew` / `B_rew` / `C_rew` / `D_rew` and the
cumulative ones `A_rew` / `AB_rew` / `ABC_rew` / `ABCD_rew`; `ABCD_rew` is the
whole rewDSR vector, so `rewDSR` and `ABCD_rew` are numerically identical.
The pre-rename names are **rejected with an error**, not silently accepted:
`curr_rew`, `next_rew`, `two_next_rew` and `three_next_rew` are *also* real keys
in `model_EVs` holding the 9-dim ONE-HOT version of the same idea, so an
un-updated config would have fallen through to the standard path and quietly
built a different model (mismatch 2/9 instead of 1) under an unchanged name.
See `LEGACY_MODEL_NAMES` / `check_no_legacy_names`.
`condition_files/rsa_instruction_full.json` was updated to the new names so it
keeps building exactly the models it built before.

Because the chunks are equally long, Hamming on a cumulative model is *exactly*
the mean of the per-step Hammings it contains — verified numerically on sub-02
(max abs difference 1.1e-16). So the family is strictly nested and `rewDSR` **is**
the k = 4 member; no separate model had to be defined for it. Each has an
`_instr` counterpart built through `instruction_relabel_dict` as before, and the
2×2 sub-block uniformity check passes for `two_rew_instr` / `three_rew_instr`.

**Per-model scope.** A combo entry may carry `"scope"` (a string or a list),
and single models are governed by a config-level `single_model_scopes`,
e.g. `{"execution": ["within_only", "across_only"], "instruction":
["within_only"]}` — applied by name, anything ending in `_instr` counts as
instruction. Both override `data_rdm_scope`; the resulting output maps are
suffixed `_within` / `_across` / `_full`. The asymmetry is forced by the design:
an execution model has variance in both blocks, an instruction model is constant
across halves. All scopes are now column masks over the strict
lower triangle of the *same* cached (2n × 2n) data RDM — `within_only` keeps the
two within-half blocks (90 cells for n = 10), `across_only` their complement
(100 cells, exactly the across block), `full_no_diag` all 190 — so fitting one
combo in several scopes costs no extra searchlight computation. The legacy
top-level `'across_only'` mode still uses its own (n × n) cache and refuses to be
mixed with per-combo scopes.

**New config** `condition_files/rsa_instruction_cumulative_rew.json`
(`data_rdm_scope: "within_only"`, since every instruction model is degenerate
across halves). Single models: all 14, each execution one fitted within AND
across halves, each `_instr` one within only. Combos: `first_exe_vs_instr`, `two_exe_vs_instr`,
`three_exe_vs_instr`, `four_exe_vs_instr` (each = execution vs instruction
variant of the same k), `instr_split` (the four instruction step channels), and
`exe_split` (the four execution step channels) fitted **both** within and across
task halves.

All designs are full rank (sub-02, TR4). Execution-vs-instruction shared
variance within half: `A_rew` r = +0.15, `AB_rew` r = +0.16,
`ABC_rew` r = +0.26, `ABCD_rew` r = +0.23. Regressors of `exe_split` are far
more collinear across halves (r = +0.46 to +0.73) than within (r = −0.10 to
+0.45), which is worth keeping in mind when comparing the two `exe_split` fits.

**Reporting.** Model RDMs are now plotted once per scope each model is actually
fitted in (cells outside the scope NaN/white), instead of always showing the
across block; the same holds for the example data-RDM panel. Each figure is
accompanied by a printed line giving the number of fitted cells, distinct
values, range and sd, so a degenerate regressor is visible without opening the
png. Regressor correlations — per combo, per scope — and the
execution-vs-instruction correlations are now written into
`{sub}_settings_summary.json` (`combo_regressor_correlations`,
`exec_vs_instr_correlations`) rather than only printed to stdout.

Also removed the dead `_needs_split_exec` / `_needs_split_instr` expressions that
had already been superseded by the explicit `_base_of` check.

## 2026-09-03 (new instruction-period first-level GLMs) — 11 epoch-wise instruction GLMs

`scripts/create_EVs_instruction_period.py` + `condition_files/EV_config_instruction.json`.
Same shape as the existing `01-TR{k}` per-TR instruction GLMs, but the epochs
are defined by what is on screen rather than by TR. 11 GLMs per task half, each
with 10 EVs (one per task) + the button nuisance:

| GLM | window | dur |
|---|---|---|
| `instr_see-{A,B,C,D}-first` | 0–1.5, 1.5–3, 3–4.5, 4.5–6 | 1.5 s |
| `instr_see-{A,B,C,D}-second` | 6–7, 7–8, 8–9, 9–10 | 1.0 s |
| `instr_collapsed-first-instruction` | 0–6 | 6 s |
| `instr_collapsed-second-instruction` | 6–10 | 4 s |
| `instr_empty-screen` | 10–12 | 2 s |

**Timings read off the experiment code**, not assumed —
`mc/latest_experiment/3x3_fMRI_part1.py` lines 697–712 (opacity schedule) and
744–915 (component stop times). The routine is non-slip timed to 12 s. Note the
second pass is **1.0 s per coin starting at 6 s**, not 1.5 s from 4.5 s, and all
coins plus the "backwards" warning stop being drawn at 10 s — only
`sand_pirate` stays to 12 s, which is what makes 10–12 s an empty screen.

Each GLM is named after what it measures rather than by an index, so the output
is `EVs_instr_see-A-first_pt01/` and `sub-02_draft_GLM_01_instr_see-A-first.fsf`.
EVs are named `{task}_{direction}_instruction_onset`, which is what
`pair_correct_tasks` in `scripts/fMRI_run_RSA_instruction.py` expects.
`load_data_EVs_instr_TRwise` now takes `TR=None` to mean "regression_version is
already the full GLM name" (it still builds `{version}-TR{TR}` otherwise, so the
existing `01-TR{k}` configs are unaffected); the epoch configs set
`"regression_version": "instr_see-A-first"` and no TR.
Submit with `mc/fmri_analysis/subject_GLM_instruction_epochs.sh`, which is
`subject_GLM_RDM_conds.sh` plus a loop over the epoch names.

**Why one GLM per epoch and not one GLM with all epochs.** A joint design was
built first and rejected on the numbers, measured on FEAT's own design matrix
(`feat_model`, sub-02 pt1, gamma HRF δ=6/σ=3, 100 s highpass, TR 1.078):

| design | VIF over the epoch regressors |
|---|---|
| joint, 6 epochs/task (61 EVs) | 4 – 132 |
| joint, 9 epochs/task (91 EVs) | 22 – **2555** |
| **one GLM per epoch (11 EVs)** | **1.01 – 1.05**, max abs r 0.005 |

The epochs are 1–1.5 s, back-to-back, in a fixed order, never jittered, so in a
joint design adjacent second-pass coins correlate at r = 0.95 and each beta is a
mixture of its neighbours. Since the hypothesis under test is a gradient across
A→D, a design that mechanically blends the coin epochs could manufacture that
signature. Per-epoch GLMs cannot: each holds one regressor per task, ~150 s
apart. This does not fix precision — there is still one 1.5 s event per
condition per half — but it makes the noise unstructured rather than organised
by the design.

**Instruction onset.** `instruct_start = (first start_ABCD_screen of that task)
− 12 s`. Validated against the on-flip timestamps (`sand_pirate.started`) in the
`*_all.csv`: those sit on a different clock origin than `globalClock`, but the
interval to the first `start_ABCD_screen` is constant to **SD ≈ 8–12 ms (< 1
frame)** across the 10 tasks of a session. Replaces the legacy `01` approach,
which used `t_reward_afterwait + 3.5` for tasks 2–10.

**Button nuisance regressor: kept, but it does nothing either way.** Measured
variance inflation on the instruction betas 1.0007 (max 1.004), max abs r with any
instruction EV 0.12. It does not fix the baseline — the instruction period is
6.7% of the run and there is no rest anywhere in the design, so the betas are
unavoidably "instruction vs. execution". For RSA that is a common offset across
conditions and cancels in correlation distance. Kept for comparability with the
`01`/`01-TR` GLMs; `"regress_buttons": false` switches it off.

**Runners.** `mc/fmri_analysis/subject_GLM_instruction_epochs.sh` submits the
FEAT jobs (subject x half x epoch) and
`mc/fmri_analysis/submit_RSA_instruction_epochs.sh` the RSA jobs, writing one
config snapshot per epoch with `regression_version` set and `TR` null.
`wrapper_python_fMRI_RSA_clean_config.sh` is unchanged — it only forwards
subject, config and script name.

**`mc/fmri_analysis/check_GLMs_ran.py`** verifies the FEAT runs before they are
trusted. Per (subject x half x GLM) it separates: no EV folder, no .feat, no
stats/, incomplete PEs (killed mid-FILM), FILM unfinished (all PEs but no
dof/smoothness/sigmasquareds/threshac1 — those are written last), task-to-EV.txt
pointing past the design, and a `+.feat` twin, which matters because FEAT never
overwrites an existing output dir: it appends '+' and leaves the stale original
in place, and the stale one is what the RSA reads. `--check-data` additionally
loads every PE the RSA uses and flags all-zero or non-finite maps. Prints a
per-GLM resubmit list and exits 1 when anything failed. Expected GLM names come
from the same EV config the EV script uses, so the two cannot drift apart.

**Caveat — sub-10 pt1.** Its last task's instruction (B1_backw) ends at 1878 s.
If that run is the nominal 1670 vols x 1.078 s = 1800 s, `feat_model` does not
warn — it **aborts** with "No valid [onset duration strength] triplets found"
and builds no design at all. Confirmed locally. The script prints this as an
upfront warning. Check the real dim4 for sub-10 pt1 before running; the old `01`
instruction GLMs have the same exposure.

**Figures.** `plot_instruction_RDM` and `plot_model_correlations` in
`mc/analyse/my_RSA.py` now draw 4 x 4 cm panels in Arial with a 9 pt floor, and
save .pdf next to .png. Specifics: correlation-matrix annotations went from 4 pt
(unreadable at print size) to 9 pt, and that panel grows past 4 cm when a matrix
has too many models for a 9 pt number to fit in a cell — the font floor wins
over the target footprint; `annot=False` keeps 4 cm. RDM tick labels fall back
to per-half block labels when the matrix is an assembled (2n x 2n) one, and
otherwise to every kth real label — labelling both axes TH1/TH2 on the (n x n)
across block, where rows are TH1 and columns are TH2, would have been wrong.
Colorbars carry only their two end ticks, since the default '0.0/0.5/1.0' set is
wider than a 4 cm panel leaves and was being clipped.


## 2026-09-02 (per-TR instruction RSA) — within-half-only run: the offset shrank, flipped sign, and still drove the stats

**Inputs.** `group_RSA_within_th_only_intr-vs-exe_glmbase_01-TR{0..11}_cropped`,
24 maps, 32 subjects, brain mask 144 404 voxels. Analysed with
`scripts/per_TR_loso.py` (statistics in `mc/analyse/loso.py`), 10 000 sign-flip
permutations for the SVC/LOSO, 1 000 for whole brain, masks mPFC + MTL.
Results in `data/derivatives/group/within_th_only_intr-vs-exe_allTRs_2026-09-02`
(raw) and `..._allTRs_DEMEANED_2026-09-02` (demeaned; see below).

**Did dropping the across-half block remove the bias? Partly, and it flipped.**
Median whole-brain t, averaged over maps of each class:

| analysis | execution | instruction | exec - instr |
|---|---|---|---|
| full (across+within), 2026-08-28 | -0.584 | +1.236 | **-1.82** |
| within-half only | +0.171 | -0.450 | **+0.62** |
| within-half + per-subject demeaning | -0.003 | +0.006 | **-0.01** |

The gap shrank ~3x but **reversed sign** — an attenuated version of the same
artefact would have kept its sign.

**Measured cause: restricting to within-half cells RAISES the execution/
instruction regressor correlation.** Pearson r between each execution model and
its own instruction counterpart, sub-02, over the cells the OLS actually fits
(printed by `fMRI_run_RSA_instruction.py`, re-run locally with TR4 — the model
RDMs are built from behaviour, so the TR does not enter this number):

| pair | `full_no_diag` (90+90 cells) | `within_only` (90 cells) |
|---|---|---|
| curr_rew vs curr_rew_instr | +0.02 | **+0.146** |
| next_rew vs next_rew_instr | +0.11 | **+0.318** |
| two_next_rew vs two_next_rew_instr | +0.11 | **+0.318** |
| three_next_rew vs three_next_rew_instr | +0.02 | **+0.146** |
| rewDSR vs rewDSR_instr | (n/a) | **+0.231** |

So in the full scope the two regressor families were near-orthogonal (r = .02-.11)
and the old bias came from the block-structure offset, NOT from exec/instr
collinearity. Dropping the across-half block removed that block offset but
*tripled* the direct correlation between execution and instruction regressors.
In a joint OLS, positively correlated regressors give anti-correlated betas: if
execution carries the signal, the instruction beta is pushed negative to
compensate. That is exactly the observed instruction-negative /
execution-positive offset, and it explains the sign flip. Design-induced
collinearity, not a scanner artefact — but still an additive offset rather than
a regional effect.

(Consistent with the structural argument: of the 90 within-half cells, 10 are
same-task-letter pairs where instruction dissimilarity = 0 while execution
dissimilarity = 1.)

**The residual offset was producing the "significant" results.** Across the 24
maps, a map's global median t predicts its peak statistic almost perfectly:

    corr(global median t, mPFC negative peak t) = +0.918, p = 2.6e-10
    corr(global median t, MTL  negative peak t) = +0.698, p = 1.5e-04
    median t of the 7 maps WITH a significant negative peak: -0.832
    median t of the 17 maps WITHOUT:                         +0.073

The sign-flip null is centred on zero, so a map whose whole brain sits at -0.9
yields a "significant" negative peak somewhere regardless of any regional
effect. **The negative mPFC/HC hits in the raw run are therefore not findings**
(largest: `next_rew_instr` mPFC t = -6.84 p_FWE = .0006; MTL 28/-20/-18, 90 %
right hippocampus, t = -5.43 p = .016). The LOSO timecourses inherit it too
(r = +0.66 mPFC, +0.44 MTL), which invalidates `three_next_rew` MTL
(LOSO p = .017) — it has the largest positive offset of all 24 maps (+0.545).

**Fix: `--demean` (new flag).** Subtracts each subject's whole-brain mean,
separately per TR, from the data array before any statistic, so every sign-flip
permutation inherits it and there is no separate permutation path (rule 4). The
test becomes "is this voxel above this subject's own brain-wide level at this
TR" rather than "above zero". Note this was chosen POST HOC, after seeing the
offset; both runs are kept and reported side by side. Median t ~ 0 afterwards is
guaranteed by construction, not a result — the question is which peaks survive.
After demeaning, corr(offset, mPFC negative peak) collapses +0.918 -> -0.402.

**What changes.** Maps with a positive offset lose, maps with a negative offset
gain, and one result moves the other way — the signature of a real regional
effect (demeaning also removes subject-specific global fluctuation, cutting
between-subject variance):

| map | raw t / p_FWE | demeaned t / p_FWE |
|---|---|---|
| `CURR_REW-splitDSR_vs_instr` (execution, reward A) TR2, 22/-90/32 occipital pole | 5.90 / .110 | **6.86 / .032** |
| `curr_rew` (execution) TR2, 22/-92/34, same site | 5.22 / .300 | 6.32 / .097 |
| `CURR_REW_INSTR-instr_split` (instruction) TR3, 16/-70/26 cuneus/precuneus | 7.18 / **.007** | 6.19 / .103 |

**Surviving results after demeaning** (whole brain corrected over 144 404
voxels x 12 TRs; SVC corrected within each mask over voxels x TRs):

- Whole brain: `CURR_REW-splitDSR_vs_instr`, TR2, MNI 22/-90/32, t = 6.86,
  p_FWE = .032 (44 % occipital pole). Replicated by `curr_rew` at 22/-92/34
  (p = .097). Execution, not instruction.
- MTL SVC, POSITIVE and instruction-driven, at one voxel:
  `REWDSR_INSTR-rewDSR_vs_instr` TR5, MNI -28/-22/-32, t = 5.40, p_FWE = .021;
  `three_next_rew_instr` TR6, same voxel, t = 5.03, p_FWE = .049.
  Harvard-Oxford: 29 % left anterior parahippocampal gyrus, 12 % posterior
  temporal fusiform — parahippocampal/entorhinal, NOT hippocampus proper.
  **These two are NOT independent**: `three_next_rew` is literally one quarter
  of the `rewDSR` A_reward vector (`REWDSR_SPLIT_CHANNELS`), and their t-maps
  correlate r = +0.53. Worse, the result is fragile at threshold —
  `rewDSR_instr` and `REWDSR_INSTR-rewDSR_vs_instr` are near-identical maps
  (r = +0.973) peaking at the SAME voxel, yet t = 4.70 / p = .111 versus
  t = 5.40 / p = .021. A 0.7 change in t moves it across the threshold, so this
  voxel should be treated as a lead to test in an independent contrast, not as
  a result.
- MTL, right posterior parahippocampal 34/-28/-18: `NEXT_REW_INSTR-instr_split`
  TR11 SVC p = .092 with LOSO p = .029; `next_rew_instr` LOSO p = .031 at TR10.

**Multiple comparisons, stated plainly.** 24 maps x 2 masks = 48 SVC tests plus
24 whole-brain tests; each p_FWE corrects only over its own voxels x TRs. None
of the above is corrected across that family. At 48 tests, ~2.4 hits below .05
are expected by chance and we have 2 in MTL, which are not independent of each
other — so the MTL parahippocampal result is NOT established by this run.

**Figure.** `offset_diagnostic.pdf` / `.png` in the DEMEANED folder, produced by
the archived `make_offset_diagnostic.py` beside it.

**Two code fixes made while getting this to run** (both in `mc/analyse/loso.py`
+ `scripts/per_TR_loso.py`):
1. `check_inputs` — a pre-flight that compares each .nii.gz gzip trailer against
   the size its NIfTI header implies, confirming only the suspects by real
   decompression. A truncated download has a readable header and dies inside
   `get_fdata`, which killed a run partway; this fails in ~1 s instead. It found
   exactly the 24 files `gzip -t` found (all of TR0, an interrupted transfer,
   since re-downloaded). Bypass with `--skip-input-check`.
2. `peak_TR` was a position along the TR axis, not a TR number, so any run over
   a non-contiguous `--trs` subset mislabelled its peak (and the plot x-axis,
   labelled "instruction period (s)", assumed index = second). Real TR labels
   are now passed into `run_svc` / `run_loso` / `run_wholebrain`, stored in each
   summary JSON as `trs`, and used for both. Contiguous 0-11 runs are unaffected.

**Not repeated / dead ends.** Testing the negative direction against the same
sign-flip null is only valid when the map has no global offset; with one, it is
guaranteed to "find" something. Do not report negative-direction peaks from a
map whose global median t is far from 0 without demeaning first.

## 2026-09-02 (later still) — Shortest-path metric in `behaviour_summary.py`

Added one behavioural metric to `scripts/behaviour_summary.py`: the percentage
of walks between two consecutive rewards that took the minimum possible number
of steps. The 3x3 grid is 4-connected (no diagonals, no wrap-around), so the
minimum is the Manhattan distance between the two rewards. Four walks per
repeat: D->A, A->B, B->C, C->D. New outputs `fmri_shortest_paths.csv` and
`ephys_shortest_paths.csv` (one row per walk: n_steps, min_steps, is_shortest),
new per-subject/per-session column `shortest_path_percent`, group entries
`shortest_path_percent` / `_pooled` / `_by_repeat` (fMRI) resp.
`_by_rep_correct` (cells), plus one overview histogram per modality.

**Numbers (all data, no exclusions beyond the ones already in the script):**
fMRI 95.8 +- 4.1 % (mean +- s.d. over 33 subjects, 12 514 walks; pooled 95.8 %);
cells 97.4 +- 2.2 % (63 sessions, 56 904 walks; pooled 97.4 %). Mild improvement
over repeats (fMRI 94.5 % -> 96.3 % from repeat 1 to 5; cells 97.0 % -> 97.6 %).
Detours are essentially always +2 steps (fMRI 472 of 524 non-shortest walks).

**Two definitional choices, both to keep the metric about paths BETWEEN
rewards.** (i) The first A of an fMRI configuration is skipped: the location the
subject starts from is not stored in the cleaned table. (ii) In the cells data
the D->A walk is only counted when the preceding attempt was itself a retained,
correct repeat that this attempt continues from, so the exploratory search for A
at the start of a grid never enters the metric. A variant that counts every
D->A / grid-initial walk regardless gives 97.1 % pooled instead of 97.4 %, so
this choice barely moves the number.

**One data glitch, left in.** Exactly 1 of 134 381 cell-data location
transitions is between non-adjacent squares (a dropped sample in the 25-ms
trace), which makes 1 of 56 904 walks come out one step SHORTER than the
Manhattan distance. It is counted as not-shortest; no filter was added.

**Unrelated speed fix in the same file.** The 25-ms location traces are stored
as a single very wide row and `pd.read_csv` parses them column by column
(~0.35 s per file, 1489 files ~ 9 min). `_read_location_trace` reads the line
and parses it with numpy (~10 s total). `ephys_trajectory_candidates` now uses
it too.

## 2026-09-02 (later) — Single-lag reporting for the double dissociation

Added two ways to report ONE lag instead of the 30+60 / 0+330 window in
`scripts/overlay_double_dissociation.py`, because the correction scope has to
match how the lag was chosen. Outputs:
`overlay_prespecified_lag_tests.csv`, `overlay_peak_lag_permutation.csv`.

**(a) a-priori lag, BH across the 3 ROIs** (`PRESPECIFIED_LAG_DEG`, default
mPFC 30 deg, HC 0 deg).

**(b) peak lag chosen by looking, sign-flip permutation, max-t across the 12
lags** (10000 perms, seed 42). Motivated by the lag-lag correlation in
`lag_lag_correlation_noctrl.csv`: neighbouring lags correlate at r = .33
(mPFC), .23 (HC_mid), .22 (HC_anterior), so BH over 12 lags assumes an
independence that does not hold. The permutation flips the sign of whole unit
curves (keeps each unit's lag-lag structure) and shares a subject's sign
across ROIs in the across-ROI null. `_check_vectorised_t` asserts the
vectorised permutation t reproduces `_tstat_gt0` on the observed data, so
empirical and null use one estimator (CLAUDE.md rule 4).

**Result: the max-t correction buys almost nothing** — mPFC 30 deg cell-level
BH q = .0177 vs FWE p = .0159; HC_mid 0 deg cell BH q = .057 vs FWE p = .054.
The lag curves are not correlated enough for the search penalty to shrink much.

**Peak lags under FWE (max-t across 12 lags):** cell mPFC 30 deg t(157) = 3.02,
p_FWE = .016 (*) is the ONLY peak that survives. Cell HC_mid 0 deg p_FWE = .054,
cell HC_anterior 330 deg p_FWE = .13, subject mPFC 60 deg p_FWE = .18, subject
HC_mid 0 deg p_FWE = .14, subject HC_anterior 0 deg p_FWE = .36.

**The a-priori mPFC lag choice is decisive, and that is a trap.** With mPFC
fixed at 30 deg: cell q = .0044 (**) but subject q = .108 (n.s.). With mPFC
fixed at 60 deg: subject q = .029 (*) and all three ROIs significant at subject
level (HC_mid .029, HC_anterior .040), while cell mPFC drops to q = .0496.
This is the cell-vs-subject peak reversal already documented in the supplement.
Picking 30 or 60 after seeing these numbers would be circular; recorded here so
that the choice is made explicitly and on a priori grounds, not by outcome.

**Conclusion: the two-lag window remains the only route where all three ROIs
survive under BOTH weightings**, so it stays the inference for Fig 3c; single
lags are fine to quote descriptively.


## 2026-09-02 — Double-dissociation overlay: readable results table + results.json

`scripts/overlay_double_dissociation.py` now writes `overlay_results.json` and
`RESULTS.md` (replacing `WEIGHTING_RESULTS.md`, which only showed 4 of the 12
lags and one q column). No statistic changed -- this is a reporting refactor.

**Why:** the old CSVs made it impossible to see which t belonged to which r/z,
and three differently-scoped FDR corrections were all called `..._fdr_...`
without saying what family they corrected over.

- Every test record now carries `mean_fisher_z` (what the t was computed on),
  `r_from_fisher_z = tanh(mean_fisher_z)` (the same effect in r units) and
  `mean_raw_r` (what the figure plots), so t <-> r <-> z is unambiguous.
- FDR columns renamed to name their family: `q_*_within_roi_12_lags`,
  `q_*_within_lag_across_overlay_rois`, `q_*_across_rois_and_lags`,
  `q_*_across_rois`. `FDR_FAMILIES` in the script documents each one and is
  copied into the json and the md.
- `overlay_results.json` nests `p` and `fdr_q` inside each test record; also
  holds `settings`, `test_definitions` and `fdr_families`.
- Re-ran on
  `per_lag_encoding/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled/`
  (the run with the corrected cell coordinates).

**Fig 3c star, target-window contrast (predicted lags > other 10 lags,
one-sided, BH across the 3 ROIs), subject-balanced:** mPFC 30+60 deg
dz = 0.057, t(32) = 2.86, p = .0037, q = .0055 (**); HC_mid 0+330 deg
dz = 0.047, t(35) = 2.88, p = .0034, q = .0055 (**); HC_anterior 0+330 deg
dz = 0.032, t(51) = 1.81, p = .038, q = .038 (*). Cell-weighted: mPFC
t(157) = 2.87, q = .0057; HC_anterior t(294) = 2.69, q = .0057; HC_mid
t(232) = 2.27, q = .012. So all three survive FDR under both weightings.


## 2026-09-01 (later still) — Phase-residualisation choice is now self-documenting

`PHASE_RESIDUALISE = 'cosine_2h'` was justified from a run on the old 8-recday
set, and the justification lived only in a comment. It is now RECOMPUTED EVERY
RUN: `PHASE_BASES_TO_COMPARE = ['cosine', 'cosine_2h']` sends each basis through
the same `run_all_recdays` -> `methods_results_stats` path as the primary
analysis, prints a read-out, and writes
`key_analysis_stats.json['phase_residualisation_comparison']` with the
criterion stated in the file. Costs one extra full pass per non-default basis.

Criterion (fixed in advance): use the basis that removes the Subgoal Progress
effect, since subgoal progress dominates this dataset and any residue would
inflate the action-plan fit.

Re-tested on the full 25 recdays / 7 mice — the original conclusion holds:

| basis     | analysis      | Subgoal Progress beta | t(23) | q_FDR    | sig |
|-----------|---------------|-----------------------|-------|----------|-----|
| cosine    | full_z        | +0.0495               | +5.40 | 1.2e-05  | YES |
| cosine    | across_halves | +0.0593               | +3.26 | 2.3e-03  | YES |
| cosine_2h | full_z        | -0.0229               | -5.82 | 1.00     | no  |
| cosine_2h | across_halves | -0.0520               | -3.60 | 0.999    | no  |

A single harmonic leaves a significant positive subgoal-progress effect; two
harmonics remove it. Rodent phase tuning (von Mises, kappa = 3.33) is sharp
enough to carry second-harmonic structure that one harmonic leaves behind.

Important for the manuscript: the choice does NOT manufacture the main effect.
Action Plan is essentially unchanged either way — full_z 0.2259 (t = 9.23) with
'cosine' vs 0.2324 (t = 9.57) with 'cosine_2h'; across-halves 0.2820 (t = 6.73)
vs 0.2964 (t = 7.21). All significant at q < 1e-5.

Caveat to state honestly: 'cosine_2h' does not leave Subgoal Progress at zero,
it leaves it reliably NEGATIVE (t(23) = -5.82). The one-sided criterion "not
significantly positive" is met, but describing it as "null" is inaccurate — it
is better described as over-corrected. Worth pre-empting, since a reviewer can
read it as over-residualisation.

## 2026-09-01 (later) — Full OSF release downloaded + uniform self-normalisation

The authors have now confirmed that **the normalisation they settled on is not
the one published** in `Basic_analysis.ipynb`, and it has not been shared. That
closes the earlier puzzle: the published `raw_to_norm` reproduces their released
`Neuron_*` arrays only to r ~ 0.88 because it is a different method, not because
we were running it wrong. Their released normalised files are therefore
unreproducible, and the only self-consistent option is to normalise everything
ourselves with one function.

**Downloaded (OSF 3d9r2):** all 25 combined ABCD recdays, 7 mice
(ab03, ah03, ah04, ah07, me08, me10, me11), 193 sessions, ~2.0 GB of
`Neuron_raw` / `Location_raw` / `trialtimes` / `Task_data`. Was 8 recdays /
5 mice. ab03 and ah07 are entirely new — they do not exist in the Drive share.

**New: `scripts/normalise_rodent_ephys.py`** + `raw_to_norm`, `normalise_segment`,
`state_boundaries` in `mc.analyse.analyse_ephys_clean`. Transcribed from the
authors' published `partition`/`normalise`/`raw_to_norm`. Output:
`derivatives/normalised_loc-max_<timestamp>/`, with a manifest and a settings
JSON recording every parameter and per-session shape. The raw release is never
modified.

    25 recdays / 7 mice / 193 sessions / 1252 neurons, 0 NaNs, all (n, trials, 360)

**Bug found and fixed in applying their code to Location.** Their `normalise`
stretches a state segment shorter than 90 raw bins by `np.repeat(x,10)/10`. The
`/10` is correct for a firing RATE and wrong for a categorical node ID — it
turns node 7 into 0.7. Their released `Location_*` arrays hold clean integers,
so they clearly do not divide there. `normalise_segment` now takes
`rate_scaled`, defaulting to True for 'mean' and False otherwise. Affects 1.12%
of state segments (109/9720) across 38 sessions; exact-bin agreement with the
released Location files rose 92.3% -> 93.9%.

**Undocumented choice, flagged:** the statistic for Location is not stated
anywhere by the authors. `--location-statistic` defaults to `max` (their
`take_max` option, and the only semantically sound one for node IDs). Agreement
with their released files: median 93.3%, max 92.3%, min 93.0%, mean 88.3% — no
statistic reproduces them, consistent with their method differing. Worth
confirming with them.

**CORRECTION to the earlier entry today.** The three "orphan" normalised
sessions (`ah04_05122021_06122021_3`, `ah04_09122021_10122021_3`,
`me10_09122021_10122021_8`) were downloaded and turn out to be **empty arrays**,
shape `(0,)`. So they were a genuine bad-session flag after all, just encoded as
an empty file rather than a missing one — the previous 61-session analysis was
correct and nothing was being wrongly discarded. `cross_view_session_ids` now
drops sessions that are absent OR empty in either view, so both encodings are
caught.

**`analysis_rodents_complete_clean.py`:** new `NORM_FOLDER` setting (now pointed
at the run above); the recday list is taken from whichever source supplies the
normalised view, so the analysis widens to 25 recdays automatically.
`load_ephys_data` gained `norm_folder`. Also added a **within-mouse robustness
test**: per-recday betas averaged within animal, then run through the IDENTICAL
`methods_results_stats` path (same one-sided t-test, same BH-FDR), written to
`key_analysis_stats.json` as `full_z_by_mouse` / `across_halves_by_mouse`.

Not yet re-run: all previously reported rodent numbers are from the authors'
normalised files at n = 8 recdays / 5 mice and are superseded.

Low-yield recdays to keep an eye on when the results land:
`me10_20122021_21122021` has 1 neuron and `me10_17122021_19122021` has 6, so
their per-recday betas will be very noisy.

## 2026-09-01 — Rodent data: the full release is on OSF, not the Drive share

New: `scripts/download_rodent_ephys_data.py` — per-file downloader (never
builds the multi-GB archive that makes the web download crash), verifies each
file with `np.load`, retries with back-off, resumes after a crash.

**Two sources, and they are not the same dataset.**

| | recdays | mice | raw | normalised 360-bin |
|---|---|---|---|---|
| OSF `3d9r2` (public release) | 25 | 7 | yes | **no** |
| private Google Drive share   | 14 | 5 | yes | yes, for 8 recdays |

Our 8 recdays came from the Drive. A recday is `{mouse}_{day1}_{day2}` — a
recording UNIT (two days spike-sorted together, 6 task configs), not an animal;
the 8 are ah03 x1, ah04 x3, me08 x1, me10 x1, me11 x2 = **8 recdays / 5 mice**.
The analysis docstring previously implied 8 animals; corrected, and
`key_analysis_stats.json` now carries a `settings.sample` block with
`n_recdays`, `n_mice`, `recdays_per_mouse`.

**Missing: 17 recdays, ~2.0 GB, including two entire mice.** `ab03` (3 recdays)
and `ah07` (3 recdays) are absent from the Drive share altogether. On disk vs on
OSF, per mouse: ab03 0/3, ah03 1/2, ah04 3/5, ah07 0/3, me08 1/3, me10 1/4,
me11 2/5. Five of the 25 (`combined_ABCDonly_notone_days.npy`) were recorded
without the state tones — a different sensory regime, kept separable via
`--tone-only`.

**Blocker on using them: OSF ships raw only.** The DSR analysis runs on the
normalised view (n_neurons x n_trials x 360, 90 bins/state), which OSF does not
include.

The authors' normalisation code IS public — `raw_to_norm` / `normalise` in
`Basic_analysis.ipynb` cell 21 of github.com/mohamadyelgaby/mFC_schema:

    Trial_times_conc = np.hstack((np.concatenate(tt[:,:-1]), tt[-1,-1])) // 25
    segments  = partition(raw_neuron, Trial_times_conc)      # one per state
    per_state = binned_statistic(arange(L), seg, 'mean', bins=90)[0]
                # with: if len(seg) < 90 -> seg = np.repeat(seg,10)/10 first
    Neuron_norm = per_state.reshape(n_states//4, 360)        # NO smoothing
                # (smoothing_sigma=10 applies only to raw_to_norm(return_mean=True))

Running it verbatim still does NOT reproduce the shipped `Neuron_*` files.
Over 18 sessions from 6 recdays: **mean r = 0.877** (range 0.75-0.97) for
neurons, **0.785** for locations, **zero exact matches**, and the trial count is
off by one in 8/18 sessions.

Cross-checked against OSF, not just the Drive: `Neuron_raw`, `Location_raw` and
`trialtimes` for ah03_18082021_19082021 (sessions 0, 2) and me08_10092021_11092021
(sessions 0, 2) are BIT-IDENTICAL between the two sources, and re-running the
authors' code on the freshly downloaded OSF raw gives exactly the same r
(0.8605 / 0.8050 / 0.9088 / 0.9648). So the mismatch is not a Drive-vs-OSF
artefact — the released raw is the same everywhere, and it still is not the
array that produced the released normalised files.

Why it cannot match: the bin values are exact rationals whose NUMERATORS agree
with ours but whose DENOMINATORS do not. For ah03_18082021_19082021_0, trial 0,
bin 0 the shipped value is 11/13 = `raw[5:18].mean()`, while their code on the
shipped raw gives 11/9 = `raw[0:9].mean()`. Same spikes, wider window, offset
start. I.e. **the published `Neuron_raw` is not the exact array that was fed to
`raw_to_norm`** — there is an alignment/binning difference upstream of the
released files. The off-by-one trial counts point the same way. So the recipe is
recoverable; the authors' exact output is not.

Consequence: do NOT mix the authors' normalised arrays for the old 8 with a
home-made version for the new 17 — the preprocessing difference lines up
exactly with the mouse/recday split and would confound the group test. Rebuild the normalised view from raw
for ALL 25 recdays with the `raw_to_norm` recipe above — that is now the only
self-consistent option, since the authors' own output cannot be reproduced.

**Fixed (Drive, changes results): 3 orphan normalised sessions restored.**
`Location/Neuron_ah04_05122021_06122021_3`, `..._ah04_09122021_10122021_3` and
`..._me10_09122021_10122021_8` existed on the Drive but had never been
downloaded. `cross_view_session_ids` drops sessions absent from the normalised
view assuming absence is the authors' implicit "bad session" flag; for these
three it was a download gap, so they were being discarded for no reason.
Downloaded 2026-09-01. Session counts now:

    ah04_05122021_06122021   7 -> 8
    ah04_09122021_10122021   7 -> 8
    me10_09122021_10122021   8 -> 9

Raw and normalised session lists now agree exactly for all 8 recdays (64
sessions, 504 neurons), i.e. the cross-view gate is currently a no-op. **The
analysis must be re-run — every number from before 2026-09-01 was computed on
61 sessions.**

Note on where raw is used: with `run_continuous: False` (the setting in
`analysis_rodents_complete_clean.py`) the raw branch of `process_one_recday` is
skipped entirely, so no reported result is computed from the raw files. They are
loaded only to build the `cross_view_session_ids` gate.

Indexing notes: gdown cannot enumerate the Drive folder (Google's folder HTML
caps at 50 entries per folder, the folder has 5037 files) — the script scrapes
`embeddedfolderview` instead. OSF is walked via its public API and cached as
`_osf_index.json`.

## 2026-08-29 — fMRI RSA: collinearity between the model RDMs of `DSR-contr_except_prev_but`

New: `scripts/plot_model_RDM_correlations.py` + `mc.plotting.results.plot_model_correlation_matrix_pub`.
Correlates the *model* RDMs of one combo GLM with each other (design
collinearity check for the RSA), and plots them as a 4 x 4 cm publication panel.

The model RDMs are built with the SAME code path as the searchlight RSA
(`pair_correct_tasks` -> per-model metric -> upper triangle, `diagonal_included:
false`), and, because the config sets `masked_conds: true`, restricted to the
same RDM cells the GLM is fit on — `make_category_masks(...,
mask_only_path_rew_combos=True)` keeps only same-type pairs (path-path and
reward-reward): 1,560 of 3,160 cells. No cells are dropped anywhere else.

Combo `DSR-contr_except_prev_but` (`rsa_config_quarters_DSR_controls.json`,
EVs `DSR_loc-fut-rews-state-dur-type`), n = 32 subjects with a local EV pickle,
group value = Fisher-z mean of the per-subject Pearson r:

|              | DSR   | location | A-state | l2_norm | next_buttons |
|--------------|-------|----------|---------|---------|--------------|
| location     |  .531 |          |         |         |              |
| A-state      | -.009 |  -.024   |         |         |              |
| l2_norm      |  .294 |   .570   |  -.018  |         |              |
| next_buttons |  .269 |   .199   |  -.014  |  .265   |              |
| buttons_out  |  .269 |   .163   |   .015  |  .224   |  .162        |

SD across subjects <= .054 everywhere, i.e. the design geometry is essentially
identical in every subject. Highest collinearity is location <-> l2_norm (.57)
and DSR <-> location (.53) — both expected (the DSR is built from the location
vectors; l2_norm is a graded version of location). A-state is orthogonal to
everything (|r| <= .024).

Outputs (mean/SD csv, per-subject matrices npy, settings json, pdf + png):
`data/derivatives/group/model_RDM_correlations_DSR-contr_except_prev_but_29-08-2026/`

## 2026-08-29 — mid-HC diagnosis: the "future-only" DSR is not a separate test, and the location result is a control-stack artefact

Diagnostic pass on `DSR_RSA_simple_ROI/2026-08-27_19-18-20` (latest),
`2026-07-30_15-58-51-fixed_cells-fixed_perms` and `2026-07-30_11-11-36`, to
resolve why mid HC carries the strongest concurrent-future β (Fig 2d) while its
cells are tuned to now / just-past (Fig 3c). No new runs; all numbers read from
stored results.

### 1. `dsr_fmri_fut` is a re-run of `dsr_fmri`, not an independent test

On the 4,560 valid RDM cells (HC_mid, `split_halves_z` mask):

    corr(dsr_fmri, dsr_fmri_fut) = 0.980

Dropping lag 0 removes 1 of 12 lag-windows from a Hamming distance over the
rolled 144-int trajectory, so the geometry barely moves. Consequences visible
in both runs: `ctrl_dsrFULL` and `ctrl_dsrFUT` return **identical p_perm to 3
dp** for every ROI (mPFC .022/.022, mOFC .866/.866, PCC .666/.666, HC_ant
.103/.103, HC_mid .001/.001) and identical control βs to 4 dp.

Also, lags 1 and 11 (30°, 330°) coincide with the current location on 39% of
bins (the autocorrelation figure already in Methods), so "future only" still
contains the present. **`ctrl_dsrFUT` cannot be cited as evidence that mid HC
codes the future.** Either drop it or replace it with a genuinely disjoint
model (e.g. lags 3–9 only).

Related: `corr(dsr_fmri, location) = 0.408` but `corr(dsr_fmri_fut, location) =
0.218` — dropping lag 0 does halve the location confound, yet the HC_mid β only
moves 0.073 → 0.068. So the mid-HC effect is not simply location leaking in
through lag 0.

### 2. The lag decomposition resolves the Fig 2d / Fig 3c tension

Joint quarter fit, `ctrl_dsrQUARTERS` in `2026-07-30_11-11-36` (4 quarters
compete against each other plus location + bttn_curr). Quarter k = lags
{3k, 3k+1, 3k+2}, i.e. curr = 0–60°, next = 90–150°, next2 = 180–240°,
next3 = 270–330°:

| ROI | curr | next | next2 | next3 |
|---|---|---|---|---|
| mPFC | .021 (p=.298) | .003 (p=.626) | .021 (p=.204) | .031 (p=.104) |
| mOFC | .012 (p=.155) | −.020 | .005 | −.005 |
| PCC | .021 (p=.200) | −.053 | −.005 | **.048 (p=.005)** |
| HC_anterior | **.074 (p=.001)** | −.013 | .003 | .011 (p=.294) |
| HC_mid | .046 (p=.059) | .020 (p=.243) | .010 (p=.477) | **.073 (p=.001)** |

Single-model (no controls, latest run) shows the same shape for HC_mid:
curr .076 (p=.001), next .033 (p=.026), next2 .026 (p=.054), next3 .088 (p=.001).

**There is no contradiction.** Each quarter correlates 0.607 with the full DSR
by construction, so a region that matches on 2 of 4 quarters yields a large
full-DSR β. Mid HC's fit is carried by the two quarters flanking the present
(now and just-past) — exactly the cell-level profile (0°, p=.0131; 330°,
p=.0449). Anterior HC is purely present. The full-DSR regressor simply cannot
report which lags carry it.

### 3. Caveat: the decomposition does NOT confirm the mPFC 30/60° peak

`dsr_fmri_informed` (lags 1,2 = 30°+60°, the pre-registered mPFC window) in
`ctrl_dsrInformed`: mPFC β = +0.0196, **p = 0.130 (n.s.)**, while HC_anterior
β = +0.0557 (p=.001) and HC_mid β = +0.0436 (p=.006). This is because
`informed` correlates 0.873 with `curr_quarter` and 0.187 with `location` — it
is largely a near-present model, which is why the hippocampi load on it.

mPFC's DSR fit sits numerically in the later quarters (next2 p=.078, next3
p=.072 single-model; next3 p=.104 joint) but nothing survives. With n = 65 mPFC
cells this may be power, but at the RDM level the mPFC lag profile does **not**
independently reproduce the cell-level 30/60° result. This should be stated
rather than glossed — it is the weak point Jensen and Dorrell will find.

### 4. The mid-HC location effect is a control-stack artefact, not a data change

Same run, same cells (`2026-07-30_15-58-51-fixed_cells-fixed_perms`), location
as the read-out sub-model:

| combo | members | HC_mid | HC_ant |
|---|---|---|---|
| `ctrl_dsrFULL` | state, location, bttn_curr, dsr_fmri | **+.0330 (p=.025)** | **+.0403 (p=.009)** |
| `fmri_ctrl_dsrFULL` | + **l2_norm, bttn_next** | +.0145 (p=.223) | +.0356 (p=.040) |

Latest run carries only the second stack: HC_mid +.0153 (p=.215), HC_ant
+.0330 (p=.052).

**The culprit is `l2_norm`**: `corr(location, l2_norm) = 0.588`. It is a second
parameterisation of the same variable (graded negative distance to each of 9
grid locations vs categorical location). Singly in HC_mid: location .0643
(p=.001), l2_norm .0631 (p=.002) — near-identical. Jointly they split the
variance and neither clears.

Not a data change: the 07-31 and 08-27 runs return the same numbers despite
HC_mid n going 143 → 145, and 07-30's `fmri_ctrl_dsrFULL` (+.0145) matches
08-27's `ctrl_fMRI-state_dsrFULL` (+.0153).

**Manuscript consequence.** Fig 3g's caption describes exactly `ctrl_dsrFULL`
("controlling for future locations, position in sequence and current actions")
— no l2_norm, no next button — while Fig 2d uses the stack that includes them.
The two figures therefore use different control models, and the location claim
survives only under the leaner one. Options, in order of preference:

  (b) Test `location` + `l2_norm` jointly as one spatial-code contrast rather
      than pitting two parameterisations of one variable against each other.
      Correct given r = 0.59, and removes the arbitrariness.
  (a) Harmonise on the full stack and report mid HC as n.s. (p=.215), ant HC
      marginal (p=.052). The present-coding claim then rests on the cell-level
      result and on the quarter split (HC_ant curr p=.001, HC_mid next3
      p=.001), which is stronger evidence anyway.
  (c) Drop l2_norm from the location model only — hard to justify while it
      stays in the DSR model.

### Next step

`ctrl_dsrQUARTERS` is currently commented out in `RSA_DSR_ROIs_simple.py`
(combo_models). The joint quarter numbers above come from the pre-relabelling
cell set (mOFC 85 vs 74, HC_ant 162 vs 171, HC_mid 143 vs 145), so it needs a
re-run on the current cells before it can go in the paper. That single figure
answers [79], [102], [85], [107] and [109] in the co-author comments.


## 2026-08-28 (night) — `within_only` scope: the right fix for the instruction question

The user's suggestion -- drop the across-half block entirely -- is better than
the block nuisance regressor, and it is now implemented as
`data_rdm_scope = "within_only"`.

**Why it removes the artefact structurally.** The bias came from a similarity
offset BETWEEN the within-half and across-half blocks (data: 0.828 vs 0.863
dissimilarity, within < across in 89% of searchlights) which the instruction
regressors encode (r = +0.42 to +0.54 with an across-half indicator). Keep only
within-half cells and that contrast does not exist, so no regressor can absorb
it. Unlike the nuisance regressor, nothing has to be modelled away.

**It keeps the contrast the design was built for.** Of the 90 within-half
cells, 10 are same-task-letter pairs (the two directions of one task inside one
half). On exactly those cells the instruction dissimilarity is **0** (they saw
the same sequence) and the execution dissimilarity is **1** (they execute the
reverse). That is the instruction x execution dissociation, and it lives
entirely inside the within-half block.

**Empirical check** (sub-02, TR4, 3161 searchlights, mean single-subject t):

| combo | scope | instr | exec |
|-------|-------|-------|------|
| rewDSR_vs_instr | full_no_diag | +0.268 | -0.254 |
| rewDSR_vs_instr | **within_only** | -0.472 | **+0.058** |
| splitDSR_vs_instr | full_no_diag | +0.120 | -0.063 |
| splitDSR_vs_instr | **within_only** | **-0.081** | **+0.011** |

The mirror-image offset is gone. The residual -0.47 for `rewDSR_instr` in the
2-regressor combo is not obviously an artefact: a negative instruction beta on
within-half cells means same-instruction pairs are LESS similar than the
instruction model predicts, which is what you expect if the same-letter pair
(same instruction, reversed execution) is dominated by execution coding.

**No temporal-proximity confound.** Within a half, same-task pairs are closer
in time in TH1 (mean gap 362 vs 594 s) but FURTHER apart in TH2 (575 vs 528 s)
for sub-02 -- the ordering is not systematic, and the gaps are hundreds of
seconds, far outside BOLD autocorrelation.

**No recomputation needed.** `within_only` reads the same `data_RDM_full.npy`
cache as `full_no_diag` and keeps 90 of its 190 columns. Cell ordering is
`np.tril_indices(n_all, k=-1)` in both `get_full_instruction_RDM_per_searchlight`
and `_lower_tri_flat`, verified, so the mask aligns. If those caches survive on
the cluster this is a re-fit, not a re-run.

**Implemented** in `scripts/fMRI_run_RSA_instruction.py`: `within_half_mask()`,
`data_rdm_scope = "within_only"` accepted, model regressors and the cached data
RDM both subset by the same mask, and the `block` nuisance now asserts against
`within_only` (constant there, and unnecessary). Config
`condition_files/rsa_instruction_within_and_across_th.json` renamed to
`rsa_instruction_within_th_only.json`, `name_of_RSA =
within_th_only_intr-vs-exe`, scope `within_only`, block nuisance dropped.

Verified: all 10 single models and all 3 combos are full rank on 90 cells
(2/2, 3/3, 5/5, 9/9).

**The figures and printouts were WRONG for the new scope and are now fixed.**
Three places still described the across block regardless of `data_rdm_scope`:

1. The model-RDM figures always plotted `model_RDM_dir[model]`, the (n, n)
   across block, and the assembled (2n, 2n) figure only fired for
   `full_no_diag`. In `within_only` that meant the saved figure showed cells
   that are not fitted -- and for every instruction model it is a uniform block
   of 1.0, carrying no information at all. Replaced with `_display_RDM()`,
   which returns the matrix for the current scope with excluded cells set to
   NaN (`plot_instruction_RDM` already renders NaN white via
   `masked_invalid` + `set_bad`). Filenames now carry the scope:
   `{results_dir}_{model}_{data_rdm_scope}`.
2. The example data-RDM figure had the same problem; same fix.
3. The printed "execution vs instruction Pearson r" fell through to the
   across-block branch for any scope other than `full_no_diag`, so in
   `within_only` it would have reported a correlation against a constant
   vector. It now always uses `_model_regressor()`, i.e. exactly the vectors
   the OLS sees. `_model_regressor` was moved above the verification block so
   both share one definition.

Also `verify_instruction_rdm_blocks` now runs on the matrix actually fitted
(across block for `across_only`, the assembled matrix otherwise) and prints
which scope it checked.

Rendered all three scopes for `rewDSR_instr` and `rewDSR` to confirm: 100 / 400
/ 200 cells shown for across_only / full_no_diag / within_only, and
`rewDSR_instr` under `across_only` is `unique = [1.]` -- the degenerate uniform
block, now impossible to mistake for a real model RDM.

**Config as it will run:** `within_th_only_intr-vs-exe`, scope `within_only`,
TR set per run, 10 single models + 3 combos (`rewDSR_vs_instr` 2,
`instr_split` 4, `splitDSR_vs_instr` 8) = 24 beta/t/p map sets per subject per
TR. No block nuisance (nothing left to absorb).

**Status of the three scopes.**
- `across_only` -- correct for EXECUTION. Instruction is constant, so it cannot
  be tested there at all.
- `within_only` -- correct for INSTRUCTION, and it also carries the
  instruction x execution dissociation. Execution is estimable but from
  within-run cells only.
- `full_no_diag` -- mixes the two and introduces the block offset. Superseded;
  use it only with the block nuisance, and only as a control.

## 2026-08-28 (evening) — the full_no_diag t-bias diagnosed, and a block nuisance regressor

**The bias is real and it is not regressor collinearity.** Whole-brain t over
all voxels x 12 TRs, `instr_test_full`:

| map | mean t | % voxels > 0 |
|-----|--------|--------------|
| rewDSR_instr (single model) | **+2.07** | 97.0% |
| REWDSR_INSTR-rewDSR_vs_instr | +2.06 | 97.0% |
| rewDSR (single model) | **-1.04** | 15.7% |
| REWDSR-rewDSR_vs_instr | -1.03 | 16.0% |
| simple | -1.18 | 13.7% |

Single-model fits show the same offset, so it is not suppression between
regressors -- and r(rewDSR, rewDSR_instr) = -0.004 anyway. For comparison, the
`across_only` maps sit at -0.48 to +0.71.

**Cause: a within/across block offset in the DATA that the instruction models
encode.** Of the 190 lower-triangle cells, 90 are within-half and 100 across.

- Data (sub-02 TR4, 126 404 searchlights): mean cosine dissimilarity **0.828
  within vs 0.863 across**, a 4.0% offset, with within < across in **88.9%** of
  searchlights. Run-level noise.
- Regressors, correlation with an across-half indicator: every `_instr` model
  **+0.42 to +0.54** (within mean 0.711, across constant 1.000); every
  execution model **-0.13 to -0.17** (0.911 vs 0.820).

An instruction regressor therefore says exactly what the run noise says, and
collects a large positive beta in nearly every voxel; execution collects the
mirror-image negative one.

**The existing group maps CANNOT be corrected post hoc.** 82% of the block
indicator is a direction orthogonal to the old design `[1, instr, exec]`, i.e.
information the fit never computed. Predicting the correctly-fitted beta from
everything that was saved (beta and t for both regressors, 3152 searchlights):
R^2 = **0.577** for instruction, 0.936 for execution. Not a correction, and not
close enough to be one.

**But a re-fit may not need the expensive step.** The searchlight data RDMs are
cached as `data_RDM_full.npy` and `fMRI_run_RSA_instruction.py` already skips
computation when the cache exists. 99.8% of searchlights have no NaN cell (only
3 distinct NaN patterns among the other 293), and the OLS for all 126 111 clean
searchlights at one design takes **21 s** in the current per-searchlight loop
(0.6 s vectorised via `evaluate_model_vec`, 34x). So if those caches survive on
the cluster this is minutes per subject-TR, not days. Worth checking before
committing to a long re-run.

**Implemented:** `build_block_nuisance_RDM()` in
`scripts/fMRI_run_RSA_instruction.py`, reserved regressor name `block`, and a
config flag `add_block_nuisance: true` that appends it to every combo so it
cannot be forgotten in one. Asserts `full_no_diag` scope (in `across_only` it
would be constant). Enabled in
`condition_files/rsa_instruction_within_and_across_th.json`.

Verified on sub-02 TR4: all three combos stay full rank with it
(4/4, 6/6, 10/10), and over 3161 searchlights the mean single-subject t for
`splitDSR_vs_instr` moves from instr +0.120 / exec -0.063 to
**instr -0.101 / exec +0.003 / block +1.319** -- the nuisance takes the offset
and execution recentres on zero.

**What it fixes and what it does not.** After the nuisance absorbs the offset,
where each regressor's remaining variance lives:

| regressor | % within-half | % across-half |
|-----------|---------------|---------------|
| rewDSR_instr | **100.0%** | 0.0% |
| curr_rew_instr / two_next_rew_instr | **100.0%** | 0.0% |
| rewDSR | 21.4% | 78.6% |
| curr_rew / two_next_rew | 33.1% | 66.9% |

Execution keeps 67-79% of its variance in across-half cells, so the nuisance
makes those fits interpretable. Instruction keeps **zero** -- across halves the
same task is instructed in the reverse order, so those cells carry no
instruction information at all and the estimate is always a purely within-run
comparison, where "same instruction" is also "same stimulus, same run". No
nuisance regressor can change that; it is a property of the counterbalancing.

**Recommendation:** the re-run buys trustworthy execution numbers in
`full_no_diag` (which `across_only` already provides more cleanly) and a clean
demonstration that the instruction effect was the block artefact -- a good
supplementary control. It does NOT make the instruction models usable as
evidence for instruction coding.

## 2026-08-28 (later) — all three per-TR datasets on identical footing

All three now run through `scripts/per_TR_loso.py` / `mc.analyse.loso`, same 5
masks, same seeds, 10 000 SVC perms, LOSO k=50/100/200, whole-brain at 1000.

| # | dataset | maps | scope | output |
|---|---------|------|-------|--------|
| 1 | `instr_test_full` | 27 (25 usable) | `full_no_diag` | `per_TR_svc_instr_test_full_allTR_2026-08-28` |
| 2 | `split_rew_DSR_per_TR` | 8 | `across_only` | `per_TR_svc_split_rew_DSR_allTR_2026-08-27` |
| 3 | `instruction_per_TR` | 1 (rewDSR) | `across_only` | `per_TR_svc_instruction_rewDSR_allTR_2026-08-28` |

**The reported number reproduces exactly** through the new runner:

    reported   : t=5.079 TR4 MNI -6/32/18 p_FWE=.0407 n_vox=4181
    new runner : t=5.079 TR4 MNI -6/32/18 p_FWE=.0407 n_vox=4179

(The 2-voxel difference is the mask-intersection change; it does not move the p
at four decimals.) Its LOSO is p_FWE = .0144 at TR4. Nothing in MTL or visual
(best p = .47), nothing whole-brain (p = .451).

**Dataset 1 result: every single significant map is an instruction model, and
NO execution map is significant in ANY mask.**

| mask | sig maps | best instruction | best execution |
|------|----------|------------------|----------------|
| mPFC | 6/25 | curr_rew_instr t=6.94 TR2 -6/66/18 **p=.0002** | NEXT_REW-splitDSR_noInstr t=3.77 p=.313 |
| MTL_L | 8/25 | rewDSR_instr t=8.30 TR5 -32/-2/-34 **p<.0001** | THREE_NEXT_REW-splitDSR_noInstr t=3.20 p=.482 |
| MTL_R | 7/25 | curr_rew_instr t=6.41 TR0 22/0/-24 **p=.0002** | t=2.76 p=.665 |
| visual | 6/25 | CURR_REW_INSTR-splitDSR_vs_instr t=8.52 TR3 **p<.0001** | t=4.72 p=.136 |

0 of 10 execution maps reach p<.05 in any of the five masks. Whole brain: 7 of
25 significant, all instruction, peaking t=10.3 at TR2 (rewDSR_instr, 0/-32/-10).

**The scope, not the model, decides the answer.** The same subjects and the
same rewDSR construct give t=5.08, p=.041 in mPFC under `across_only`
(dataset 3) and nothing at all under `full_no_diag` (dataset 1, best mPFC
execution p=.313), while the instruction models go from structurally
impossible (constant regressor) to t=6.9-10.3. Adding the within-half cells
does not add power to the execution test -- it destroys it and replaces it
with a large instruction effect. That is what the within-half confound
predicts: within a half, "same instruction" IS the same visual stimulus in the
same run, so those cells inject stimulus-repetition structure that the
instruction regressor fits and the execution regressor does not.

**Timing supports the stimulus reading.** Peak TRs of the significant
instruction maps cluster early -- TR0:3, TR1:4, TR2:11, TR3:5, TR4:2, TR5:4,
TR6:6 (mode TR2) -- and the LOSO timecourses
(`per_TR_timecourses_instr.pdf`) rise at TR1-3 and decay, the shape of a
response to a screen that is on from 0 s, not of a plan assembled once all
four rewards are known at 6 s. Compare dataset 2, whose across-half execution
channels peak LATE (mPFC two_next TR4-6, left MTL next_rew TR7-8).

**Conclusion for the manuscript:** `across_only` is the defensible scope for
these questions. `full_no_diag` should not be used to compare instruction
against execution, because it is exactly the scope in which the two are
confounded. The instruction models in dataset 1 should not be reported as
evidence for instruction coding.

**Robustness fixes made while running this:**
- `_load_with_retry` -- the first 12-TR `instr_test_full` attempt died at model
  14/27 when `nib.load` transiently failed on an intact TR7 file (sync-backed
  storage). Reads now retry 4x with a 10 s wait; genuinely corrupt files still
  raise.
- `--resume` -- skips models whose outputs exist and rebuilds both summary
  tables from every per-model json on disk, so an interrupted run resumes
  without redoing finished work and still writes complete tables. Verified:
  "resume: 13 of 27 models already complete, 14 to run" -> "summary table
  covers 27 of 27 models".
- `load_ref` now intersects the group masks over the TRs that HAVE one and
  prints how many it used. The `instruction_per_TR` folders only ship
  `mask_all_32_subjects` for TR0 and TR3 -- which is why the original script
  read TR0's and stopped -- and this would otherwise have been a hard failure.
- `base_channel` strips a trailing `_instr`, so a reward channel keeps one
  colour across its execution and instruction variants.

## 2026-08-28 — per-TR LOSO analysis refactored into one runner + one library

The scripts folder had grown four files for one analysis. Consolidated:

**`mc/analyse/loso.py`** (new, registered in `mc/analyse/__init__.py`) holds
everything: inputs (`resolve_nii`, `load_ref`, `load_mask`, `load_masks`,
`discover_models`, `read_model_columns`), statistics (`tstat`, `null_max_t`,
`adaptive_pblock`, `voxel_fwe_p`, `perm_wholebrain`), tests (`run_svc`,
`run_loso`, `run_wholebrain`), volume writing (`vol_from_cols`,
`write_mask_maps`, `write_wholebrain_maps`), results loading (`load_loso`,
`load_settings`, `result_masks`) and plotting (`plot_per_TR_timecourses` plus
the CLAUDE.md palettes and the reward schedule).

**`scripts/per_TR_loso.py`** (new) is the only runner, with three modes:
`--mode run` (analyse, no figures), `--mode plot` (load an existing
`--out-dir` and plot, no recomputation), `--mode both` (default).
In plot mode it reads `settings.json` and defaults to the four reward channels
when present, else every model.

**Archived** to `scripts/old/per_TR_loso_pre_refactor/` with a README:
`svc_loso_test.py`, `svc_loso_batch.py`, `plot_per_TR_timecourses.py`,
`hemisphere_contrast.py` (the last retired at the user's request).

**Equivalence verified before archiving** — both implementations run on the
same input:

    tstat       bit-identical: True
    null_max_t  bit-identical (pblock = 250 and 1000): True
    run_loso    bit-identical to the inline LOSO of svc_loso_test.main(): True

One deliberate behavioural difference remains, as before: `load_ref` intersects
the group mask across all included TRs rather than taking TR0's alone (~21
voxels; mPFC 4181 -> 4182), so the reported `BA32-9-10` p will not reproduce
bit-for-bit. Documented in the archive README.

**References repointed:** `scripts/future_step_dominance_mPFC_lOFC.py` imported
`tstat, null_max_t` from `svc_loso_test` and now imports them from
`mc.analyse.loso`. Prose references updated in
`scripts/fMRI_run_RSA_instruction.py`, `scripts/mask_stats.py`,
`docs/rerun_after_roi_update.md` and
`scripts/old/instruction_phase_alternatives/README.md`.

**Still duplicating this code:** `scripts/mask_stats.py` and
`scripts/mask_stats_spyder.py` carry their own copies of `load_ref`,
`load_mask`, `extract_betas`, `tstat` and `null_max_t` (they compare voxel-wise
FDR against permutation FWE). They pre-date this work and were left alone; they
are the obvious next thing to fold into `mc.analyse.loso`.

## 2026-08-27 (evening, later) — HC/EC hemisphere split: next_rew is left-lateralised

**Masks:** `Garvert_MTL_2mm.nii.gz` split at MNI x = 0 into
`data/masks/Garvert_MTL_2mm_L.nii.gz` (1352 vox, 1332 in-brain) and
`..._R.nii.gz` (1364 vox, 1354 in-brain), provenance in
`Garvert_MTL_2mm_hemispheres.json`. No voxel sits at x = 0, and the two are
near-symmetric in size, so their FWE thresholds are comparable
(t_crit = 4.70 vs 4.67).

**Re-ran** `per_TR_svc_split_rew_DSR_allTR_2026-08-27` with 5 masks
(mPFC, MTL, MTL_L, MTL_R, visual). Same seeds/data, so mPFC / MTL / visual and
the whole-brain maps are unchanged; the folder now also holds the hemispheres.

**SVC, next_rew:**

| mask | peak t | TR | MNI | p_FWE | LOSO peak | LOSO p |
|------|--------|----|-----|-------|-----------|--------|
| MTL bilateral | 5.64 | 7 | -12/-38/-10 | .0116 | TR7 t=2.40 | .0510 |
| **MTL left** | 5.64 | 7 | -12/-38/-10 | **.0054** | TR8 t=3.07 | **.0124** |
| MTL right | 3.68 | 4 | 24/-36/2 | .3224 | TR2 t=0.53 | .6277 |

Splitting HELPS: the same peak voxel goes from p = .0116 bilaterally to
p = .0054 in the left mask, because halving the search volume lowers the
threshold while the effect is entirely on the left. LOSO likewise strengthens
(.051 -> .012, 3 significant seconds TR7/8/9).

**Direct L-R contrast** (`scripts/hemisphere_contrast.py`, new). "Significant
in L, not in R" is not a lateralisation test, so this tests L - R itself on the
per-subject LOSO held-out arrays (each hemisphere selected its own top-k on
n-1 subjects, so the paired difference stays unbiased), with the same
`null_max_t` sign-flip null corrected over the 12 seconds:

| channel | largest \|L-R\| | t(L-R) | p_FWE(L>R) |
|---------|---------------|--------|------------|
| **next_rew** | TR7 | **+2.83** | **.0243** |
| two_next_rew | TR10 | +1.91 | .1479 |
| three_next_rew | TR0 | -1.53 | .2648 (R>L) |
| curr_rew | TR0 | -1.22 | .4399 (R>L) |

So the left-lateralisation of `next_rew` is a real difference, not just a
difference in significance — L > R at TR7 (and TR8, p = .027), FWE-corrected
over seconds. No other channel is lateralised either way.

Left HC/EC t per second for next_rew: 0.14, 0.20, 0.51, 0.79, 0.99, 1.58,
2.36, 2.89, 3.07, 2.64, 1.85, 1.37 — a slow build peaking at TR7-8, i.e. during
the fast second pass, not when B is first shown at 1.5-3 s. Right HC/EC is flat
throughout (max \|t\| = 0.53).

**Figures:** `per_TR_timecourses_MTL_hemispheres.pdf/.jpeg` (+ `_peaks.csv`),
`hemisphere_contrast_MTL_L_vs_MTL_R.csv`.

**Caveat unchanged:** 8 models x 5 masks now, uncorrected across that family.
The lateralisation contrast is 4 tests; next_rew at .024 would not clear
Bonferroni over 4 (.0125). It confirms the direction of an effect selected on
other grounds rather than establishing it independently.

## 2026-08-27 (evening) — split_rew_DSR across all 12 TRs: no reveal staircase

**Data:** `group_RSA_split_rew_DSR_per_TR_glmbase_01-TR{0..11}_cropped` — all 12
TRs present and intact (unlike `instr_test_full`, which is still 11/12
truncated). Its `sub-XX_settings_summary.json` has no `data_rdm_scope` key, so
it ran the default **`across_only`**: every RDM cell is an across-half
comparison, which makes it free of the within-half same-stimulus confound that
limits the `full_no_diag` instruction models. Execution channels only — there
are no `_instr` models in this run.

**Analysis:** `svc_loso_batch.py`, 3 masks, 10 000 SVC perms, LOSO k=50/100/200,
whole-brain maps at 1000 perms. Output
`data/derivatives/group/per_TR_svc_split_rew_DSR_allTR_2026-08-27/`
(+ `per_TR_timecourses.pdf/.jpeg/_peaks.csv` from the new
`scripts/plot_per_TR_timecourses.py`).

**Timing ground truth.** `create_EVs_for_RDMs.py` builds the `01-TR{n}` EV as a
1-s boxcar at `instruct_start + n`, HRF-convolved by FEAT — so the TR axis is
neural seconds with no lag to add back. `show_rewards` in
`mc/latest_experiment/3x3_fMRI_part1.py` shows ONE reward at a time: A 0-1.5,
B 1.5-3, C 3-4.5, D 4.5-6, then a faster refresh A 6-7, B 7-8, C 8-9, D 9-12.

**Result: neither the sequential-reveal staircase nor a synchronous rise at
TR6.** LOSO t per second (k=100), mPFC:

| channel | TR0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curr (A) | -1.25 | -2.08 | -1.66 | -0.92 | -0.51 | -0.59 | -0.81 | -0.92 | -0.90 | -1.13 | -1.46 | -1.60 |
| next (B) | -0.30 | -0.83 | -0.22 | 0.83 | 1.91 | **2.48** | 2.34 | 2.07 | 1.94 | 1.70 | 1.20 | 0.68 |
| two_next (C) | 0.10 | -0.03 | 0.97 | 2.27 | **3.45\*** | **3.47\*** | **2.82\*** | 2.15 | 1.23 | 0.16 | -0.36 | -0.37 |
| three_next (D) | 0.27 | 0.65 | 1.32 | **1.70** | 1.56 | 1.02 | 0.41 | 0.06 | -0.13 | -0.21 | -0.46 | -0.70 |

- `curr_rew` (reward A) is **never** represented — flat or negative at every
  second, in every mask.
- Peaks run D (TR3) -> C (TR4-6) -> B (TR5-8): if anything the REVERSE of the
  reveal order, and nothing like a double sweep.
- `two_next_rew` is the only channel with FWE-significant LOSO seconds
  (TR4/5/6, p < .05 corrected over the 12 seconds).
- MTL: only `next_rew`, SVC t = 5.64 at **TR7**, MNI -12/-38/-10,
  p_FWE = .012 — five seconds after B is first shown, but coincident with B's
  second appearance (7-8 s).
- Whole brain: only `next_rew` (t = 6.97, TR7, p_FWE = .014) and its combo
  regressor (t = 7.27, TR7, p_FWE = .010). Peak MNI -10/70/-4 is at the very
  anterior edge of the brain mask — check that it is not a smoothing/edge
  artefact before believing it.
- Occipital: nothing (best p_FWE = .144).

**The reported mPFC effect decomposes into the MIDDLE two channels.** The
published instruction-phase result is parent `rewDSR`, t = 5.08 at TR4,
MNI **-6/32/18**, p_FWE = .041. In the split:
`next_rew` t = 4.76 at TR4, MNI **-6/32/18** (p = .081), and `two_next_rew`
t = 4.51 at TR4, MNI **-8/32/18** (p = .113) — the same voxel and the same
second. `curr_rew` (t = 2.64, p = .93) and `three_next_rew` (t = 3.12, p = .74)
contribute nothing. So the mPFC effect is carried by rewards B and C, not by
where the subject is now and not by the last reward. Neither child alone beats
the parent, consistent with them contributing jointly rather than one driving it.

**Multiple comparisons:** 8 models x 3 masks plus 8 whole-brain tests, not
corrected across that family (per the standing request). Bonferroni over 8
models would need p < .00625; `next_rew` whole-brain (.010/.014) and MTL (.012)
do not clear that. Treat the decomposition as the robust part and the
individual p-values as suggestive.

**Still open:** the same split for the INSTRUCTION channels needs the
`instr_test_full` download to finish, and it will carry the within-half
confound, so it is not directly comparable to this run.

## 2026-08-27 (later still) — whole-brain t / FWE-p / uncorrected-p volumes added

**Script:** `scripts/svc_loso_batch.py`, new `--wholebrain` branch.

Motivation: the SVC test only ever reported numbers inside a mask, so there
was nothing to scroll through in fsleyes. `--wholebrain` now writes, per model,
in `wholebrain/`:

| file | what |
|------|------|
| `{model}_t.nii.gz` | observed group t, all brain voxels |
| `{model}_1minusFWEp.nii.gz` | 1-p, FWE over whole brain x all TRs (max-t null) |
| `{model}_1minusp_uncorr.nii.gz` | 1-p, uncorrected, that voxel's own permutation p |
| `{model}_summary.json`, `{model}_null_max_t.npy` | peak stats, the null itself |
| `wholebrain_summary_table.csv` | one row per model |

3-D for a single TR, **4-D (X, Y, Z, TR)** as soon as several TRs are passed —
so the fsleyes TR slider scrubs the instruction period. `--wholebrain-neg`
adds the negative-direction p maps; `--wholebrain-models` restricts which
models get volumes.

**Implementation notes.**
- One read per model still serves everything: with `--wholebrain` the
  extraction target becomes the whole brain mask and each ROI is a column
  subset of it, so adding whole-brain costs no extra I/O.
- `perm_wholebrain()` keeps only the max-t null plus a per-voxel exceedance
  tally, never the (n_perm x n_vox) null, so memory is set by the block size
  and not by n_perm. Blocks are sized to ~1e7 floats.
- The permutation t uses the same sign-flip identity as
  `svc_loso_test.null_max_t` (`var = (S2 - n*M^2)/(n-1)`). To make CLAUDE.md
  rule 4 checkable rather than assumed, the function **asserts** that the
  all-plus-one flip reproduces `tstat(D)` on the observed data — empirical and
  permutation statistic are therefore verifiably the same statistic.
- `n_p_FWE_lt_05` in the whole-brain summary is counted off the p map itself,
  not off the 95th-percentile t threshold; at low n_perm the two drift apart
  (1320 vs 1324 voxels at 200 perms) and only the former matches what you get
  by thresholding the saved map at 0.95.

**Interpretation guard rails, written into the docstrings.** The whole-brain
FWE null corrects over the entire brain mask AND every included TR at once, so
it is far stricter than the small-volume p in the mask folders and the two must
not be compared. The uncorrected map is for looking around, not for claims.
A voxel no permutation beat gets p = 0 / 1-p = 1; that means p < 1/n_perm, not
a real zero.

**Verified** (rewDSR_instr, TR5, 200-perm smoke run): map peak t = 8.8615 at
MNI -34/-2/-34 matches the summary json exactly; 147 358 in-brain voxels
match; uncorrected p <= FWE p at every brain voxel.

**TR5 re-run with the maps** (1000 whole-brain perms, 147 358 voxels; the SVC
mask numbers are unchanged from the earlier entry). 7 of 25 usable maps survive
whole-brain FWE, and every one of them is an `_instr` model:

| model | peak t | MNI | p_FWE | vox p_FWE<.05 |
|-------|--------|-----|-------|---------------|
| rewDSR_instr | 8.86 | -34/-2/-34 | <.001 | 1406 |
| REWDSR_INSTR-rewDSR_vs_instr | 8.86 | -34/-2/-34 | <.001 | 1406 |
| curr_rew_instr | 7.76 | -14/-22/10 | <.001 | 1086 |
| CURR_REW_INSTR-splitDSR_vs_instr | 7.10 | 46/-46/-14 | .001 | 394 |
| two_next_rew_instr | 6.77 | -12/-22/8 | .002 | 109 |
| next_rew_instr | 6.29 | -26/0/4 | .005 | 55 |
| three_next_rew_instr | 6.00 | -12/-22/10 | .008 | 68 |

No execution model comes close (best non-`_instr` whole-brain p = .129). Note
the peaks sit in thalamus / temporal pole / posterior insula rather than in any
of the a-priori regions — consistent with the within-half same-stimulus
confound noted in the entry below, and a further reason not to interpret these
until that control is run.

Output size: 33 MB of volumes for 27 models at one TR; expect ~0.4 GB and
roughly 45-90 min for the full 12-TR sweep.

## 2026-08-27 (later) — instruction models are degenerate in `across_only`, fine in `full_no_diag`

**Diagnosed on sub-02's actual model vectors** (`rewDSR` at the A_reward anchor,
`condition_files/rsa_instruction_full.json` settings).

**Why the across-half instruction RDM is constant.** Within one half, forw and
backw saw the SAME instructed sequence (backw reverses it mentally), so
`instruction_relabel_dict` is correct. But the same task letter is instructed in
the OPPOSITE order in the two halves — task A: half-1 instruction = 1,7,5,3,
half-2 instruction = 3,5,7,1. Reversing four distinct reward positions leaves no
slot matching, so every same-letter across-half cell is a Hamming mismatch (1),
and different-letter cells are 1 as well. The entire TH1 x TH2 block is
therefore uniformly 1.0 — zero variance, for `rewDSR_instr` and all four
`*_rew_instr` split channels alike.

**Consequence — `across_only` is dead for every instruction model.**
`evaluate_model_vec` zeroes a constant column, then `matrix_rank(XtX) < n_aug`
makes it return NaN for EVERY regressor in the design, and
`save_my_RSA_results` writes those NaNs as an all-zero map with no error. Ranks
in `across_only`: `rewDSR_vs_instr` 2/3, `instr_split` 1/5,
`splitDSR_vs_instr` 5/9 — all dead.

**`full_no_diag` is NOT ill-defined.** The within-half blocks W1/W2 carry the
instruction structure ("same letter within this half" = same stimulus = 0), so
the regressors have real variance (std 0.266 for `rewDSR_instr`, 0.344 for the
split channels) and every requested design is full rank: `rewDSR_vs_instr` 3/3,
`instr_split` 5/5, `splitDSR_vs_instr` 9/9, max |r| between regressors 0.644.
Instruction and execution are close to orthogonal there: r = -0.004 (rewDSR),
+0.020 / +0.109 / +0.109 / +0.020 (curr/next/two/three).

**But the identifying variance is entirely within-half.** All 100 across-half
cells sit at the same value, so no across-half pair can inform an instruction
beta — it is estimated only from the 90 within-half cells. Within a half,
"same instruction" is identical to "same task letter", i.e. the same visual
stimulus in the same run. That is precisely the shared-run-noise bias the
`data_rdm_scope` docstring flags and does not correct. The TR5 SVC result
(MTL t = 8.3, visual t = 6.2 for `rewDSR_instr`) is therefore most likely a
within-run same-stimulus effect and must not yet be read as instruction coding
in EC/HC. A within-half-only null (or splitting W1 vs W2) is the control to run.

**Cause of the two all-zero maps found in the earlier TR5 run.** Combo
`rewDSR_noInstr` = [`rewDSR`, `simple`]: `simple` is finite in only 30 of 190
lower-triangle cells, and on exactly those 30 cells `rewDSR` and `simple`
correlate at **r = 1.0** -> rank 2/3 -> NaN for both -> all-zero maps. Not a
bug in the group test. Do not re-run that combo as specified.

**Changes.**
- New `condition_files/rsa_instruction_within_and_across_th.json`
  (`name_of_RSA = within_and_across_th_intr-vs-exe`, `data_rdm_scope =
  full_no_diag`): 10 single models plus combos `rewDSR_vs_instr` [rewDSR,
  rewDSR_instr], `instr_split` [the four `*_rew_instr` channels], and
  `splitDSR_vs_instr` [all 8 exec+instr split channels, as in the old file].
  `simple` dropped.
- `scripts/fMRI_run_RSA_instruction.py`: new `design_rank_report()` plus a
  pre-flight loop that checks every single model and every combo for constant
  regressors / rank deficiency BEFORE any searchlight OLS runs, and raises with
  the offending list instead of silently writing zero maps. Verified: it passes
  all 13 designs of the new config and flags exactly `rewDSR_noInstr` in the old
  one.

## 2026-08-27 — SVC max-t + LOSO over all 27 maps of `instr_test_full`, TR5 only

**Scripts:** `scripts/svc_loso_batch.py` (new), `scripts/svc_loso_test.py` (patched
to resolve `.nii`/`.nii.gz`).
**Output:** `data/derivatives/group/per_TR_svc_instr_test_full_TR5_2026-08-27/`
— `settings.json`, `summary_table.csv`, `run.log`, and per mask/model:
`_svc_summary.json`, `_loso_results.json`, `_loso_k{K}.npy`, plus the volumes
`_t.nii.gz` (observed t), `_voxelFWEp.nii.gz` (voxel-wise FWE p against the
max-t null), `_voxel1minusFWEp.nii.gz` and `_voxel1minusFWEp_neg.nii.gz`
(threshold at 0.95 for p_FWE < .05), and `_null_max_t.npy` (the null itself,
for re-thresholding). 3-D here because one TR was analysed; 4-D (X,Y,Z,TR) when
several are. Neither `svc_loso_test.py` nor the first version of
`svc_loso_batch.py` wrote volumes — added 2026-08-27, verified to reproduce the
json peaks exactly.

**Test.** Identical to the reported instruction-phase test — one-sample t over
32 subjects, sign-flip max-t permutation (10 000 perms, seed 0) corrected over
all voxels in an a-priori mask, plus the LOSO cross-validated readout at
k = 50/100/200. `tstat`, `null_max_t` and the LOSO selection are imported from
`svc_loso_test.py`, so empirical and permutation values come from the same code.

**Masks (3).**
- `mPFC` = `masks/mask_PFC_LR_smoothed_resampled.nii.gz` (BA32/mBA9/mBA10), 4182 vox
- `MTL` = `masks/Garvert_MTL_2mm.nii.gz` (HC/EC), 2695 vox
- `visual` = `masks/visual_occipital_HO25_2mm.nii.gz` (NEW: union of the 8
  occipital Harvard-Oxford cortical labels at thr25 — LOC sup/inf,
  Intracalcarine, Cuneal, Lingual, Occipital Fusiform, Supracalcarine,
  Occipital Pole), 12 497 vox in-brain of 27 220 in the atlas. The 7T FOV
  truncates the occipital pole (3 % covered) and occipital fusiform (1 %), so
  this mask is effectively dorsal/medial occipital + LOC-superior.

**BLOCKER — only TR5 could be analysed.** Of the 12
`group_RSA_instr_test_full_glmbase_01-TR{n}_cropped` folders, only TR5 finished
downloading. All 26 beta maps in each of the other 11 folders fail `gzip -t`,
truncated at exact 256 KB boundaries (6–10 MB of an expected ~28 MB) — an
interrupted transfer, 275 corrupt files. So this run is `--trs 5`: max-t is
corrected over voxels ONLY, not over voxels x TRs, and there is no timecourse.
Re-run across all 12 TRs once the transfer completes.

**Two maps are entirely zero** and were dropped from the tables (25 of 27
remain): `REWDSR-rewDSR_noInstr` and `SIMPLE-rewDSR_noInstr` — i.e. the whole
`rewDSR_noInstr` combo model in `condition_files/rsa_instruction_full.json`
wrote all-zero output. `splitDSR_noInstr` is fine, so it is specific to that
combo. Upstream bug, not a bug in this test.

**Multiple comparisons.** By request, no correction across the family:
each p_FWE is corrected within its own mask only. 25 maps x 3 masks = 75 tests.

**Result — the `_instr` (visual instruction similarity) models dominate
everywhere.** All p_FWE below are within-mask, one-sided positive; LOSO p at
k=100.

| mask | top map | peak t | MNI | p_FWE | LOSO t | LOSO p |
|------|---------|--------|-----|-------|--------|--------|
| mPFC   | curr_rew_instr | 4.79 | -4/38/8 | **.0037** | 3.26 | .0010 |
| mPFC   | CURR_REW_INSTR-splitDSR_vs_instr | 4.70 | -8/42/10 | **.0051** | 3.02 | .0026 |
| mPFC   | rewDSR_instr (= REWDSR_INSTR-rewDSR_vs_instr) | 4.46 | 4/20/34 | **.0107** | 2.26 | .0151 |
| mPFC   | two_next_rew_instr | 4.28 | 4/48/36 | **.0208** | 3.56 | .0007 |
| MTL    | rewDSR_instr | 8.30 | -32/-2/-34 | **<.0001** | 7.18 | <.0001 |
| MTL    | two_next_rew_instr | 6.26 | -30/-24/-22 | **.0002** | 4.87 | <.0001 |
| MTL    | curr_rew_instr | 6.11 | -32/-22/-24 | **.0004** | 4.60 | <.0001 |
| MTL    | three_next_rew_instr | 5.75 | -24/-8/-24 | **.0006** | 4.04 | .0001 |
| MTL    | next_rew_instr | 5.32 | -22/-10/-28 | **.0023** | 4.53 | <.0001 |
| visual | curr_rew_instr | 6.06 | -10/-90/36 | **.0003** | 3.70 | <.0001 |
| visual | rewDSR_instr | 6.16 | 24/-84/26 | **.0006** | 3.82 | <.0001 |
| visual | two_next_rew_instr | 5.45 | -44/-62/2 | **.0030** | 3.81 | .0002 |

**Not significant anywhere:** every execution-similarity model on its own —
`rewDSR` (mPFC p = .74, MTL .86, visual .49), `simple` (.96 / .99 / .99),
`curr_rew`, `next_rew`, `two_next_rew`, `three_next_rew`, and all four
`*-splitDSR_noInstr` regressors. In the `*_vs_instr` combos the execution
regressor is likewise null while its `_INSTR` partner carries the effect —
i.e. at TR5 the instruction-similarity regressor explains the variance and
leaves nothing for the execution regressor.

**Caveat on the effect sizes.** MTL t = 8.3 and visual t = 6.2 for
`rewDSR_instr` are far above anything the execution models produce, and the
`_instr` models are uniform within (task_i, task_j) 2x2 sub-blocks by
construction. A block-structured regressor of that kind can be picked up by any
residual block structure in the data RDM (e.g. run/session or scanner-drift
structure aligned with task identity), so these should be read as "the
instruction-block model fits" and not yet as evidence about representation.
Worth a control before interpreting.

**Also recorded, descriptive only:** `peak_t_neg` / `p_FWE_neg` — the negative
peak against the same (symmetric) sign-flip null, a second one-sided test not
corrected for testing both directions. 11 of 75 cells have p_neg < .05, the
strongest being `NEXT_REW-splitDSR_noInstr` in MTL (t = -5.02, 32/0/-20,
p = .0035). At an uncorrected .05 with 75 tests, ~3.75 are expected by chance.

**Method note:** the group brain mask here is the intersection of
`mask_all_32_subjects` over all included TRs, not TR0's mask alone as in
`svc_loso_test.load_ref` (the per-TR group masks differ by ~21 voxels; a voxel
entering the max-t search must be valid at every TR the search runs over).

## 2026-08-09 (later) — YER micro positions reconstructed from macro probes

**Script:** `scripts/cell_to_roi_july26.py`

YER's v2026 file ships no `microwires` / `sEEG-micro` rows (202 rows, all
`Type == sEEG`, no `m`-prefixed labels, `NSxSource`/`NSxIndex` empty), so
its 26 cells were the last Baylor cells still on the pre-2026 big-table
macro position.

**The 3.15 mm constant.** Across **all 119 bundles in all 19 files that do
carry `microwires` rows**, the `microwires` position sits exactly
**3.15 mm** from the `sEEG-micro` position (min 3.15, max 3.15), always
*beyond* the probe tip along the insertion axis. Baylor is applying a
nominal Behnke-Fried protrusion, not localising wires individually.

**Validation.** Rebuilding each bundle from its macro probe as
`contact01 − 3.15 mm × unit(contact02 − contact01)` reproduces Baylor's
own supplied `microwires` coordinate to **median 0.25 mm, max 1.07 mm**
over the 113 checkable bundles (YEN excluded; its MNI152 is the corrupt
column). 112/113 within 1 mm. So this is Baylor's own construction, not
an approximation.
*(Sort key must be `Label` — `ElectrodeID` is a mixed int/str column and
sorts lexicographically, which silently picks the wrong contact.)*

YER's own MNI152 column is sound (median 0.00 mm from Fischl(MNI305),
sensible whole-head extent), so the rebuild uses it directly. The
supplied `YER_electrodes.pptx` independently confirms the probe
inventory: 6 probes labelled "microwire" — RT2cHbEb, RT2bHaEa, RF2aCa,
LT2cHbEb, LT2bHaEa, LF2Ca — matching the CSV `ProbeName`s.

Implemented as `reconstruct_micro_from_macro()`, applied only to files
with no micro rows at all, tagged
`baylor_v2026_micro_reconstructed_from_macro` with
`coord_verified = False` (inferred, not supplied).

**Result:** 26 YER cells move by median 3.15 mm (max 10.19 mm). **2 cells
change ROI** (`mLT2bHaEa03`, cells 1 & 2: HC_mid → HC_anterior; that
bundle moves 10.19 mm because the big-table macro coord was on a
different contact). No Baylor cell now uses the big table.

Cumulative vs the aug-09-2026 reference: 112/984 cells moved,
**3/984 changed `alt_final_roi`** (HC_anterior 275→276, HC_mid 232→231;
all other ROIs unchanged).

### How much does the MNI152 provenance actually matter?
Measured on the 78 cell-carrying bundles in the 18 subjects where both a
supplied MNI152 and MNI305 exist: supplied-152 vs Fischl(305→152) differ
by **median 1.79 mm, 95th pct 5.0 mm, max 9.0 mm**, and the ROI verdict
flips for **5/78 bundles = 50/608 cells (8.2 %)**. So coordinates derived
via the plain Fischl affine — YEN (broken 152, 35 cells), YER (26), and
the five affine-only files YEL/YEP/YEQ/YEU/YFT (146) = **207/669 Baylor
cells** — carry roughly a 2 mm positional and ~8 % ROI-label uncertainty
relative to a proper MNI152 normalisation.

## 2026-08-09 — Baylor v2026 electrode tables for YEL / YEN / YFT added to the cell→ROI pipeline

**Script:** `scripts/cell_to_roi_july26.py`
**Output:** `data/ephys_humans/derivatives/neurons_with_ROI_labels.csv`
**Reference (previous run):** `data/ephys_humans/derivatives/old_electrode_tables/aug-09-2026/neurons_with_ROI_labels.csv`
**Per-cell diff:** `data/ephys_humans/derivatives/ROI_assignment/cells_step8_change_vs_previous_run.csv`

### Source data
New `-electrodes_v2026.csv` files placed at the top level of
`data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026/`: **YEL, YEN, YFT**
(byte-identical re-sends of YEU and YFI arrived as `...[37].csv` /
`...[94].csv` and are ignored by the loader's `-electrodes_v2026.csv`
suffix filter — no content change). **YER** still contains no
`microwires` / `sEEG-micro` rows at all, so its 26 cells remain on
`baylor_bigtable_pre2026_macro_position`.

### Code changes
- `load_baylor_v2026` now reads micro-bundle rows of **either** `Type ==
  "microwires"` **or** `Type == "sEEG-micro"` (new helper
  `_micro_bundle_rows`). `microwires` (the bundle itself, label
  `mLT2bHb01`) wins; `sEEG-micro` (contact 01 of the carrying macro
  probe, label `LT2bHb01`, ~3 mm shallower) only fills bundles with no
  `microwires` row. Both are keyed on the m-prefixed bundle name.
  *In the current data this fallback never fires* — every bundle that
  has an `sEEG-micro` row also has a `microwires` row, so all coordinates
  still come from `microwires`.
- Fixed a pre-existing ordering bug: `alt_final_roi` was consumed by the
  step-7 RSA-ready plot but only assigned at the very bottom of the
  script (`KeyError`). It is now assigned in step 7 where
  `analysis_rois` is computed.
- New **step 8**: change report vs `REFERENCE_TABLE` (an archived copy of
  this script's previous output), printing coordinate shifts per subject,
  cells shifted > 10 mm, `alt_final_roi` counts old vs new, and the
  transition matrix.

### Result
984 cells, unchanged row count. 86 cells moved coordinate; all of them in
YEL (17), YEN (35), YFT (34).

| provenance | before | after |
|---|---|---|
| `baylor_v2026_bundle_micro` | 557 | 608 |
| `baylor_v2026_bundle_305to152_unreliable_file` | 0 | 35 |
| `baylor_bigtable_pre2026_macro_position` | 112 | 26 |

YEN's `MNI152_*` columns are internally inconsistent with its `MNI305_*`
columns (mean 55.6 mm, max 71.2 mm disagreement under the Fischl
305→152 transform), so the existing reliability gate correctly rejects
them and uses `MNI305 → 152` instead (`..._305to152_unreliable_file`).

**Coordinate sanity:** 85 / 86 moved cells shifted **3.13–3.15 mm** —
exactly the known macro-last-contact → micro-bundle offset, i.e. a small
local correction, not a relocation. The single exception is
`BY2-YEN`, electrode `mLT2bHb07`, cell idx 2, which moved 60.6 mm:
its big-table coordinate was `(32.3, -19.8, -17.9)` — the **right**-
hemisphere `mRT2bHb` coordinate — despite an `mL...` (left) electrode
label. The label-driven v2026 lookup places it at `(-28.1, -24.2, -14.9)`
together with its seven `mLT2bHb` siblings. This is a big-table
data-entry error being corrected, not a localisation change.

**ROI changes (`alt_final_roi`, the column read by
`mc.analyse.roi_relabel.relabel_per_cell`):** exactly **1 of 984** cells
changed label — the `mLT2bHb07` cell above, `HC_anterior → HC_mid`.

| ROI | before | after | Δ |
|---|---|---|---|
| HC_anterior | 275 | 274 | −1 |
| HC_mid | 232 | 233 | +1 |
| mOFC | 163 | 163 | 0 |
| mPFC | 155 | 155 | 0 |
| PCC | 61 | 61 | 0 |
| EC | 38 | 38 | 0 |
| (NaN / excluded) | 60 | 60 | 0 |

Downstream RSA results are therefore essentially unaffected; the value of
the update is 86 cells now sitting on verified micro-bundle coordinates
rather than inferred macro-contact positions.

---

## 2026-08-13 — Cell ↔ fMRI future-lag gradient: extensive exploration (mostly null)

**Question:** do human mPFC single units recapitulate the fMRI DSR "preferred
future angle" gradient — i.e. if we average cells, does anatomical position
predict their preferred spatial-tuning lag the way the fMRI angle map does?

**Scripts (all read `per_cell_ALL_ROIs.csv`, mPFC, 155 cells / 32 subjects):**
`cell_gradient_master_table.py` (master per-cell + group tables),
`gradient_brain_cells_by_lag.py` (MNE medial surfaces, cells coloured by lag),
`cell_gradient_principal_curve.py` (bent-axis sliding window + shift test),
`cell_gradient_split_table.py`, `cell_gradient_split_permgated.py`,
`cell_gradient_full_factorial.py` (320-row robustness grid).
Outputs under `data/ephys_humans/derivatives/group/cell_gradient_master/2026-08-13_09-45-28/`.

**Methods settled on:** pool the 12-lag profiles within a cell group, then read
the preferred lag off the pooled profile (argmax); per-cell argmax and the
continuous first-harmonic angle are both too noisy (harmonic vector length
≈ 0.07; per-cell harmonic angle ~ uniform). fMRI angle sampled at each cell
via symmetrised + 3 mm cos/sin smoothing + 6 mm sphere (quarters map).
Anatomical axis = PC1 of the gradient-mask voxels (folded x) = essentially
dorsoventral (loads [-0.04, -0.41, 0.91], r = 0.98 with MNI z).

**NULL / dead ends (do not re-run):**
- **Continuous gradient does not exist.** Spearman(future-score, arc-length on a
  bent principal curve) = +0.058, circular-shift p = 0.25, subject-bootstrap
  95% CI [-0.10, +0.19]. In the full factorial, 31/32 correlation configs are
  n.s. (r ≈ 0.0–0.17); ctrl-mode correlations ≈ 0 or negative.
- **Continuous first-harmonic angle** per cell is unusable (near-uniform;
  group bootstrap CIs span the whole circle).
- **Controlled tuning (`_ctrl` columns) shows nothing** reproducible — binned
  pooled lags are chaotic across weighting/gating (confirms prior expectation).
- **Subject-first vs cell averaging** does not stabilise; it only *diverges from*
  cell-weighting where the pooled profile is flat (peak r ≲ 0.05), i.e.
  divergence is a noise flag, not a fixable choice.
- **Perm-gating** (only cells significant at a lag contribute to that lag) does
  not sharpen results; it thins data (median 2–7 cells/lag) and only confirms
  the already-robust groups.
- A real **240° "backward" signal** sits mid-axis (pc1-Q2 / z-middle,
  peak r 0.085–0.10) — genuine, and it blocks any monotone ventral→dorsal ramp.

**The one robust positive:** under **noctrl**, the **future-end bin of the
gradient axis** pools to **60°**, invariant to cell-vs-subject weighting AND to
perm-gating: `all/pc1/half:end`, `all/pc1/quartile:Q3`, and `all/z/quartile:Q4`
all give 60/60/60/60, matching the local fMRI angle (~63°, err 3–9°; pooled
peak r ≈ 0.08–0.16). Interpretation is **local, not a gradient**: "cells in the
deep-future (dorsal/high-z) end of the DSR gradient prefer 60°, matching fMRI
there," NOT "a cell gradient mirrors the fMRI gradient."

**One nominally-significant correlation (EXPLORATORY — treat with caution):**
`noctrl / in_mask / z / perm-gated (≥1 sig lag) / cell-weighted`:
Spearman(future-score, MNI z) = **0.238, shift-p = 0.032, n = 54**. This is
1 hit in 32 tests and does **not survive subject-weighting** (r = 0.028,
p = 0.45), so it is not corrected-significant. Splitting those 54 cells at
median z: low-z (n=27) argmax 240°, future-score +0.00; high-z (n=27) argmax
**60°**, peak r 0.159, future-score +0.087 — the same "high-z → 60°" story.

## 2026-08-26 — SWR pipeline rewrite to Chen/Staresina standard: session audit (Milestone 0)

**Scripts:** `mc/analyse/swr_io.py` (new), `scripts/swr_audit_sessions.py` (new).
Outputs under `data/ephys_humans/derivatives/group/swr/`
(`session_manifest.csv`, `session_blocks.csv`, `session_manifest.json`, `settings.json`).

Motivation: comparison of the existing ripple pipeline against Chen, Staresina et al.
2025 (J Neurosci 45:e1502252025) found seven blocking defects. The most consequential
is that `identify_HPC_ripples.py:122` estimates the detection threshold *within each
cropped snippet*, which partially normalises ripple rate to be constant per snippet and
so suppresses exactly the between-window rate difference the planning hypothesis is
about. Also absent entirely: artifact/IED rejection, a line-noise notch, spectral
validation of candidate events. Full rationale in the plan file.

**Two loaders are currently broken and cannot have run.** `all_trial_times_{XX}.csv`
has **14 columns**, but `scripts/identify_HPC_ripples.py:57` and
`scripts/preprocess_LFP.py:128` assign 13 names, so `df.columns = column_names` raises
`ValueError: Length mismatch` on every session. The correct 14-name contract already
existed at `scripts/behaviour_summary.py:47` and `mc/analyse/helpers_human_cells.py:379`
— the 14th column is `correct`. `swr_io.BEH_COLS` now mirrors it.

**`correct` is per-repeat accuracy, NOT a "plan known" state.** Verified on s05: grids 2
and 9 contain errors *after* the first correct solve. The planning boundary is therefore
derived as a cumulative max within grid (`swr_io.load_behaviour` adds `plan_known`),
which is cleaner than the `found_first_D` heuristic at
`mc/analyse/ripple_helpers.py:69-72`, but the raw column must not be used as a
per-repeat planning state.

**The behavioural clock is continuous across recording blocks; the LFP files are not.**
**25 of 60 sessions are multi-block.** Block *k+1* always continues the block-*k* clock,
separated by a real recording gap: measured range **+7.3 s (s33) to +2910.1 s (s21, a
48-minute break)**; s18 and s27 have three blocks (+229.6 s, +148.8 s). This falsifies
the assumption at `scripts/preprocess_LFP.py:213`, which maps behaviour into block 2 by
subtracting the *file duration* — valid only if recording never stopped. Block offsets
must be estimated per session against three independent references and hard-validated
(every behavioural event inside `[0, duration_k]` with >=5 s margin, else the block
emits no ripples). This is now the highest-risk item in the build; a 20 s error
misassigns every block-2 ripple systematically, which would read as a null rather than
as noise.

**Subject clustering needs a normalised key.** 63 sessions map to 43 distinct
`Subject Label` values but fewer real subjects: s29 is `'UT1-202314'` and s30 is
`'UT202314'` — the same patient in two of the four Utah label formats. 16 labels span
more than one session (`BY2-YEK` = s07/08/09; `BY2-YEX` = s43/44/49). Clustering
robust SEs on the raw label is anticonservative. `swr_io.normalise_subject_key` collapses
them and the audit prints the full map for manual sign-off.

**Audit result (run locally, 60 sessions):** 24 `ok`, 4 `needs_review`, 32
`no_raw_files`. Sites: baylor 36, utah 18, ucla 6. The 32 are simply not on the laptop
— **the audit must be re-run on ceph**, where those sessions' config defects will
surface. The 4 genuine structural defects to resolve by hand: **s03** (UCLA, 2 `.ncs`
recording blocks not represented in the YAML at all), **s18** and **s28** (duplicate
block names in `blocks`), **s32** (yaml=2 blocks, behaviour=1, files=2).

**Config YAML is demoted to a hint.** It disagrees with behaviour and with disk for a
large minority of sessions (`segment: null` for 13; no `blocks` key for s57/58/59;
duplicate block names; block counts that do not match). `session_manifest.csv` is now
the authority for everything downstream.

**NULL / dead ends (do not re-run):**
- Do not use `scipy.signal.resample` on continuous traces — it is FFT-based and wraps
  the end of the recording into the beginning. Silent, and visible only at the two ends.
  Use `resample_poly`. This is also the source of the per-snippet edge ringing in the
  old pipeline.
- Do not downsample to 500 Hz (`preprocess_LFP.py:30-32`). It makes Chen's >250 Hz RMS
  artifact criterion and the 120-200 Hz spectral rejection impossible. Keep 1000 Hz.
- Do not store full stepwise TFR power. The existing `ripple_power_dict_s05` is **5 GB**
  and is not needed by anything.

## 2026-08-26 (later) — Block structure resolved from the raw data, not the config

**Script:** `scripts/swr_diagnose_blocks.py` (new). Reads every neo segment's duration
and channel count out of the raw files and matches them against behavioural block spans.
Written because excluding sessions on a config warning discards data that the recordings
themselves can disambiguate.

**s18 and s28 are fine — the "duplicate block names" warning was a false alarm.** The
duplicates are *filename labels*, not duplicate recordings. Matching by duration is
unambiguous:

| session | behavioural block span | matched recording | slack |
|---|---|---|---|
| s18 | 363.3 s / 4006.1 s / 3432.4 s | EMU-058 seg1 / EMU-059 seg1 / EMU-060 seg0 | +197.0 / +33.9 / +226.4 s |
| s28 | 1819.7 s / 654.6 s / 927.4 s | EMU-045 seg1 / EMU-047 seg1 / EMU-048 seg1 | +41.6 / +58.7 / +36.1 s |

Ordering by **EMU number** (the acquisition counter) is the reliable chronological key;
`blk-NN` labels are not. Note each Baylor file carries a ~2.3 s stub segment plus the
real recording; picking the longest segment per file independently reproduces the YAML
`segment` field for both sessions, so that field is trustworthy where it is populated.

**s32 has a real 444.8 s hole in the middle of a single behavioural block.** Behaviour is
one continuous block (101.8 -> 3883.0 s, 304 repeats). The recording is two files with
**non-zero `t_start` on a common amplifier clock**: 12042.7 s + 1977.5 s, then 14465.0 s
+ 1516.1 s. Recorded wall span 3938.4 s brackets the behavioural span 3781.2 s, but
~445 s of task has no LFP. Usable if the repeats falling in the hole are dropped.

**`t_start` is the amplifier wall clock and gives exact inter-block offsets where it is
populated** (s32: yes; s18/s28: zero). This is the "header wall-clock" offset estimator,
and it is the principled replacement for the file-duration subtraction at
`scripts/preprocess_LFP.py:213`. It is also what the `t_start > 10000` hack at
`preprocess_LFP.py:76-83` was groping at.

**s03 (UCLA) is a SINGLE-block session and is usable.** 164 of its 308 `.ncs` files are
**header-only 16384-byte stubs** with zero data records ("TimeClosed File was not closed
properly") — an aborted first recording. All real data is in the `_0001.ncs` files, which
include both macros (`LMH*`, `RMH*`, `LA*`, `LAC*`, `ROF*`, `RPH*`) and micros (`GA*-`).
Counting the stubs as a second block is what made the YAML look inconsistent.
`swr_io.discover_raw_files` now skips any `.ncs` of exactly the header size.

**UCLA sampling rate in the YAML is wrong.** `config_human_ABCD_iEEG.yaml` declares
`sampling_rate: 1000` for all six UCLA sessions. The s03 `.ncs` headers report
**macros at 2000 Hz** (micros at 32000 Hz). Read the rate from the file header, never
from the config.

**neo cannot read these `.ncs` files.** `NeuralynxRawIO.parse_header()` raises
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` at
`neuralynxrawio.py:505` because `global_t_start`/`global_t_stop` are None. A direct
header/record reader (16 KB header, 1044-byte records of
`uint64 ts, uint32 ch, uint32 fs, uint32 n_valid, int16[512]`) works and additionally
exposes the Unix-epoch record timestamps needed for block alignment. UCLA support must
not depend on neo.

**Audit after these fixes: 26 `ok`, 2 `needs_review` (s03 informational, s32 the 445 s
hole), 32 `no_raw_files` (not on the laptop; re-run on ceph).** s18 and s28 moved to `ok`.

**NULL / dead ends (do not re-run):**
- Do not order Baylor blocks by the `blk-NN` label or by the YAML `blocks` list. Both
  contain duplicates. Order by EMU number.
- Do not use `neo.io.NeuralynxIO` / `NeuralynxRawIO` on the UCLA data in this env; it
  raises before returning anything. Parse the `.ncs` header directly.
- Do not trust `sampling_rate` in the YAML for UCLA (says 1000, actually 2000).

## 2026-08-26 (later still) — Anatomy loaders extracted to shared modules (Milestone 1)

**New:** `mc/analyse/anatomy_sources.py` (coordinate sources, all three sites),
`mc/analyse/anatomy_atlas.py` (MaxProbAtlas + the ROI rule ladder).
**Changed:** `scripts/cell_to_roi_july26.py` now imports them; 2514 -> 1880 lines.
**Unchanged:** every output of the cell pipeline.

Rationale: the LFP/ripple pipeline needs to pick hippocampal and mPFC *macro*
contacts by anatomy rather than by string matching (the old
`'H' in channel and 'T' in channel`, and `'1' in label` which also matches
HIP10/11/14). Rather than duplicating the rules, both pipelines now call one
implementation — so "the same anatomical criteria defined HC for cells and for LFP
contacts" is literally true, and a future change to `assign_atlas_roi` propagates to
both instead of silently diverging.

The move was verbatim: `sed`-extracted line ranges 221-617 (loaders) and 982-1029 +
1040-1245 (atlas layer), not retyped. Two structural edits only:
- `discover_utah_mats()` took two module globals of the script; it now takes them as
  keyword arguments defaulted to the same literal paths, so the bare call still works.
- The four atlas objects were built by `nldatasets.fetch_*` at import time; they are
  now built by `get_atlases()`, which populates the same module-level names.
  `assign_atlas_roi` still reads the module globals `juelich` and `HC_ANT_MID_Y`, so
  its body is byte-identical.

**Regression gate — PASSED.** Procedure: (1) confirm the *unmodified* script is
deterministic by re-running it and checking it reproduces its own output
(md5 `e0e758a303831cfc614a2490dcaf6aac`) — it does, so a post-refactor diff cannot be
blamed on pre-existing nondeterminism; (2) refactor; (3) re-run.

`derivatives/neurons_with_ROI_labels.csv` is **byte-identical** after the refactor, as
are all nine `ROI_assignment/cells_step*.csv` intermediates. Baseline copies kept at
`/tmp/swr_refactor_baseline/` for this session.

Note the gate must be against a *fresh run of the current script*, not against
`old_electrode_tables/aug-09-2026/` — that archive legitimately differs
(md5 `9ebfceda...`, 205394 bytes vs 204383) because new electrode files arrived since.

**NULL / dead ends (do not re-run):**
- A static AST check (names used by the script that are defined in the new modules but
  not imported) caught `_bundle_key`, which a syntax check cannot: `py_compile` passes
  on an undefined global. Use the AST check, not `py_compile`, when moving definitions
  out of a flat script.
- The new modules are deliberately NOT added to `mc/analyse/__init__.py`, which imports
  eagerly — adding nilearn/neo-heavy modules there would make every `import mc` in the
  repo pay for them. Import explicitly.

## 2026-08-26 (Milestone 2) — Macro-contact anatomy: hippocampal LFP contacts selected by anatomy, not string matching

**New:** `mc/analyse/contact_anatomy.py`, `scripts/swr_build_contacts.py`.
**Outputs:** `derivatives/s{XX}/LFP/macro_contacts_{XX}.csv` + `bipolar_pairs_{XX}.csv`
per session; `derivatives/group/swr/macro_contacts_all.csv`, `contact_qc.csv`,
`settings.json`.

Replaces contact selection by string matching — the old `'H' in channel and 'T' in
channel` (matches any label containing both letters) and `'1' in label` (also matches
HIP10/11/14) — and the hard-coded contact01−contact04 bipolar pair spanning ~15 mm.

**Result: 160 hippocampal contacts / 164 hippocampal bipolar pairs across 27 sessions**
(baylor 94, utah 60, ucla 6). The remaining 33 sessions have no raw data on the laptop;
re-run on ceph.

### Utah `ElecMapRaw` indexing — a trap that would not have raised
`ElecMapRaw` is an (n, 3) object array `[label, amplifier_channel, other_channel]`.
Coordinates must be read by the **direct row index** into `ElecXYZMNIRaw` /
`ElecAtlasRaw` / `ElecTypeRaw` — **not** by indexing with the channel number.
Verified on s02: row 41 is `bRAHIP2`, and `ElecAtlasRaw[41]` = "Right Hippocampus",
whereas indexing by column 1 gives "Right fusiform gyrus" — a different electrode
entirely, silently. Column 1 is the amplifier channel and matches
`utah_elec_labels_{XX}.csv` for all 114 rows where it is finite (the other 30 are
unlocalised and are dropped with a logged reason).

Also: `m*` = microwire, `b*` = the Behnke-Fried macro on the same shaft, bare = plain
sEEG depth. Macros use `ElecXYZMNIRaw`, not `ElecXYZMNIProj` — gray-matter projection
is right for a spiking micro tip and wrong for a 2 mm macro ring that legitimately
straddles the GM/WM boundary. Both are stored.

### ROI definition: native-space primary, shared atlas ladder as cross-check
Measured on s05/s02/s03, `anatomy_atlas.assign_atlas_roi` has **100% recall but 22%
precision** against the subjects' own segmentations: it calls 49 contacts hippocampal
where the native labels say 11, the excess being amygdala (`RAMG1-4`), lingual gyrus
(`RPHIP1-2`), VentralDC (`RT1cCM04`) and parahippocampal cortex. Expected — that ladder
was tuned for microwire *tips*, where ±2–3 mm neighbourhood rescue is desirable, and is
applied here to MNI152, where MTL registration error is large.

So a macro contact's ROI comes from the **subject's own segmentation**, matching Chen
et al., who identified hippocampal contacts "via visual inspection of postoperative
T1-weighted anatomical MRI scans". The shared ladder is still computed and stored as a
cross-check; because it is a strict superset, only the reverse disagreement is
meaningful. The cell pipeline is unaffected — `assign_atlas_roi` is untouched.

**Use the 3 mm-neighbourhood parcellation, not the single-voxel one.** For Baylor,
`ROI_DK2005_3mm` / `Matter_3mm`, not `Area_fs_vox` / `Matter_fs_vox`. A macro contact is
a ~2 mm ring recording from a volume, so the label must be sampled at that scale. On
YEJ the voxel column calls 1 contact hippocampal and the 3 mm column calls 4 — e.g.
`RT2bHaEa02` is `Right-Hippocampus` at 3 mm and `Right-Cerebral-White-Matter` at the
voxel. Voxel columns retained as `native_region_vox` / `matter_vox`.

**Pair rule:** a bipolar pair is hippocampal if **at least one** of its two contacts is
natively hippocampal — the standard montage (Chen take the most medial hippocampal
contact and its immediate neighbour, frequently white matter). Requiring both would
have given s05 zero pairs.

### Gate — PASSED (5/5)
1. **Probe names plausible.** baylor `RT2bHaEa02`, `LT2HbE02` (Ha/Hb); utah `LAHC1`,
   `bRAHC1`, `LPHC1`; ucla `LMH-1`, `LPH-1`, `RMH-1`.
2. **Atlas concordance 158/160.** The 2 exceptions are `LT2aA01` in s31 and s35 (same
   subject YFF): native says Left-Hippocampus, atlas says EC — an amygdala-probe
   contact 01 sitting on the HC/EC border. Borderline, not a defect.
3. **UCLA independent ground truth: 6/6.** Every UCLA HPC contact carries an ASHS
   hippocampal subfield (DG, CA1, SUB, CA2), and **0 contacts with an ASHS subfield
   were missed**. ASHS is a completely independent segmentation — this is the strongest
   validation available.
4. **Anterior/mid split clean.** HC_anterior y ∈ [−20.8, −3.8]; HC_mid y ∈ [−34.4,
   −21.4]. No overlap at HC_ANT_MID_Y = −21.0.
5. **Bipolar spans sane.** Median 4.68 mm (range 3.08–7.75), i.e. genuine adjacent sEEG
   spacing, versus ~15 mm for the old 01−04 pairing.

**Open point for Milestone 3/4:** adjacent pairs share contacts — pair (2,3) and pair
(3,4) both contain contact 3 — so the 164 pairs are not independent (median 5 per
session, max 11; Chen used ~2 per participant, one per probe). Decide before detection
whether to keep all pairs, or one anchor pair per probe.

**NULL / dead ends (do not re-run):**
- Do not index Utah coordinates via `ElecMapRaw` column 1. It returns another
  electrode's position and raises nothing.
- Do not use `Area_fs_vox` / `Matter_fs_vox` for macro contacts — single-voxel labels
  understate hippocampal coverage ~4x.
- Do not merge anatomy to channels on a key that can be NaN: pandas matches NaN to NaN,
  and s02 fanned 132 channels out to 248 rows before the null-key guard.
- Do not use the atlas ladder alone to select macro contacts (22% precision).

## 2026-08-26 (CORRECTION) — Block alignment is deterministic; the earlier "highest risk" entry was wrong

**This corrects the 2026-08-26 entry above**, which claimed the behavioural clock
includes recording gaps the LFP files do not, and that
`scripts/preprocess_LFP.py:213` (mapping block *k* by subtracting cumulative file
duration) was therefore "badly wrong". **That claim was mistaken. The old approach is
correct.** Verified rather than assumed, via `/tmp/blockfit.py`.

**The behavioural clock is cumulative file duration.** For block *k*,
`offset_k = sum(durations of files 0..k-1)`, and behavioural times map into file *k* as
`t - offset_k`. Tested on all **14 multi-block sessions with local raw data**: every
behavioural event lands inside its file, with head/tail margins of seconds.

s18 worked out exactly: file durations 560.3 / 4040.0 / 3658.8 s give offsets 0 /
560.3 / 4600.3; behavioural block 2 (580.8–4586.9 s) maps to in-file 20.5–4026.6 s
against a 4040.0 s file. The apparent 180.1 s "behavioural gap" I reported earlier is
just the 159.6 s tail of file 1 plus the 20.5 s head of file 2 — **not** a recording gap.

**Wall-clock timestamps are irrelevant, and would have been actively misleading.** The
NSx 2.2/2.3 basic header does carry a `TimeOrigin` SYSTEMTIME at byte offset 294 (neo
reports `rec_datetime = None`, but it parses directly). For s18 those wall clocks are
1943.4 s and 12459.2 s apart, versus file durations of 560.3 and 4040.0 s — i.e. real
elapsed gaps of 20–200 minutes between recordings. Had I anchored blocks to wall clock,
every block-2 ripple would have been misplaced by tens of minutes. The behaviour was
timestamped against the *concatenated* recording, not against wall time.

Two NSx variants exist: `NEURALCD` (2.2/2.3, has TimeOrigin) and `BRSMPGRP` (s32, s33 —
different layout, no TimeOrigin at that offset; neo reports a non-zero `t_start`
instead). Neither is needed for alignment.

**Three sessions overrun by 1–2 s, each by exactly one repeat.** s09 block 2 (−2.0 s),
s10 block 2 (−1.8 s), s33 block 1 (−0.9 s) — 1 repeat of 275, 274 and 296 respectively.
The amplifier was stopped a second or two before the last trial completed. Handle by
dropping the overrunning repeat and logging it; this is not a misalignment (a real one
would be off by tens or hundreds of seconds, not by one trial).

**Consequence for the plan:** block offsets need no three-way estimation and no
wall-clock anchoring. The rule is `cumsum(file durations)`, with a hard validation that
every behavioural event lands inside its file (margin ≥ 0, allowing a single trailing
repeat to be dropped). This removes what I had flagged as the highest-risk item.

**NULL / dead ends (do not re-run):**
- Do not align blocks by NSx `TimeOrigin` / wall clock. It is present and parseable but
  describes real elapsed time between recordings, which the behavioural clock excludes.
  Using it would misplace block 2+ by tens of minutes.
- Do not treat the behavioural inter-block "gap" as a recording gap. It is the tail of
  one file plus the head of the next.

## 2026-08-26 (Milestone 2, CORRECTION) — One bipolar pair per probe, not every adjacent pair

**Corrects the pair counts in the Milestone 2 entry above (164 pairs).** The final
number is **76 pairs across 27 sessions / 18 subjects**.

`build_bipolar_pairs` was generating *every* adjacent pair containing a hippocampal
contact. That over-generated (median 5 per session, max 11) and made the derivations
non-independent: contact 3 appeared in both pair (2,3) and pair (3,4), which the GLM
cannot account for.

Corrected to follow Chen et al. exactly — *"bipolar referencing was performed using the
most medial hippocampal contact and its immediate neighbour (i.e. the second-most
medial contact) **on each hippocampal probe**"* — i.e. **one derivation per probe**, so
no contact is ever reused. Chen report 34 contacts across 17 patients (~2 each); we get
76 across 27 sessions (~2.8 each).

**Anchor = most medial hippocampal contact on the probe, defined geometrically as
min |MNI x|**, not by contact number. Numbering conventions differ between sites
(Baylor contact 01 is deepest; Utah/UCLA differ), so a geometric definition of "medial"
is convention-free.

**Reference rule = immediate neighbour** (`scheme='neighbour'`, the default). Measured
alternatives on the full contact table:

| scheme | pairs | sessions | median span | ref in MTL grey | probes dropped |
|---|---|---|---|---|---|
| `neighbour` (Chen) | 76 | 27 | 5.28 mm | 49/76 | 0 |
| `white_matter` | 42 | 23 | 10.15 mm | 0/42 | 34 (no WM contact on probe) |

The white-matter montage gives a cleaner subtraction (no ripple signal in the
reference) but doubles the inter-contact distance, which enlarges the lead field,
degrades spatial specificity, and would *increase* volume-conduction confounds for the
H2 HC–mPFC analysis — the opposite of what that analysis needs. It also drops 34 probes
and 4 sessions. `scheme='white_matter'` remains available for a sensitivity analysis.

The 49/76 references that are themselves MTL grey are expected and are what Chen's own
montage produces (the second-most medial contact on a hippocampal probe is usually
still hippocampus). Ripples are spatially local at the millimetre scale, so adjacent
contacts 5 mm apart do not see identical events and the bipolar preserves them. If
common-mode cancellation turns out to matter, it will show up at the Milestone 4
checkpoint as reduced ripple amplitude or rate.

**NULL / dead ends (do not re-run):**
- Do not generate all adjacent pairs. It inflates n several-fold with non-independent
  derivations sharing contacts.
- Do not pick the anchor by contact number; numbering conventions differ per site.

## 2026-08-27 — Coordinate provenance audit of `cell_to_roi_july26.py`

Full trace of where every one of the 984 cell coordinates in
`neurons_with_ROI_labels.csv` comes from. Written up in
`docs/coordinate_provenance_audit.md`.

**Clean (906 / 984, 92 %):** 608 Baylor from the v2026 `microwires` row as shipped;
140 UCLA big-table coords independently corroborated ≤ 0.5 mm against the v2026
xlsx; 97 Utah reconstructed from the patient's own `Electrodes.mat`; 35 Baylor
MNI305 re-transformed (file failed its own 152-vs-305 gate); 26 Baylor inferred
from the macro probe with Baylor's own 3.15 mm protrusion constant.

**Not clean (78 cells, 7.9 %):** `utah_bigtable_recon_disagrees_gt3mm`. When the
reconstruction from the patient's own `.mat` disagrees with the hand-entered big
table by > 3 mm, the code keeps the **big table** and discards the reconstruction.
Median disagreement **37.9 mm**, max 62.3 mm; 73/78 exceed 10 mm. These currently
contribute mOFC 48, mPFC 18, HC_anterior 9, EC 3.

**Placeholder coordinate found:** 17 cells across UT1-202418 / UT1-202422b /
UT1-202503 all sit at the single point (4.55, 29.50, -20.63), which the atlas
calls mOFC. Those subjects (s54, s53, s55, plus s52 = UT202421) have **no
`Electrodes.mat` anywhere** — the four genuinely missing Utah files. s30 and s42
also lack one but are the same patients as s29 and s41.

**Root cause:** `discover_utah_mats()` coord-matches 3–16 big-table cells against
every folder's electrode pool with no uniqueness constraint, so s47's file was
assigned to six different patients. Measured: **8 of 12 Utah subjects match their
own `s{NN}` folder at 100 %** — the folder numbering is reliable and the
coord-matching was unnecessary. It is also circular: it validates the big table
against files using the big table's own (hand-entered) coordinates as the key.

**NOT changed:** the cell pipeline's behaviour is untouched — fixing items 1–3 in
the audit would change published cell ROIs and is the user's call. Only the
docstring was corrected (it claimed the reconstruction is preferred above 0.5 mm;
the code does the opposite at 3 mm) and an invented rationale was removed from
`discover_utah_mats()`.

**Already fixed in the SWR pipeline:** `mc.analyse.contact_anatomy.resolve_utah_mat`
resolves by folder (own → same patient → exclude with a stated reason), so no
session inherits another patient's electrodes.

### 2026-08-27 (addendum) — is the Utah .mat reconstruction trustworthy?

Validated `build_micro_map` against two independent signals: the microwire label
(`LabelMap`) vs the coordinate (`ElecXYZMNIProj/Raw` via `MicroElec`). Hemisphere
agreement **245/264 = 92.8 %**, region agreement **185/256 = 72.3 %**, against
chance of ~50 % and ~15 %. **The reconstruction method is sound** — but not
uniformly, and three subjects fail individually.

- **s23 (UT1_sj202309) is broken**: hemisphere 8/24, region 0/24. `MicroElec` empty,
  uses the `MicroElecRaw` fallback, labels misaligned to coords (`mLHIP1-8` land at
  x ≈ +8, which is not hippocampus in either hemisphere). Exclude, don't guess.
- **s47 and s39**: hemisphere 24/24 but region 8/24 — needs a look.
- **Resolving by folder fixes UT1_sj202308**: 1/16 → 16/16 hemisphere correct. The
  published laterality was right; the coord-matched file assignment was wrong.
- **MATLAB v7.3 (HDF5) files are silently unreadable**: `_load_mat` returns
  `LabelMap` as bare `None` because string cells are HDF5 object references that are
  never dereferenced. Affects s48, s52, s54, s55 — all four yield ZERO microwires
  and always fall back to the big table. Three of them are the placeholder-coord
  subjects.
- **Files after re-download**: s52, s54, s55 are present but under `Registered/` /
  `Registered-selected/`, which the loader never searches (it only looks in
  `electrodes/`). Only **s53 (UT202422b)** is genuinely absent. s30/s42 lack a file
  but are the same patients as s29/s41.

**UCLA is clean.** `load_ucla_v2026` reads the right sheet (`Sheet1`, the second
sheet, carrying `MNI_x/y/z` + `isMicro`). It does not filter on `isMicro`, but all
**140/140** UCLA cells match a microwire row at ≤ 0.5 mm and none match a macro —
so the missing filter is latent, not an actual error.

Full detail incl. per-subject tables: `docs/coordinate_provenance_audit.md`.
No code paths changed in this session; docstrings only.

### 2026-08-27 (addendum 2) — Utah coordinates: 168/175 now read directly

**Fixed the v7.3 read bug.** `_load_mat` returned `LabelMap` as a bare `None` for
MATLAB v7.3 files because cell-array strings are HDF5 object references that were
never dereferenced. s48, s52, s54, s55 therefore yielded ZERO microwires and always
fell back to the hand-entered big table. Now dereferenced.

**Files declare their own identity.** Every Utah `.mat` carries `Fname` (the original
acquisition path, e.g. `D:\Data\UIC202311\...`). Added `mat_patient_id()` to read it,
which removes the need for coord-matching entirely. Two mismatches found:
`s47` holds patient **202311**'s data (not 202302 — it is the v7 export of the same
165 electrodes as s48), and `s53`'s newly-downloaded file is a **duplicate of s52**
(202421, not 202422b). So UT1-202302 and UT1-202422b have no electrode file.
NOTE these files contain patient *names* in `PatientIDStr` — use the numeric ID.

**Removed the ordering assumption.** `build_micro_map` inferred the label↔coordinate
pairing by sorting microwires by amplifier channel against `MicroElec` (validated on
only s02/s06; failed on s23). Every file also has `MicroElecRaw` + `ElecMapRaw` +
`ElecXYZMNIRaw`, which indexed by the same row give label and coordinate together.
Added `build_micro_label_map()`. Validated across all 15 files: **352/360** microwires
have `sign(MNI_x)` matching the `mL`/`mR` in their own label (the 8 exceptions are
OFC within 2.5 mm of midline). **s23 goes from 8/24 to 24/24.**

**Census across all 984 cells:** 776 read directly from a site file; 140 (UCLA)
verified identical to the site file at ≤0.5 mm; 61 (Baylor) derived by a documented
transform; **7 guessed because no source file exists** (UT1-202302 ×3,
UT1-202422b ×4) = 0.7 %. Utah went from 97/175 to **168/175** readable.

**Not yet wired into `cell_to_roi_july26.py`** — that changes published cell ROIs.

### 2026-08-27 (addendum 3) — full coordinate rebuild from site files only

Re-derived every cell coordinate from the recording site's own electrode file.
Output: `derivatives/ROI_assignment/coordinate_rebuild_2026-08-27/`.

**Baylor and UCLA do not move (0.00 mm)** — the published table already used file
coordinates for both. Every change is Utah: UT1-202503 (52.8 mm), UT1-202418
(42.8), UT1-202421 (42.7), UT1_sj202308 (17.5), UT202314 (2.7), UT202413 (2.7),
UT1-202311 (1.3); all other Utah subjects 0.00.

**UT1_sj202309 (s23) does not move.** Its published coordinate already equalled the
direct read, so the hand-entered table was right for s23 and the old ordering-based
reconstruction was what was wrong — the old code kept the big table there for the
wrong reason.

**26 of 977 cells (2.7 %) change atlas ROI, all Utah:** mOFC→HC_anterior 12,
mOFC→mPFC 6, EC→HC_anterior 3, mPFC→mOFC 3, mOFC→HC_mid 2.

**New alt_final_roi counts (tier C, only the 7 no-file cells excluded):**
EC 38→35, HC_anterior 276→291, HC_mid 231→233, PCC 61→61, **mOFC 163→139**,
mPFC 155→158. 924→917 cells kept.

Only two non-as-shipped sources remain, both flagged: BY2-YEN (35 cells, the file's
own MNI305 through the Fischl affine) and BY2-YER (26 cells, macro + 3.15 mm, where
3.15 mm is Baylor's own constant, identical across all 119 ground-truth bundles and
reproducing them to 0.25 mm median). Dropping YER costs 16 mPFC cells (158→142).

**NULL / dead end (do not re-run):** coord-matching electrode files against the
big table. It is circular — the big table's coordinates are the thing in question —
and with 3-16 cells per subject it has no discriminative power. Use `Fname`.

## 2026-08-27 — Single-unit QC audit

`abcd_passed.mat` verified as `abcd_data_08-Sep-2025.mat` filtered to the QC-passing
units: 63 sessions, 1042 -> 984 (58 removed). Every retained spike train and every
`regionLabel` is byte-identical to the source — neural data and labels both untouched.

**Exclusions:** 36 for < 300 spikes (median 233, range 110-296); 22 as duplicates at
zero-lag r >= 0.50 on 100 ms bins. **All 22 duplicate pairs are within the same
microwire bundle and share a region label** (21/22 by bundle key; the 22nd is adjacent
contacts of one UCLA probe). That is the signature of one neuron on two wires, so the
criterion is doing what it claims — worth stating explicitly in the methods.

Retained population: median 4118 spikes, 10th pct 747, median FR 1.56 Hz, median RPV
0.00%. All three criteria assessed as sound; see `docs/cell_qc_methods.md`.

**Two problems found:**
1. The manuscript states duplicates were removed at *r = 70*. The run actually used
   **r >= 0.50**. At 0.70 only 9 units would have gone rather than 22. Methods text
   needs correcting.
2. **The QC code that produced `abcd_passed.mat` is not in the repo.** Its recorded
   settings include `MinOverallFR_Hz` and `SessionLowFR_Hz`, which appear in none of
   the four QC .m files and in no git commit. Decisions survive in
   `qc_all_sessions.mat`, but the pipeline is not reproducible as it stands.
   Separately, `qc_master_summary.txt` (Aug 2025) passed only 347/924 = 37.6% at
   nominally identical thresholds — almost certainly faulty; mark it superseded.

**New:** `scripts/plot_cell_qc_figure.py` -> `derivatives/group/cell_qc/`
(publication figure + per-cell metrics CSV + settings.json). Reads the stored QC
metrics; recomputes nothing.

Note: 924 in the manuscript is NOT the QC output (984) — it is what survives the
>= 3-subject ROI rule afterwards. The Aug QC run coincidentally had 924 as its
denominator; do not conflate them.

### 2026-08-27 — QC is reproducible again: `scripts/run_cell_qc.m`

Wrote a self-contained MATLAB script that regenerates the accepted-cell set and
`abcd_passed.mat` from `abcd_data_08-Sep-2025.mat`. Verified against the canonical
`qc_all_sessions.mat` (2026-04-16) cell by cell across all 1042 units:

    electrodeLabel differing : 0        RPV      max|diff| 0.00e+00
    n_spikes       differing : 0        corr_max max|diff| 0.00e+00
    accept/reject  differing : 0        fail reason differing : 0

1042 -> 984 (58 excluded), and `abcd_passed_rebuild.mat` matches the canonical file
in size, session count, cell count and electrode-label order.

**Resolved the missing-parameter question.** The canonical run recorded
`MinOverallFR_Hz = 0.1` and `SessionLowFR_Hz = 0.1`, which appear in no script. They
were never applied: 13 accepted units fire below 0.1 Hz (min 0.049 Hz). They are
deliberately NOT implemented in `run_cell_qc.m` — implementing them would change the
accepted set. The three criteria in the script (spike count, RPV, within-bundle
correlation) reproduce the canonical split exactly.

Outputs use an `OUT_SUFFIX` (default `_rebuild`) so nothing canonical is overwritten
until a rebuild has been verified.

### 2026-08-27 — new cell ROI table from site files only

`scripts/build_cell_roi_table.py` ->
`derivatives/ROI_assignment/cells_from_site_files_2026-08-27/neurons_with_ROI_labels_v2.csv`

Every coordinate read from the recording site's own electrode file. Utah files are
resolved by the patient ID each file declares in its own `Fname` (never by
coord-matching), and the Utah coordinate and microwire label are read from the SAME
row index, so no ordering assumption remains.

    baylor_file_micro         608     as shipped
    utah_file_micro           168     as shipped
    ucla_file_micro           140     as shipped
    baylor_file_305to152       35     BY2-YEN, same file's MNI305 + Fischl affine
    baylor_micro_from_macro    26     BY2-YER, macro + 3.15 mm (Baylor's own constant)
    no_electrode_file           7     UT1-202302 (3), UT1-202422b (4)

ROI counts, published -> rebuilt: EC 38->35, HC_anterior 276->291, HC_mid 231->233,
PCC 61->61, **mOFC 163->139**, mPFC 155->158. Cells with an ROI 924 -> 917.

**The 7 cells with no electrode file** take the collaborator's `regionLabel` from
`abcd_data_08-Sep-2025.mat`, flagged `roi_provisional=True`: UT1-202302 -> ROFC,
UT1-202422b -> RHC. Her labels carry hemisphere but not the medial/lateral or
anterior/mid distinction this taxonomy needs, so they are recorded coarsely
(`OFC_unsplit`, `HC_unsplit`) rather than invented. Both fall below the
>=3-subject rule (one subject each) and so carry `alt_final_roi = NaN` — they are
identifiable in the table but do not enter per-ROI analyses. Note this is a real
change for UT1-202422b: those 4 cells were previously counted as **mOFC** because
they sat on the placeholder coordinate; the collaborator calls them hippocampus.

Columns added: `has_coordinate`, `coord_source`, `roi_source`, `roi_provisional`,
`collaborator_regionLabel`, `published_alt_final_roi` (for diffing).

## 2026-08-27 — cell ROI pipeline consolidated onto site electrode files

`scripts/cell_to_roi_july26.py` is again the single script for cell ROIs. Its
coordinate section now reads every coordinate from the recording site's own
electrode file; `scripts/build_cell_roi_table.py` (a temporary standalone) was
deleted rather than left as a second entry point.

**s47 and s53 arrived from the collaborator and close the last gap.** Both declare
the correct patient in their own `Fname`: `s47/Electrodes.mat` -> 202302,
`s53/Electrodes.mat` -> 202422. **All 984 cells now have a coordinate from their own
patient's file — zero unresolved.** (The previously-used `s47/electrodes/` file is a
different patient, 202311; one session folder can hold two patients' files, which is
why folder position is not trusted.)

**The collaborator's labels were right.** With the real files:
  UT1-202302  chan97/102  she said ROFC -> file says mROFC1/mROFC6, atlas mOFC
  UT1-202422b chan116/119 she said RHC  -> file says mRHIP4/mRHIP7 at
                                           (23.5,-16.6,-17.4), atlas HC_anterior
The 4 UT1-202422b cells had been counted as **mOFC** off the placeholder coordinate.
She said hippocampus; she was correct.

**Final coordinate provenance (984 cells):**
    baylor_v2026_bundle_micro                     608
    utah_file_micro                               175
    ucla_file_micro                               140
    baylor_v2026_bundle_305to152_unreliable_file   35
    baylor_v2026_micro_reconstructed_from_macro    26   <- the only inference left

**alt_final_roi:** HC_anterior 275->295, HC_mid 232->233, mPFC 155->158,
**mOFC 163->142**, PCC 61->61, EC 38->35, NaN 60. 33/984 cells changed.

**Also changed, per the rule that anatomy read-in must match across analyses:**
- `load_ucla_v2026` now filters `isMicro`, so a micro cell cannot match a macro row
  (measured: it never did, so this is a guard, not a change).
- `contact_anatomy.resolve_utah_mat` (LFP pipeline) now resolves by declared patient
  ID via the same index. Utah LFP sessions resolved: **18/18**, up from partial.
- New in `anatomy_sources`: `subject_numeric_id`, `index_utah_files_by_id`,
  `index_utah_mats_by_id`, `utah_micro_coord`, `build_micro_label_map`,
  `mat_patient_id`, `mat_text`.

**NULL / dead end (do not re-run):** `discover_utah_mats()` coord-matching. No
uniqueness constraint, assigned s47's file to six patients, and circular — it
validates electrode files against the hand-entered coordinates it is meant to
replace.

**Script inventory after consolidation:**
    scripts/run_cell_qc.m            which units enter    -> abcd_passed.mat
    scripts/cell_to_roi_july26.py    cell anatomy + ROI   -> neurons_with_ROI_labels.csv
    scripts/swr_build_contacts.py    LFP contact anatomy  -> macro_contacts_all.csv

### 2026-08-27 — re-run readiness after the ROI rebuild

Audited all 18 analysis scripts against the rebuilt ROI table. Checklist:
`docs/rerun_after_roi_update.md`.

**Key fact making the re-run cheap:** the analysed cell set is unchanged — same 984
cells, the same 60 excluded by the >=3-subject rule, 0 cells entering or leaving,
33 moving between ROIs. Per-cell statistics therefore do not need recomputing; only
the ROI grouping does, which is what the existing `RELABEL_FROM` hooks do.

**One genuine trap found and fixed.** `RSA_DSR_ROIs_simple.py` had
`RELOAD_RUN = '2026-07-30_15-58-51-fixed_cells-fixed_perms'`, which skips the RSA
and permutation loop and re-renders plots from the old saved CSVs. A re-run would
have silently reproduced the previous result. Set to `None`; old tag kept in a
comment.

**Checked and left alone:** `REUSE_PERMS_FROM_PREVIOUS_RUNS = True` in the same
script is safe — the perm-cache fingerprint includes `cell_ids`, so any ROI whose
membership changed rebuilds its null and PCC (unchanged) legitimately reuses.
`per_lag_encoding.py` (RELOAD + RELABEL), `spatial_peaks_simple.py` and
`encoding_state_sustained_cv.py` (full run + RELABEL) were already correct.

**Four hardcoded upstream run directories** would otherwise mix old and new results.
Marked in source with `# >>> RERUN-CHECK` (`grep -rn "RERUN-CHECK" scripts/`):
`cell_gradient_master_table.py:CELL_TABLE`, `cell_fMRI_angle_match.py:MASTER_DIR`,
`overlay_double_dissociation.py:DEFAULT_PER_CELL_CSV` and `:DEFAULT_PER_LAG_CSV`.

**Unaffected (fMRI / behaviour / rodent only):** behaviour_summary,
create_fMRI_model_RDMs_on_clean_beh, analysis_rodents_complete_clean,
fMRI_run_RSA_without_rsatoolbox_clean, fMRI_mask_vs_cluster_extract,
harmonic_angle_maps, fMRI_run_RSA_instruction, svc_loso_test, plot_cell_qc_figure.

## 2026-08-28 — cell_to_roi_july26.py leaned; glass-brain plotting centralised

Output verified **byte-identical** to the pre-cleanup table (984 cells, mOFC 142,
EC 35, HC_anterior 295). 1790 -> 1633 lines.

**Deleted the text-based intent rescue (98 lines).** `RESCUE_MAX_DIST_MM`,
`INTENT_MAP` and the rescue bookkeeping (`n_rescued`, `rescue_rows`,
`rescue_source_atlas_label`, the unreachable rescue-breakdown report). It had been
disabled since 2026-07-29 and printed "[DISABLED]"; the 4 leftover cells stayed
leftover either way.

**Kept, deliberately:** `_neighborhood_search` and the amygdala/EC boundary pass
inside step 4. That pass is *not* the intent rescue — it is purely
coordinate-based (reassign Amygdala -> EC when Juelich entorhinal is within 3 mm)
and the amygdala pass calls the helper. It currently reassigns 0 cells, but it is
live, principled code, not dead weight. `rescue_dist_mm` is retained because it
records that pass and step 5 reads it.

**Glass-brain plotting moved to `mc/plotting/cell_results.glass_brain_cells()`**
with module constants `GLASS_MARKER_SIZE`, `GLASS_DISPLAY_MODE`, `GLASS_DPI`,
`GLASS_FIGSIZE`, `GLASS_DEFAULT_COLOUR`. Four of the nine figures now call it
(leftover scatter, per-hint leftovers, master ROI plot, per-ROI atlas overlay, and
the reusable `_master_plot`). The remaining figures build bespoke contour legends;
they now take their marker size and colours from the same constants rather than
hardcoding them.

**⚠ Corrected a palette conflict.** `mc/plotting/cell_results._EXTRA_ROI_COLORS`
had `HC_anterior = '#a30d6c'` (magenta) commented as a "CLAUDE.md override".
CLAUDE.md actually assigns **#23677E to HC_anterior** and **#a30d6c to lOFC**, and
`cell_to_roi_july26.py` followed CLAUDE.md. The module is now correct
(HC_anterior teal, lOFC magenta) and completed with PHC and the non-target ROIs,
so the script's local `ROI_COLORS` is derived from it instead of duplicating it.

**This changes figure colours in three other scripts** that call `get_roi_colour`:
`spatial_peaks_simple.py`, `roi_labelling_glassbrain_overview.py` and
`mpfc_coord_shift_glassbrain.py`. Anterior hippocampus will render teal rather
than magenta in those. Revert by swapping the two entries back if the published
figures need to match the old colours.

### 2026-08-28 — ⚠ the gradient analysis is running on stale coordinates

`cell_gradient_master_table.py:279` reads `MNI_x/y/z` from the per-lag
`per_cell_ALL_ROIs.csv`, not from `neurons_with_ROI_labels.csv`. Those coordinates
are frozen at the per-lag **base** run (2026-06-30): `relabel_per_cell` rewrites the
`roi` column only — it never touches coordinates. So the reload+relabel workflow
that is correct for every ROI-grouped analysis is **wrong for the gradient**, which
is about coordinates.

Measured against the canonical table: **140 of 158 mPFC cells carry stale
coordinates.**

    session 52 (Utah)    6 cells   42.1 mm off -- still at the placeholder
                                   (4.55, 29.50, -20.63) vs real (~3, ~19, +20.7);
                                   the z sign flips
    session  6           4 cells   10.4 mm
    sessions 45, 46      5 cells    7.0 mm
    sessions 43,44,49,
             61,62,7,8,9          3.3-4.8 mm  (Baylor - stale from an earlier
                                   coordinate change, not from this rebuild)

Both the old (2026-08-22) and new (2026-08-28) gradient runs use the same stale
source, so the old-vs-new comparison below is internally consistent but **neither
reflects the corrected anatomy**.

Old vs new gradient overlap (both on stale coordinates):

    mPFC cells             155 -> 158
    inside gradient mask    74 ->  72
    ventral / dorsal      42/32 -> 42/30
    PC1 median split     -13.81 -> -13.81 (unchanged)
    ventral pooled lag      30° ->  30°   (unchanged)
    dorsal  pooled lag      60° ->  60°   (unchanged)
    fMRI theta at sites  25-126° -> 25-126°
    recording sites          16 ->  15

**Fix required before trusting any gradient number:** take coordinates in
`cell_gradient_master_table.py` from `neurons_with_ROI_labels.csv` (join on
subject + cell idx, as `relabel_per_cell` does) rather than from the per-cell lag
CSV, which should supply only the lag statistics.

### 2026-08-28 — gradient analysis fixed and re-run on canonical coordinates

**Fix.** `cell_gradient_master_table.py` now refreshes `MNI_x/y/z` from
`neurons_with_ROI_labels.csv` (join on subject + cell idx, the keys
`relabel_per_cell` uses) via a new `refresh_coordinates()`, and raises rather than
proceeding if any cell is unmatched. `CELL_TABLE` supplies the per-cell lag
statistics only. Measured refresh: 158 mPFC cells, median shift 3.32 mm, max
43.1 mm, **143 cells moved > 1 mm**.

**`harmonic_maps_brain_overlay.py` checked and found correct** — it already maps
`MNI_*_final` onto `MNI_*` at load (lines 396-397). No change needed.

**Result (`cell_gradient_master/2026-08-28_15-19-35`):**

    run                      mPFC  in_mask  ventral  dorsal  vent_lag  dors_lag  sites
    paper / pre-rebuild       155       74       42      32       30        60      16
    new ROIs, stale coords    158       72       42      30       30        60      15
    new ROIs, FIXED coords    158       87       48      39       30        60      19

**The ventral-to-dorsal progression is unchanged: ventral 30°, dorsal 60°.** With
correct coordinates it now rests on more cells and more recording sites, not fewer.

Peak strengths shift: ventral r 0.059 -> 0.079, dorsal r 0.125 -> 0.081. The fMRI
gradient angle range sampled by the cells widens at the low end, 25-126° -> 5-120°.

**Paper numbers to update:** "74/155 mPFC units overlap the gradient" -> **87/158**;
"42 neurons at the ventral end and 32 slightly dorsal" -> **48 and 39**;
"n = 16 recording sites" -> **19**. The median split boundary and the 30°/60°
peaks stand.

### 2026-08-28 — inferential test of the ventral/dorsal gradient split

`scripts/gradient_split_stats.py`. Unit of inference is the recording site (87
in-mask cells sit at only **19 distinct coordinates**); the permutation shuffles
the ventral/dorsal label across sites and rebuilds the pooled profiles through the
same code path.

    pooled argmax difference   30 deg (30->60)   p = 0.37
    pooled circular-mean diff  23 deg (29->53)   p = 0.47
    circ-linear corr, angle vs axis position     r = 0.069, p = 0.88
    circ-circ corr, cell pref vs fMRI at site    r = -0.079, p = 0.74

Null argmax shift: median 0 deg, IQR 0-30 deg. A one-bin shift is the smallest
non-zero difference the 12-lag grid allows and occurs by chance in ~1/3 of
permutations. **The ventral/dorsal split is descriptive, not statistically
supported.**

Also: after the coordinate fix the fMRI gradient angle at the cells' own sites is
**58 deg (ventral) vs 61 deg (dorsal)** — a 3 deg difference. The manuscript clause
"consistent with the fMRI progression from 30-75 deg" is not supported at the cell
locations; before the fix these read 71 deg and 100 deg, which is where the
apparent correspondence came from.

Counts to update: 74/155 -> **87/158** in mask; 16 -> **19** recording sites;
42/32 -> **48/39** ventral/dorsal.

Write-up incl. smoothness-index suggestions for Fig 3b:
`docs/gradient_split_stats_and_smoothness.md`.

### 2026-08-28 (addendum) — two corrections on the gradient stats

**1. Tests against zero DO exist and are nominally significant.** I had tested only
the ventral-vs-dorsal contrast. At each group's own peak, one-sided:
ventral 30 deg r=0.079 t(47)=2.44 p=0.009 (cell) / t(11)=0.81 p=0.219 (site);
dorsal 60 deg r=0.081 t(38)=2.21 p=0.017 (cell) / t(6)=2.24 p=0.033 (site).
Caveats: testing at the argmax chosen from the same data is circular (Bonferroni
over 12 lags takes ventral to p=0.11), and it does not test the progression claim.

**2. The fMRI map DOES show a clear progression; my earlier number was wrong.**
Angle by MNI z in the gradient mask runs 59 deg (z -5) -> 95 deg (z +10) ->
158 deg (z +28) -> 333 deg (z +38), monotonic. My "58 vs 61 deg" came from the
pipeline's per-cell lookup, which is a **single voxel** (SPHERE_RADIUS_MM = 0)
averaged arithmetically rather than circularly. I quoted it without checking its
derivation.

The real issue is cell placement: ventral cells span z -4.6 to +4.1 (mean 0.8,
12 sites), dorsal z +1.2 to +9.8 (mean 2.8, 7 sites) -- **2 mm apart with
overlapping ranges**, sampling only the flattest ventral stretch. Recomputed with
the pipeline's constants, all three sampling schemes give dorsal <= ventral:
(a) COM vertex 52/46, (b) per-site vector mean 58/45, (c) mask voxels <=8mm 84/66.
Likely because PC1 = [-0.04, -0.41, 0.91] mixes y and z, and the dorsal group has
higher z but lower y.

**Recommended claim (option d):** the units lie at z -5 to +10 where the map reads
~60-95 deg, overlapping the immediate-future quarter (0-90 deg). Defensible;
the ventral-vs-dorsal progression is not.

### 2026-08-28 (addendum 2) — vs-zero tests stored, and the z-projection works

**New:** `scripts/gradient_split_vs_zero.py` -> `gradient_split_vs_zero.csv`
(cell / site / subject units x 12 lags + the pre-specified 30+60 window, one-sided
vs zero, BH-FDR across the 2 groups and across the 12 lags).
`scripts/gradient_fmri_z_projection.py` -> `fmri_angle_z_profile.csv`,
`fmri_angle_by_group.csv`.

Dorsal cluster is above zero at every unit (q = 0.033-0.067); ventral survives only
at cell level and neither survives the 12-lag correction, so the peak-lag test
should be reported as the pre-specified 30+60 window.

**The z-projection recovers the correspondence.** Vector-mean angle across
gradient-mask voxels per z-slab: 62 deg at z=-2 -> 71 (z=0) -> 82 (z=+2) ->
89 (z=+4) -> 97 (z=+10). Read off at the groups: **ventral z=0.8 -> 70.5 deg,
dorsal z=2.8 -> 81.6 deg** — an 11 deg progression in the predicted direction.

This supersedes the earlier "58 vs 61 deg", which came from the pipeline's
single-voxel lookup averaged arithmetically rather than circularly. Caveats: it is
descriptive, not a test, and the z-ranges overlap. The fMRI range over the cells'
z-span is **~70-97 deg**, not the 30-75 deg the manuscript states.

### 2026-08-28 (addendum 3) — consolidated gradient results table

**New:** `scripts/gradient_results_table.py` ->
`<run>/final_splits/gradient_results_summary.csv`. One tidy long-format table
holding every statistic quoted for Fig 3d, in three blocks:

    contrast  the ventral-vs-dorsal tests (site-level permutation, seed 42).
              These existed nowhere on disk before -- they had only ever run
              interactively -- so they are now reproducible.
    vs_zero   one-sided tests against zero at cell / site / subject level,
              merged from gradient_split_vs_zero.csv
    fmri_z    where each cluster sits on the fMRI gradient z-profile,
              merged from fmri_angle_by_group.csv

`scripts/gradient_split_stats.py` absorbed into it and deleted, so the contrast
tests have a single home.

Permutation values reproduce exactly from seed 42 (argmax p = 0.3722, circular
mean p = 0.4673, circ-linear r = 0.0688 p = 0.8696, circ-circular r = -0.0787
p = 0.7441).

Gradient scripts now, in run order:
    cell_gradient_master_table.py    per-cell master + canonical coordinates
    cell_fMRI_angle_match.py         ventral/dorsal splits
    gradient_split_vs_zero.py        vs-zero tests   -> gradient_split_vs_zero.csv
    gradient_fmri_z_projection.py    fMRI z-profile  -> fmri_angle_*.csv
    gradient_results_table.py        contrasts + merge -> gradient_results_summary.csv

### 2026-08-28 — manuscript gradient-section change list

`docs/manuscript_gradient_section_changes.md`. Every value in the
ventral-to-dorsal gradient section, its Figure 3 legend and the Methods
subsection, checked against the re-run results.

Changes: 74/155 -> **87/158** in mask; 16 -> **19** sites; 42/32 -> **48/39**;
median split -13.81 -> **-13.51 mm**; mPFC 60 deg t(31)=2.245 p=0.016 ->
**t(32)=2.156 p=0.019**; mid HC 0 deg r .055 t(34)=2.27 p=.0147 -> **r .060
t(35)=2.320 p=.0131**; mid HC 330 deg r .038 t(34)=2.072 p=.0229 -> **r .033
t(35)=1.745 p=.0449** (weakens); Fig 3 legend mPFC 155/32 -> **158/33**, HC mid
232/35 -> **233/36**, overlap 74/81 -> **87/71**.

Unchanged: the 30 deg / 60 deg peaks, PC1 = [-0.04, -0.41, 0.91].

Two items **unverified** because `harmonic_angle_maps.py` has not been re-run:
the subject-wise centre-of-mass linear trend (t(32) = 2.78), and the Fig 3b range
"~30 deg ventrally -> ~360/0 deg dorsally" -- my z-profile of the gradient mask
reads ~59-87 deg at the ventral end, so the ventral figure looks too low, but that
is a different computation (surface projection, per-sub-model t>=1.5) and should be
re-derived rather than taken from my number.

The sentence needing most work is the ventral/dorsal one: peaks unchanged, but the
difference is not significant (site-level permutation p = 0.37) and the "consistent
with the fMRI progression from 30-75 deg" clause should be replaced with the
z-projection read-out (ventral 71 deg, dorsal 82 deg). Suggested wording in the doc.

### 2026-08-28 (addendum 4) — gradient stats folded into cell_fMRI_angle_match.py

Deleted `gradient_results_table.py`, `gradient_split_vs_zero.py` and
`gradient_fmri_z_projection.py`. The two read-outs that are kept now live inside
`scripts/cell_fMRI_angle_match.py`, so this part of the analysis is still the
two scripts it always was (`cell_gradient_master_table.py` then
`cell_fMRI_angle_match.py`).

New outputs in `<run>/final_splits/`:

    lagwise_vs_zero.csv        one-sided t vs zero at ALL 12 lags, CELL and
                               SUBJECT level, for all three split schemes
                               (168 rows). BH-FDR across the groups of a scheme
                               within each (unit, lag).
    fmri_z_readout.csv         per group: n cells, n sites, z min/max/mean, the
                               fMRI angle at z mean, and the angle spanned across
                               the group's z range.
    fmri_angle_z_profile.csv   the map's vector-mean angle per 1 mm of MNI z
                               (z -10 to +48), for plotting.

Key values (pc1_ventral_dorsal), unchanged from the standalone scripts:

    cell    ventral 30 deg  r .079  t 2.44  p .0093  q .0185
    cell    dorsal  60 deg  r .081  t 2.21  p .0167  q .0334
    subject ventral 30 deg  r .051  t 1.20  p .1297  q .2595
    subject dorsal  60 deg  r .139  t 2.26  p .0226  q .0452
    ventral  z -4.6 to 4.1 (mean 0.8)  fMRI 70.5 deg at mean, spans 72-89 deg
    dorsal   z  1.2 to 9.8 (mean 2.8)  fMRI 81.6 deg at mean, spans 72-97 deg

The cross-lag rows also show the specificity: ventral is flat at 60 deg
(r = -0.002) and dorsal is weaker at 30 deg (r = 0.049) than at its own 60 deg.

**Dropped, per user decision:** the ventral-vs-dorsal permutation contrast
(argmax p = 0.37, circular mean p = 0.47) and the two continuous correlations.
Note this removes the basis for stating that the ventral/dorsal difference is
not significant -- that caveat now has no stored statistic behind it.

Also fixed while wiring this in: `load_master_provenance` is now called
unconditionally (the z-profile needs the harmonic-root and mask paths, not just
the brain rendering).

### 2026-08-28 (correction) — the fMRI 30 deg region is real; my read-outs were wrong

I twice reported the wrong fMRI angle for the ventral/dorsal cell groups and, on
that basis, wrongly advised that the manuscript's "consistent with the fMRI
progression from 30-75 deg" was unsupported. **It is supported.**

    circular median at the cells' own voxels   ventral  36 deg   dorsal  84 deg   <- correct
    same, eighths map                          ventral  37 deg   dorsal  75 deg
    vector mean at the cells' voxels           ventral  58 deg   dorsal  61 deg   <- misleading
    whole-mask z-profile at group mean z       ventral  70 deg   dorsal  82 deg   <- wrong region

Two compounding errors. (1) I summarised a broadly distributed circular variable
(0-120 deg at the recording sites) with its **vector mean**, which pulls towards
the middle and hid a 36->84 deg separation behind "58 vs 61". (2) Correcting that,
I switched to a **whole-mask z-profile**, which averages the entire y (16-70) and
x (+-14) extent of the gradient mask rather than the neighbourhood the electrodes
occupy -- that region is dominated by 90-120 deg voxels.

`fmri_z_readout.csv` now stores both, named so they cannot be confused:
`fmri_at_cells_median_deg` / `_vec_mean_deg` / `_min_deg` / `_max_deg` (quote
these) and `fmri_zprofile_at_z_*_deg` (context only). Added `_circ_median_deg`
to `cell_fMRI_angle_match.py`.

**Manuscript impact:** item 3 of the change list is withdrawn -- "theta = 30 deg to
75 deg" should stay (measured 36-84 deg); only 74/155 -> 87/158 changes. The
suggested rewording for the ventral/dorsal sentence now quotes 36 deg and 84 deg.

## 2026-08-29 — SWR macro contacts rebuilt on the corrected anatomy

`scripts/swr_build_contacts.py` re-run after the cell-anatomy work. Three fixes,
then 28 -> 32 sessions with a hippocampal bipolar pair (78 -> 98 pairs).

**1. Utah files resolved by declared identity** (from the 2026-08-27 work).
`s47` had been using patient 202311's electrode file and yielded **0** hippocampal
contacts; with its own file (202302) it yields 8 contacts / 3 pairs. `s24` moved
from s23's file to its own (10 -> 8 contacts). Net +1 session.

**2. `index_utah_mats_by_id` now prefers the file that carries the macro arrays.**
It took the first file per patient, which for s04 is a bare top-level
`ChannelMap.mat` with no `ElecMapRaw`/`ElecXYZMNIRaw`. Candidates are now scored on
those keys. Also fixed `merged.setdefault(...)` in `build_macro_table` -- a
DataFrame has no `setdefault`, a latent crash in the no-anatomy branch that only
fired once a mat without macro arrays reached it.

**3. Atlas fallback for `is_hpc` where the site segmentation is silent.**
Six Baylor subjects (YEL, YEP, YEQ, YER, YEU, YFT) ship an electrode file whose
`ROI_DK2005_3mm` column is **0 % populated**, against 91-98 % for every other
subject. Reading only that column concluded "no hippocampal contact" and dropped
**10 sessions** -- a property of the file, not the patient. `_hpc_with_atlas_fallback`
now uses the probabilistic atlas where the site said nothing.

    NULL / dead end: gating that fallback on `native_roi` (the MAPPED ROI) is
    wrong -- it is None for every non-target region, so the atlas overruled 110
    contacts the site had explicitly placed in white matter, fusiform gyrus and
    the inferior lateral ventricle. Gate on `native_region`, the raw
    segmentation string. Caught before it reached any result.

    Final scope: 69 fallback contacts, ALL with `native_region == 'Unknown'`,
    in 3 subjects (YEL, YER, YEU). Recovered s10, s11, s18, s25.
    `hpc_source` ('native' / 'atlas_fallback' / 'none') is written per contact
    so the fallback can be audited or excluded downstream.

**Current state:** 32/60 sessions usable locally (18 baylor, 1 ucla, 9 utah ->
now 22 baylor, 1 ucla, 9 utah), 98 hippocampal bipolar pairs.

**Still excluded:** 26 sessions have **zero raw LFP files on this machine** --
a data-location issue that should resolve on ceph, and the single biggest
remaining gain. s16 (BY2-YEP) reads only 4 channels from its raw header, which
looks like a wrong-nsx or header problem worth a look. s04 (UT1-202216) resolves
its electrode file but 0 of 132 channels join: the table carries `bLACG6`-style
anatomical labels while the raw header carries clinical names (`LCM1`...), so
that join needs a channel-index bridge.

### 2026-08-29 — hippocampal contacts: coordinate-only, one per electrode

Per user decision, macro-contact location is now determined **solely from the MNI
coordinate**. All inference of brain location from site-supplied region strings
was removed.

**Deleted:** `native_roi_label` (the region-string -> ROI mapper) and
`_hpc_with_atlas_fallback` (yesterday's native/atlas hybrid). `native_region` is
still carried in the table as metadata; nothing reads it. The white-matter
reference picker now filters on `atlas_roi` rather than `native_roi`.

**New:** `anatomy_atlas.hippocampal_probability(coords)` -- P(hippocampus) in per
cent from the Harvard-Oxford subcortical PROBABILITY maps. A max-prob atlas gives
a label, which cannot rank two contacts that are both "hippocampus"; picking one
contact per electrode needs a continuous measure, and probability makes
"deepest in the structure" the winner.

    NULL / dead end: `fetch_atlas_harvard_oxford('sub-prob-2mm')` returns 22
    labels but only 21 volumes -- 'Background' has no volume, so the 4-D index
    is label_index - 1. Indexing by label position returns 0 % everywhere,
    including at the hippocampal centroid. Caught by the sanity check.

**New:** `contact_anatomy.select_hpc_contacts()` -- per probe, rank contacts by
`hpc_prob` and keep the single highest, provided it clears `HPC_PROB_MIN = 25 %`
(matching the maxprob-thr25 atlases used elsewhere). Adds `hpc_prob`,
`hpc_rank_in_probe` and a strictly one-per-probe `is_hpc`.

**Result:** 108 hippocampal contacts across 108 distinct (session, probe) pairs --
**0 probes with more than one**, invariant verified. 32/60 sessions, 107 bipolar
pairs (baylor 76, ucla 3, utah 28; one Utah probe has no valid reference
partner). Selected contacts: P(hippocampus) median 61 %, IQR 42-84 %, min 26 %.

Of the 128 runner-up contacts on those same probes, 56 would themselves clear
25 % -- i.e. the old rule was admitting roughly twice as many contacts, several
per electrode, which is what the one-per-probe constraint now prevents.

`swr_build_contacts.py`: the atlas is no longer optional. Contact selection is
coordinate-based, so without it no contact can be chosen, and the run yields zero
pairs loudly rather than silently falling back to labels.

### 2026-08-29 — docs and preflight updated for the coordinate-only rule

`data/final_results/ripple_analysis/methods.md`:
- **§4.1 rewritten.** Was "ROI: native-space primary"; now "from the MNI coordinate
  alone". States why the earlier rule was dropped (0 %-populated column for six
  Baylor subjects costing 10 sessions; not comparable across sites; cannot rank
  contacts within a probe) and reports the concordance honestly: of 108 selected
  contacts, 90 have a site label and **71 (79 %)** of those agree. The 19
  disagreements are fusiform (6), amygdala (7), ventricle (5); they carry lower
  confidence (median P(HC) 51 % vs 63 %). Flags the five ventricle contacts as
  worth revisiting once ripple rates exist.
- **§4.2 rewritten.** Anchor is now the highest-P(hippocampus) contact per probe,
  not the most medial. Notes that 56 of 128 runner-ups would clear 25 %, so the
  one-per-probe constraint is doing real work.
- **§4.3** replaced with the current state: 108 contacts, 107 derivations, 32
  sessions, 22 subjects; invariant verified; span median 5.48 mm. Flags that this
  covers only 34 of 60 sessions locally, and that **s61-s63 are absent from the
  config entirely**.
- **§3.5** gains the Utah file-identity rule (`Fname` / `PatientIDStr`), why
  coord-matching was dropped, and that s47 holds two different patients.

`HOW_TO_RUN.md`: cluster prerequisites rewritten. **`nilearn_data` is now
REQUIRED** (the old text said it was not) -- rsync `~/nilearn_data/fsl` (~34 MB)
and set `NILEARN_DATA`, since `sub-prob-2mm` is a new dependency that will not be
in an older cache. Added the rsync for Utah `Electrodes.mat` files, which live in
three different subdirectory names and at the top level for s47/s53.

`scripts/swr_check_inputs.py`: nilearn reclassified from "not required" to
required, and the preflight now actively probes `hippocampal_probability()` at a
canonical hippocampal voxel, failing loudly with the rsync instruction if the
atlas is unreachable.

## 2026-08-29 — first full cluster run, and repo-hygiene fixes

**Cluster result: 50/60 sessions, 173 bipolar derivations** (baylor 131, ucla 11,
utah 34), against 32 sessions / 107 pairs on the development machine. The
one-per-electrode invariant holds at every site (`n_hpc == n_pairs`). Baylor's
131 contacts give 128 `n_hpc_pairs` because 3 anchors are assigned EC by the atlas
ladder despite clearing the P(hippocampus) threshold -- the HC/EC border case.

**⚠ NEVER write into the repository.** `scripts/batch_swr_on_ceph.sh` used
`logs_path="./logs/..."`, relative to the CWD, so SLURM `.out`/`.err`/`.sbatch`
files were written into the git tree -- and an earlier run's had been *committed*.
It now resolves the data root (`$SWR_DATA_ROOT`, then ceph, then the local path)
and writes to `<data_root>/derivatives/group/swr/slurm_logs/`. The tracked `logs/`
directory was removed and `logs/ *.log *.out *.err *.sbatch` added to `.gitignore`.
Swept the rest of the SWR scripts: no other relative output paths.

**Stale `bipolar_pairs` files could send stage 2 down the wrong path.**
`swr_check_inputs.py` counts pair FILES on disk, not sessions that produced pairs
in the latest build, so a file left over from an earlier run reads as "ready for
stage 2". That is the 52-vs-50 discrepancy in the cluster output: 52 files, 50
sessions with pairs, so 2 files are stale. `swr_build_contacts.py` now deletes a
session's pair file when that build produces none, and logs the removal.

**The Utah preflight warning was misleading.** It looked only in
`s{NN}/electrodes/` and reported sessions 30, 42 and 53 as missing an electrode
file. The pipeline resolves by the patient ID each file declares in its own
`Fname`, across several subdirectory names -- s30 and s42 have no file of their own
but share a patient with s29 and s41 and resolve fine. The check now uses
`index_utah_mats_by_id`, the same path the pipeline uses.

**Still to diagnose (9 sessions):** excluded as "no channel matched an electrode
table". Locally this was 2 (s04 Utah, s16 Baylor); on ceph it is 9, so more raw
data has exposed more channel-naming mismatches. Plus s39, which has no `.ns2`
files on ceph at all.

### 2026-09-01 — Utah channel join: two naming conventions, only one supported

The cluster run left 9 sessions at "no channel matched an electrode table", six of
them Utah with exactly 132 channels and **0** resolved (s04, s41, s42, s47, s48,
s53) -- including s47 and s48, which resolve fine on the development machine.

**Cause.** `build_macro_table` keyed the Utah join on `^chan(\d+)$` only. Utah
recordings use two naming conventions and both occur in this dataset:

    s01 / s02 / s23   chanN 128/132 | label 0/132     <- the old key works
    s04               chanN  15/132 | label 89/132    <- every channel dropped

The channel names and the electrode-table labels are the *same vocabulary*
(`LAMG`, `LANT`, `LCM`, `LINS`, `LPHIP`, `bLACG`, `bLAHIP`, `bLMCG` appear in
both), so this is a rename, not different anatomy.

**Fix.** The Utah branch now builds both candidate keys and uses whichever
resolves more channels. Trailing analog channels (`EyeX`, `EyeY`, `Pupil`, `BP`)
fail to match under either key and are reported unresolved -- never matched on
position, which the s02 132-names-vs-128-columns case rules out.

Locally: Utah resolved 809 -> **898**, HC contacts 29 -> 31, sessions **32 -> 33**
(s04 recovered). One-per-probe invariant still holds.

**New:** `scripts/swr_diagnose_channels.py` prints, per session, the channel
source, the first channel names, how many match each candidate key, and the
electrode table's labels -- so a join failure shows the mismatch instead of just
reporting zero. Prints only; writes nothing.

Not addressed yet: s16 (BY2-YEP) reads only 4 channels from its raw header;
s50/s51 (UCLA) are Blackrock but the UCLA join expects `.ncs` filename stems;
s39 has no `.ns2` files on ceph.

### 2026-09-01 — cluster diagnostic: the 9 failures split into four causes

`swr_diagnose_channels.py` on ceph resolved every remaining failure to a cause.

**(a) Clinical channel names, fixed by the dual-key join** — s04, s41, s42, s48.
Their channels are `LCM1` / `RMPFC1` / `RINS1`, not `chanN` (only 14-16 of 132
match). s04 verified: `join by chanN 0/132, join by label 89/132`.

**(b) Electrode file never reached ceph** — s47 and s53 report "electrode table:
NONE for patient 202302 / 202422". Both files exist locally
(`s47/Electrodes.mat`, `s53/Electrodes.mat`, 8 MB each) and are the TOP-LEVEL
Utah files. The rsync include pattern missed them. Nothing to fix in code.

**(c) Degenerate channel cache** — s16's `channels.npy` holds exactly
`['empty-064','empty-128','empty-192','empty-256']`, shadowing a 232-channel
recording. `_load_channels` now ignores a cache whose names are all placeholders
and falls through to the raw header.

**(d) UCLA Blackrock sessions have no bridge** — s50/s51 read `chan129..chan256`
from an `.ns3` header while their xlsx lists `LAI-1` / `LA1`. Unlike Utah, the two
vocabularies do NOT overlap, and the localizations xlsx carries no channel-number
column, so there is nothing to join on. The other UCLA sessions are Neuralynx,
where the `.ncs` filename IS the electrode name. **These two need a channel map
from UCLA; they cannot be recovered from the files we hold.**

Plus s39, which has no `.ns2` files on ceph at all.

Diagnostic now prints both candidate key match counts and says which wins, rather
than only testing `chanN`.

### 2026-09-01 — misfiled-file warning, subject-level coverage, UCLA verdict

**s47 is clean locally, not on ceph.** `s47/Electrodes.mat` declares 202302
(correct for session 47) and `s47/electrodes/Electrodes.mat` is now absent here,
so each folder holds one patient. ceph still carries the misfiled 202311 copy
under `s47/electrodes/`, which is why s48 was served from s47's folder -- both
declare 202311 and the index took whichever sorted first.

`index_utah_mats_by_id` now prints `[MISFILED] patient X declared by BOTH a and b`
when the two locations are in DIFFERENT session folders. Companion files in the
same folder (`Electrodes.mat` + `ChannelMap.mat`) are normal and no longer warn --
the first version flagged all 16 patients, which would have trained the reader to
ignore it.

**UCLA s50/s51 cannot be recovered from the files we hold.** The cell pipeline
never needed a channel bridge: UCLA cells are named `elec2`, `elec36`, and were
matched to the localizations xlsx **by coordinate** (`ucla_file_micro`), inheriting
`source_electrode` such as `RAI_micro-1`. The macro LFP has no coordinates of its
own to match with -- it has channel names (`chan129..chan256`) and needs a mapping
to xlsx electrode names (`LAI-1`, `LA1`). Those vocabularies do not overlap, and
the xlsx carries no channel-number column (checked). The other UCLA sessions are
Neuralynx, where the `.ncs` filename IS the electrode name, which is why they work.
**These two need a channel map from UCLA.**

**Subject-level coverage now reported by `swr_build_contacts.py`**, since several
subjects contribute 2-3 sessions and a session count overstates independent
coverage. Locally: 24 subjects, 24 with LFP, **23 with a usable hippocampal
derivation (96 %)**; the one exception is BY2-YEP, whose only session with data
(s16) has the degenerate channel cache.

s39 has no LFP file at all (confirmed by the user: never received). It is reported
as excluded rather than treated as a failure.

### 2026-09-01 — remaining SWR failures resolved or classified

After the dual-key join and the two rsyncs, the ceph diagnostic shows **6 of 9
recovered**:

    s04  join by label 89/132      s47  join by chanN 91/132  (own file now present)
    s41  join by label 88/132      s48  join by label 90/132  (own file, not s47's)
    s42  join by label 88/132      s53  join by chanN 82/132  (own file now present)

s47 now reads `s47/Electrodes.mat` (202302) and s48 reads
`s48/electrodes/Electrodes.mat` (202311) -- each session on its own patient.

**The remaining three are missing data, not defects:**

- **s16** -- its two `.ns3` files are both the **NSP-2** amplifier and each carries
  4 placeholder channels (`empty-064`...). The raw header gives the same 4 as the
  cache, so this was never a degenerate cache: the NSP-1 files simply are not
  present. Checked every session: **s16 is the only one with NSP-2 and no NSP-1**
  (all others are NSP-1 only), so `files[0]` is not picking the wrong amplifier
  anywhere else. Exclusion reason now says so rather than "channel list unreadable".
- **s39** -- no LFP file was ever received.
- **s50 / s51** -- need a channel map from UCLA (see the previous entry).

**Refactor:** `_load_channels` existed twice, in `swr_build_contacts.py` and
`swr_diagnose_channels.py`, and had already drifted -- the degenerate-cache check
was only in one, so the diagnostic reported s16 differently from the build. Moved
to `contact_anatomy.load_channel_list()`; both call it.

### 2026-09-01 — ripple QC figures

**New:** `scripts/swr_plot_ripples.py` -> `<session>/LFP-ripples/<run>/figures/`
  grand_average_ripple    mean waveform + ripple-locked TFR (the checkpoint)
  examples_best           clearest accepted events
  examples_borderline     accepted events nearest the threshold
  examples_rejected       what the spectral criterion discards

Validated on s02 (239 ripples, 0.168 Hz, Chen ~0.17-0.24): the TFR shows a tight
narrowband peak at ~95 Hz centred on t = 0, inside 80-120 Hz -- a real ripple, not
a broadband artifact.

Two plotting corrections made while building it:
- the mean of the BAND-PASSED signal is near zero however strong the ripples are,
  because ripple phase is not locked across events. Plot the mean ripple
  ENVELOPE; the first version showed a flat red line and looked like a failure.
- events too close to a recording edge were skipped after selection, leaving holes
  in the example grid. Select from cuttable events instead.

**Two things for the user to judge from these figures:**
1. The borderline accepted events (z = 3.0-3.1) show near-continuous 80-120 Hz
   activity with only a modest increase in the detected window. Whether
   `PEAK_SD = 3.0` is too permissive is a judgement call these panels make visible.
2. Several REJECTED events look like clean ripples (e.g. z = 7.6 at 90 Hz).
   Strict spectral rejection is 32.1% here against Chen's 23.4% +- 9.9%; the
   relaxed variant gives 18.2%. Worth deciding which is primary before the full run.

**PSD questions answered:** the notch is adaptive (`notch_ratio_threshold = 2.0`),
so sessions differ by design -- s02 had a 60 Hz line-noise ratio of **16958x**,
notched to a residual of **0.014x**, while a session with no line noise is left
untouched and shows a smooth 1/f. A narrowband peak INSIDE the ripple band (the
~90 Hz spike on one s01 contact) is not addressed by the notch and is a genuine
concern for that contact.

### 2026-09-01 — QC consolidated into swr_qc_report.py, plus a numeric checkpoint

**`swr_plot_ripples.py` deleted, merged into `swr_qc_report.py`.** That script
already produced the 6-panel checkpoint figure, so a second plotting script was
duplication. `swr_qc_report.py report --session=N` now also writes
`figures/examples_{best,borderline,rejected}.pdf`.

**New: a numeric checkpoint.** "Looks fine" does not scale to 56 sessions, so the
methods.md section 6 criteria are now evaluated numerically against Chen's
reference values, with FAIL (outside a hard range -- do not analyse as is) and
CHECK (outside the reference range -- look at it) verdicts:

    rate_hz              hard 0.05-0.60   ref 0.17-0.24
    spectral_reject_pct  hard 5-50        ref 13.5-33.3   (Chen 23.4 +- 9.9)
    peak_freq_hz         hard 80-120      ref 85-115
    duration_ms          hard 38-500      ref 40-120
    clean_frac           hard 0.33-1.0    ref 0.50-1.0
    ripple_gain          hard >1.2        ref >1.5

`ripple_gain` is the mean ripple-band envelope at the peak over its value at the
window edges -- a detector triggering on broadband noise gives ~1, so it turns
"the grand average looks like a ripple" into a number.

    swr_qc_report.py report  --session=N   figure + grids + metrics
    swr_qc_report.py metrics --session=N   metrics only (fast, for the cluster)
    swr_qc_report.py group                 aggregate every session -> triage table

**Bug found by the new metric:** `clean_frac` was medianed over *all* derivations
including ones already excluded for contamination, so s02 reported 0.361 (a CHECK)
when the analysed pair actually keeps 0.586. An excluded pair must not drag down the
session it was excluded from. `qc_metrics` now filters on `excluded` first.

s02 after the fix: rate 0.168 (CHECK, two ripples below Chen's 0.17), rejection
32.1%, peak 100 Hz, duration 56 ms, clean 0.586, ripple gain 2.98. No FAILs.

**PEAK_SD = 3.0 is kept, and the visual worry about it was wrong.** Events at the
threshold look unconvincing on the example grids, but that is a plotting artifact:
the band-passed trace is scaled x9-x34 to share an axis with broadband, which
magnifies the ongoing band activity as much as the burst. Measured per event
(envelope at peak / envelope at +-0.20-0.25 s), the z=3.0-3.5 bin has median gain
2.65 with 1% below 1.5 -- these are real bursts. Raising to 3.5 would drop 94 of 238
events (0.168 -> 0.098 Hz, well below Chen) for +0.26 gain. Recorded in methods.md
section 6.3 with the full table.

**NULL / dead ends (do not re-run):** rate over *total* recording time is 0.098 Hz
and is the wrong number -- Chen's denominator is artifact-free time, giving 0.168.

**Fixed a misleading plot title.** `qc_psd.png` said "Bipolar derivations after
notch" whether or not the notch had fired, so a session with no line noise looked
like a filter failure. It now reads either "notch applied at 60, 120, 180 Hz" or
"no notch needed (line-noise ratio below threshold)".

**Group QC across the 8-session development set — no FAILs, all usable.**

    session  clean  dur_ms  peak_Hz  rate_Hz  gain  reject%  verdict  n
      s02    0.586    56     100.0   0.168   2.98   32.1     CHECK   239
      s03    0.546    61      97.5   0.190   3.12   36.7     CHECK  1057
      s06    0.686    58     100.0   0.198   3.00   28.1     PASS    692
      s09    0.643    56      97.5   0.154   2.92   35.0     CHECK   592
      s12    0.685    58      97.5   0.158   2.94   41.1     CHECK   378
      s13    0.675    58      97.5   0.176   2.80   36.8     CHECK   327
      s14    0.564    58      95.0   0.180   2.87   34.3     CHECK   301
      s38    0.608    59      97.5   0.206   2.97   31.9     PASS    608

Rate 0.154-0.206 (Chen 0.17-0.24), peak frequency 95-100 Hz, duration 56-61 ms,
ripple gain 2.80-3.12. Those four are tight across sessions, sites and montages,
which is the real evidence that detection is stable.

The one systematic deviation is spectral rejection: 6 of 8 sessions above Chen's
upper bound, mean 34.5% strict vs 21.6% relaxed. Relaxed matches Chen almost
exactly -- and that is NOT a reason to switch, because picking the criterion that
best reproduces another paper's number after seeing which one does is post-hoc.
Strict stays primary as pre-declared; relaxed remains the sensitivity analysis on
19% more events (5006 vs 4194). See methods.md section 6.2.

### 2026-09-02 — Chen Fig 2 replication, and what it revealed about our averages

**`swr_qc_report.py report` now runs every session by default** (`--session=N` for
one), then draws the pooled group figure and prints the group triage table. New
`mc/plotting/ripple_figures.py` holds the publication figures.

**Three questions about ripple quality, answered with measurements.**

1. **Amplitude was never off by an order of magnitude.** The comparison had been
   between different traces. Chen's grand average is +-5 uV and their example
   ripple is +-20-25 uV broadband; ours are +-2 to +-9 uV and +-20-50 uV. Mean
   single-event ripple-band amplitude is 3.79 uV (range 1.46-9.07 across 14
   derivations). The large numbers are the broadband trace in both papers.

2. **Our grand average was cancelling itself.** Detection marks events at the
   peak of the RMS *envelope*, which carries no phase, so the 80-120 Hz
   oscillation sat at a random phase in every snippet:

       envelope-locked (was)  0.50 uV in the average --  7% of single-event
       trough-locked  (now)   7.18 uV in the average -- 95% of single-event

   Trough-locking reproduces Chen's figure and lands on their +-5 uV scale.
   DISPLAY ONLY -- detection, counts, rates and all statistics still use the
   envelope peak. `trough_lock` is called from plotting code and nowhere else.

3. **The sharp wave: present on some contacts, cancelled on others, absent on
   the rest.** Two findings, the second of which corrected the first.

   (a) The QC figure had been plotting an 8-40 Hz band-pass as "sharp wave",
   which high-passes away a deflection 40-100 ms wide. Median |SNR| 0.28 vs 1.37
   for a 20 Hz low-pass, now used.

   (b) New control `swr_qc_report.py sharpwave --session=N` re-reads the raw
   channels WITHOUT the bipolar subtraction (`preprocess_session(monopolar=True)`)
   and averages the same event times on each contact separately. Detection never
   runs on them. Ripple-locked deflection below 20 Hz, in uV:

       session  derivation             monopolar  bipolar  reduction
       s38      RT2aA03-RT2aA04             8.12     1.49      5.4x
       s38      RT2cHbE02-RT2cHbE03         7.29     2.13      3.4x
       s38      RT2bHa02-RT2bHa03           6.31     3.05      2.1x
       s06      LPHIP1-LPHIP2               7.87     6.24      1.3x
       s03      LMH-1-LMH-2                 3.42     6.58      0.5x
       s03      LPH-1-LPH-2                 1.89     1.01      1.9x
       s03      RMH-1-RMH-2                 1.07     0.61      1.8x
       s06      bLAHIP1-bLAHIP2             0.28     0.57      0.5x
       s02      bLHIP1-bLHIP2               0.11     0.15      0.7x

   5 of 9 derivations carry a ripple-locked deflection >=2 uV before subtraction;
   in 3 of those the bipolar difference removes most of it (2.1-5.4x); and in 4
   derivations a >=2 uV deflection SURVIVES into the analysed bipolar signal.
   The other 4 show nothing even monopolar -- s02, the development session, is
   one of them, which is why the first look suggested it was missing everywhere.
   Per-contact heterogeneity is what the anatomy predicts: the dipole reverses
   across the CA1 pyramidal layer.

   CAVEAT NOT YET RESOLVED: the "monopolar" trace is still referenced to the
   amplifier reference, so a deflection shared by two neighbours is equally
   consistent with a local sharp wave and with a global ripple-locked slow
   potential. The discriminating test is the same average on a NON-hippocampal
   contact from the same subject; `swr_build_contacts.py` does not emit those.

   Reporting: call the events **ripples**. The evidence is the ripple band, not
   morphology. Show the sharp wave per contact where it survives, never pooled
   (signs cancel).

   Two bugs fixed in this control before trusting it: the monopolar channel key
   followed "is ns_pos non-NaN" instead of the reader, silently missing every
   UCLA session; and the verdict compared SNR, which is not comparable across
   montages because bipolar shrinks the flank noise along with the signal -- it
   reported the opposite of what the microvolt amplitudes show.

**NULL / dead ends (do not re-run):** averaging raw snippets on the envelope
peak (cancels to 7%); using an 8-40 Hz band to show the sharp wave (SNR 0.28);
judging the monopolar-vs-bipolar sharp wave by SNR rather than microvolts.

**Speedup:** scipy `hilbert` FFTs at the signal's own length, which for a real
recording is an awkward number -- 1.08 s per frequency for s02 against 0.14 s at
`next_fast_len`, an 8x saving that matters because the TFR runs one pass per
frequency per derivation. The zero-padding perturbs only the edges (median
difference 8e-8 uV, interior max 1.1e-5 uV against a 0.35 uV median envelope).
Adopted in the plotting TFR. NOT applied to `swr_detect._event_baseline_spectra`,
which would change existing detection output -- available if detection is ever
re-run from scratch.

Per-session TFRs are cached to `ripple_stacks.npz`; the group figure reads those
rather than recomputing, since the TFR is the expensive part of the QC stage.

## 2026-09-02 — DSR main effect on the fsaverage surface; fixing the "two islands" artefact

**Problem.** The surface overlay in `scripts/harmonic_maps_brain_overlay.py`
drew the DSR main-effect boundary as two disconnected blobs, while the
volumetric (FSLeyes) view of the same cluster is one elongated medial-wall
band. Readers could not tell that the outline *is* the main effect.

**Cause (visualisation only — no analysis was affected).**
`nilearn.surface.vol_to_surf(vol, pial)` with a single mesh defaults to
`kind='line'`: it averages the volume along the vertex normal over ±3 mm.
Thresholding that mean at 0.5 demands that more than half of a 6 mm line lies
inside the cluster, which erodes an elongated cluster that only grazes the
cortical ribbon into disconnected gyral patches.

**Fix.** Binary masks (the ROI gate and the DSR outline) are now projected
with the FreeSurfer `--projfrac-max` recipe: sample at 11 depths strictly
between the white and pial surfaces, take the max, then close 2 mesh rings and
drop patches < 40 vertices (speckle only; counts printed and logged in the
settings JSON). Continuous cos/sin maps are unchanged. Implemented in
`scripts/dsr_main_effect_surface.py` and imported by
`scripts/harmonic_maps_brain_overlay.py`; new figures carry a `_ribbonmax` tag
so the old renderings are not overwritten.

**Effect on the projected cluster (cluster-mass FWE p < 0.05):**

| hemi | old vol_to_surf | ribbon-max raw | + 2-ring closing | final |
|------|-----------------|----------------|------------------|-------|
| lh   | 2333 vtx, 8 patches  | 3264 vtx, 13 patches | 3521 vtx, 6 patches | 3476 vtx, 4 patches |
| rh   |  864 vtx, 1 patch    | 1058 vtx,  4 patches | 1164 vtx, 2 patches | 1160 vtx, **1 patch** |

**Anatomical finding that resolves the figure.** The medial-wall part of the
cluster is right-lateralised in its dorsal extent. On the **right** hemisphere
the cortical ribbon intersects it as a *single* connected patch spanning
z = 14–56 mm — the elongated band seen in the FSLeyes sagittal view. On the
left, the ribbon intersection genuinely breaks in two (z = 15–36 and
z = 37–49), which is the "two islands" look. **The rh medial view is therefore
the honest and the visually clean panel**; the lh medial view additionally has
the lateral-OFC cluster shining through the translucent pial surface, which
adds spurious-looking speckle.

**New script.** `scripts/dsr_main_effect_surface.py` renders the main effect on
fsaverage in three modes — `cluster_t` (t inside the FWE cluster, red-yellow,
black boundary), `dual` (t > 1.7 faded + FWE cluster opaque, the FSLeyes
two-layer look, implemented as one MNE layer with per-vertex opacity because
MNE keeps a single `data` overlay), and `binary` (solid fill). Outputs +
`settings.json` in
`data/derivatives/group/Main_Results_fMRI/dsr_main_effect_surface/<date>/`.

The `dual` mode confirms the complaint about the FSLeyes screenshot: at
t > 1.7 the sub-threshold layer covers most of the brain and, on a translucent
surface, also shows through from the far hemisphere. Recommendation is
`cluster_t` on the pial glass brain, which needs no second threshold.

### 2026-09-02 (later) — diagnostic figures for everything we throw away

Standing requirement from SK: visualise what is removed, never just count it.

**New `figures/artifact_rejection.pdf`** (per session). Nothing previously
visualised the artifact stage at all, despite it deleting 35-54% of every
recording. Four panels: per-criterion flagged fraction; union + 1 s padding per
derivation against the 2/3 exclusion line; when in the recording; and the
removed stretches next to kept ones ON THE SAME Y-AXIS, marked by which criteria
fired. `swr_artifact.artifact_mask(..., return_per=True)` now optionally returns
the individual criterion masks so the marks are accurate.

s38: removed stretches are +-250 uV IED-shaped spikes, kept windows are +-25 uV
ordinary LFP. The 54% loss is real but it is not data. Individual criteria flag
only 0.04-3.2% each (`artifact_criteria.csv`); the total is almost entirely the
+-1 s padding around several hundred crossings.

**New `figures/sharp_wave_examples.pdf`.** Single ripples from the derivation
with the largest surviving low-frequency deflection, broadband + <20 Hz + ripple
band overlaid. On s38 RT2bHa02-RT2bHa03 the ripple sits on a clear slow wave.
Deliberately a best-case display and labelled as such -- the typical case is the
per-derivation table where 4 of 9 show nothing.

**Example-ripple selection fixed.** The Chen panel picked the event with the
largest RMS z, which reliably landed on the contact with the biggest slow waves,
i.e. the one where the ripple is least visible. Now scored by ripple envelope at
the peak / SD of the same window below 40 Hz -- how clearly the ripple stands out
from its background. The example panel also carries the band-passed overlay now,
because a single unaveraged hippocampal trace is busy (Chen's example is too).

**`HOW_TO_RUN.md` section 4 rewritten** (I started a second run doc, then
deleted it -- `HOW_TO_RUN.md` already covered this and better, including the
cluster rsync steps). It now carries the new entry points, a table of which
figure would invalidate what, and the warning not to expect a sharp wave under
the grand average. Two stale `swr_qc_report.py --session=N` invocations fixed:
the CLI now needs a verb.

**Follow-up (same day).** `cluster_t` on the translucent pial surface, medial
view, was chosen as the house style. `harmonic_maps_brain_overlay._make_brain`
now delegates to `dsr_main_effect_surface.make_brain`, so surface, size,
background, `cortex='low_contrast'` shading and `SURF_ALPHA` are defined once
and cannot drift between the main-effect panel and the gradient overlay; the
duplicated `_set_brain_surface_alpha` fallback moved there too. The gradient
overlay is rendered for lh + rh medial only (`RENDER_FILTER`).

### 2026-09-02 (later still) — repo cleanup, and the hypothesis tests

**Archived 7 scripts** to `derivatives/group/swr/archived_scripts/` (also still
in git history). Repo is down to 10 SWR scripts, all on the pipeline path.

    swr_diagnose_blocks.py       block structure is resolved, recorded in session_blocks.csv
    swr_diagnose_channels.py     the joins are fixed, contact_qc.csv reports the outcome
    swr_collect_figures.py       swr_qc_report.py report now does every session + the group
    swr_organise_box_download.py the download is done and sorted
    swr_plot_discovery.py        superseded by swr_hypotheses.py
    swr_build_windows.py         "
    swr_h1_stats.py              "

**Per-uncovering behaviour IS available** -- `windows_error_correct` carried a
caveat saying it was not, which was wrong. New `mc/analyse/swr_behaviour.py`
reconstructs every uncovering attempt and its outcome in session seconds from
the 25 ms derivatives:

  - clock: session_s = grid_onset + bin * 0.025, verified against
    all_trial_times on s02 to within one bin (max 0.02 s)
  - attempts: transitions into the uncover key in buttons_per_25ms_*
  - correctness: the 4 reward times per repeat are the correct ones

Validated against the stimulus PC's own BEH_*.mat on s47: truth 1903 attempts /
1288 correct / 615 errors, reconstruction 1891 / 1285 / 606 = 99.4% of attempts
and 99.8% of correct uncoverings. The raw .mat is NOT used directly: its
BUTTON_PRESS_TIMES are on the stimulus PC clock (s54 starts at 509637 s, machine
uptime), not the photodiode clock, and only 5 of those files exist locally.

Sanity: s47 discovery repeats average 16 errors / 20 uncoverings / 9.4 s; later
repeats 0.7 / 4.7 / 3.0 s.

**New `scripts/swr_hypotheses.py`** -- four hypotheses, one script. H1 (ripples
at the FIRST arrival at D) is primary and pre-declared, uncorrected; H2-H4 are
secondary and FDR-corrected across themselves. Confounds handled by a
log(artifact-free seconds) offset, a movement-key covariate in every model,
fixed window lengths within design, pooled rates as summed counts over summed
exposure, subject-level error bars, and circular-shift permutation as the
primary inference.

NOTE: the permutation null is NOT centred on zero (H1: -0.14). A log ratio of
small counts is biased downward and first-D windows are few. That is exactly why
inference is against the shifted null rather than against zero -- the null is
built by the identical code and carries the identical bias.

Bugs fixed while building: `n_events` vs `n_ripples`; `g.sem` returning the
DataFrame *method* rather than the column (attribute shadowing); H2 keying on a
`phase` column that windows_pauses does not emit (it is `phase_after`); H3
building a window frame without going through `_finalise`, so it lacked
duration_s.

**Development-set hypothesis results (8 sessions, 6 subjects, 200 perms) --
EXPLORATORY, not a result:**

    H1  first_D vs later_D          +24.5%   z=+2.84  p=0.005   (primary)
    H2  exploration vs later pause   -4.6%   z=-0.32  p=0.64   q=0.67
    H3  rate -> later errors      b=-0.79    z=+1.81  p=0.055  q=0.16
    H4  error vs correct feedback   -4.5%   z=-0.37  p=0.67   q=0.67

H1 is in the predicted direction: 0.217 Hz at the first arrival at D vs 0.173 Hz
at later arrivals in the same grid. Its GLM p is 0.22 -- the disagreement is
because a cluster-robust SE over 6 clusters is itself unreliable, and the
permutation is primary (section 13.2).

H3 points the right way (more ripples after first-D -> fewer subsequent errors,
holding discovery difficulty and pause length constant). WARNING for anyone
reading the GLM output: it reports p < 0.0001 while the permutation reports
0.055. The GLM treats 297 grid-derivations as independent when several share a
session and therefore share ripples. TRUST THE PERMUTATION. Averaging
derivations within a grid before the regression should be done before reporting.

H2 and H4 are flat (|z| < 0.4). If that holds on the full set the reading is
that the effect is specific to the moment the plan becomes knowable -- but that
must not be written up as if a null had been predicted there.

Figure bug fixed after the run: the H2 panel keyed on `phase` and collapsed to a
single "pause" bar; the column is `phase_after`.

### 2026-09-02 (cont.) — three more hypotheses, and a portable results bundle

**Three hypotheses I had left untested, added.** SK asked for them and one of
them (H6) already had its window builder written -- `windows_discovery` -- which
I should have run in the first pass.

    H5  explore / plan / execute as THREE phases. add_phase's "exploration"
        merges searching for unknown rewards with knowing all four but not yet
        executing reliably; new add_phase3 separates them.
    H6  is D special BECAUSE it completes the plan? The interaction
        (D - mean(A,B,C)) on the first traversal minus the same later.
    H7  feedback that could change behaviour vs feedback that could not:
        correct DURING discovery, error AFTER it, vs the other diagonal.

Development-set (8 sessions, 6 subjects, 60 perms):

    H5  plan vs execute pauses          -7.6%   z=-0.57  p=0.70
    H6  (D - ABC) first vs later       +38.7%   z=+3.12  p=0.016  q=0.049
    H7  informative vs uninformative    -3.0%   z=+0.27  p=0.43

**H6 is the discriminating test and it comes out the right way.** Within the
discovery traversal D carries +46% the ripple rate of A (GLM p=0.0016) and that
advantage is GONE at later traversals (interaction p=0.0020). "D is special
because it is last / ends the traversal" predicts a main effect of state with no
interaction. It should probably be the primary contrast for the full run rather
than H1, which is the same claim in a weaker form.

**New `swr_hypotheses.py export`** -- one pickle (+ the same tables as CSVs) with
everything needed to redo any of these statistics WITHOUT the LFP: ripple
timestamps per derivation, artifact-free intervals, pairs with coordinates,
behaviour per repeat, every uncovering attempt with its outcome, channel QC.
1.9 MB for 8 sessions, so ~14 MB for the full set against tens of GB of LFP.

WARNING recorded in the bundle's meta and in HOW_TO_RUN: a ripple rate is events
per ARTIFACT-FREE second. Anything that divides by wall-clock time is wrong --
that is why `intervals` is in the bundle.

CLI is now `swr_hypotheses.py run` / `swr_hypotheses.py export` (fire needs a
verb once there is more than one entry point).

Fixed: the summary's "role" column still read "FDR across H2-H4" after the
secondary set grew to six.

### 2026-09-02 (cont.) — the audit now says which warnings matter, and s09 fails

**Two cluster-run failures, both mine.**

1. `swr_qc_report.py report` on ceph errored with "no value for the required
   argument: session". The cluster copy is STALE -- the session=None default and
   the dispatcher are local and unpushed.
2. `batch_swr_on_ceph.sh qc_rep scripts/swr_qc_report.py report <condfile> ...`
   -> "condition file not found: report". The batch script's third argument IS
   the condition file; it runs `python <script> <condition line>` verbatim, so a
   fire VERB has to be inside the condition lines, not on the submit command.
   `swr_make_conditions.py` now also writes `swr_qc_<name>.txt` with the verb
   already in each line: `metrics --session=2 --analysis_name=swr_v1`.

**The audit could not tell a cosmetic warning from a dangerous one.** 17 of 60
sessions came back `needs_review`, which is unusable as a signal because almost
all of it is bookkeeping: a null `segment`, a missing `blocks` key, a YAML block
count disagreeing with the behaviour. The YAML is a hint, not the authority
(methods 3.3), so none of that can corrupt a result.

`swr_audit_sessions.py` now runs the CLOCK GATE the plan always called for:
read the raw file headers, map each behavioural block onto its recording, and
report the head/tail margins. New columns `clock_status`,
`min_head_margin_s`, `min_tail_margin_s`; new status `clock_failed`. The report
separates "the clock is fine, the rest is bookkeeping" from "this one can
corrupt a result".

**CORRECTION to the paragraph below, made the same day.** The first version of
the gate treated ANY overrun as a failure, which was wrong and would have thrown
away four good sessions. A behavioural clock can overrun the recording for two
unrelated reasons: the recording was stopped a second or two before the task
ended (truncation -- harmless), or the block mapping is wrong (corrupting). They
are told apart by magnitude. Across all 60 sessions: 1 genuine failure (s32,
-1905.5 s, files=2 vs beh_blocks=1) and 4 truncations of 0.9-2.0 s (s09, s10,
s26, s33). Truncation needs no action at all -- a window past the end of the
recording gets zero artifact-free exposure and `fit_count_glm` already filters
`exposure_s > 0` (both verified directly).

**So s09 does NOT invalidate the H1/H6 results.** It loses roughly its last two
seconds of behaviour, which is at most one window. The earlier claim that the
development-set numbers needed re-running without s09 was wrong.

**What the gate found (original wording, kept for the record): s09 FAILS.**
Block 2's offset is the
cumulative duration of block 1 (1527.4 s), which is only valid if recording
never stopped. Behaviour block 2 needs 1402.9 s and sits 53.4 s into a 1454.2 s
file, so it overruns the end by 2.0 s. Every block-2 event in s09 is misaligned
by up to ~2 s -- against 2 s analysis windows, that is total.

**s09 is in the 8-session development set**, so the H1/H6 numbers reported
earlier include one session with a broken clock. They need re-running without it.

**A false alarm I introduced and removed in the same pass:** the >= 5 s margin
criterion also flagged s38 as "tight" (3.0 s tail). s38 is SINGLE-block --
behaviour and LFP share t=0, there is no offset to get wrong, and a small
positive tail margin just means the recording ran on after the task. The
criterion exists to validate multi-block OFFSETS and now only applies when
there is more than one block.

**I archived `swr_diagnose_blocks.py` this morning saying "block structure is
resolved". That was wrong** -- s09 proves it. The capability is back, and better,
as part of the audit rather than a separate script to remember to run.

## 2026-09-03 — full-cohort SWR hypotheses (n=46): the development-set results do not replicate

Ran all seven hypotheses on the full cluster output: **46 sessions, 33 subjects,
52,129 accepted ripples, 166 bipolar derivations** (144 after exclusions), via a new
bundle-backed source so the statistics run on the laptop without the LFP.

### Two method defects fixed first (both affect every earlier p-value)

1. **The permutation was shifting in wall time, not on the artifact-free axis.**
   `methods.md` §specifies the clean axis and `assign_events_to_windows` has a
   `shift_s` parameter built for it, but `_shifted_counts_factory` did
   `t_sh = lo + mod(t - lo + sh, L)` on wall time. Shifted events could land inside
   artifact stretches where the detector could never have found them. Now uses the
   documented path. **All 8-session p-values came from the wall-time null.**
2. **`CleanAxis.to_clean` looped over intervals in Python** (~4,500 per pair) and
   dominated runtime. Vectorised with searchsorted; verified bit-identical against
   the loop on all 166 derivations (max abs difference 0.0).

### Results

| | contrast | observed | z | p (one-sided, pre-declared) | q FDR |
|---|---|---|---|---|---|
| H1 | first-D vs later-D | -5.5% | -1.78 | 0.966 | (primary, uncorrected) |
| H2 | exploration vs later pauses | -6.1% | -2.08 | 0.973 | 1.000 |
| H3 | rate after first-D -> later errors | b=-0.26 | +1.93 | 0.025 | 0.150 |
| H4 | error vs correct feedback | -7.9% | -5.20 | 1.000 | 1.000 |
| H5 | plan vs execute | -5.1% | -1.71 | 0.948 | 1.000 |
| H6 | D special x discovery | -6.2% | -1.35 | 0.916 | 1.000 |
| H7 | informative feedback | +0.4% | +0.77 | 0.218 | 0.653 |

**H1 does not replicate.** Development 8 sessions: +23.3%. The other 38: **-9.6%**.
Across all 46, 21/46 sessions positive, median log RR -0.023. The earlier +24.5%
(z=+2.84, p=0.005) was a small-sample artefact of a lucky draw (s38 +0.69, s06 +0.40,
s02 +0.40).

**H6, the discriminating test, is null** (-6.2%, p=0.916). Neither "D completes the
plan" nor its alternative is supported at n=46.

**H4 is the one well-powered effect, in the direction opposite to the pre-declared
one.** One-sided p=1.000 because the observed value is 5.2 SD on the wrong side of its
null; two-sided p ~ 2e-7. Ripple rate is **7.9% LOWER after an error than after a
correct uncovering**, confirmed independently by the GLM (`feedback[T.error]`
coef -0.159, p=0.0001) with an interaction (coef +0.106, p=0.036: suppression stronger
during discovery). The PETH shows the dip beginning BEFORE the press. Post-hoc
direction -- needs held-out confirmation before it is a claim.

### Definitional sweeps (GLM only, `sweep` verb, n_perms=0)

**H1 has no stable window.** The effect grows monotonically with window length
(0.5s +0.018 -> 5s -0.106), which is the opposite of what an event-locked response
does. The **largest effect is in the 1 s BEFORE D** (log RR -0.205, p=0.0001) --
before the event the hypothesis is about. Clipping at the next repeat halves the 2 s
effect (-0.057 -> -0.020), so part of it was spillover into the next traversal.
Conclusion: the first traversal has a **sustained lower baseline rate**, and the
window-averaged tests measure that offset, not a response at D.

**H3 is not robust.** Only 1 of 12 specifications reaches p<0.05 (variable pause,
per grid, all-later errors: p=0.019). All fixed-window variants null (p=0.29-0.99);
per-derivation p=0.078. `rate_hz = n/pause` is mechanically coupled to the
`log(pause_s)` covariate in the same model, and 36% of grids have zero ripples in
that pause. Do not report H3 as a finding.

### Peri-event time histograms (exposure-corrected, new `describe` verb)

Ripples are **phasically locked to reward uncovering**: a transient peak at
**+250-750 ms**, riding on a first-traversal baseline that is LOWER than later
traversals (~0.17 vs ~0.20 Hz). A 2 s window averages the transient away against the
depressed baseline, which is exactly how H1 becomes -5.5%. The remedy is
baseline-corrected, time-resolved testing (post vs pre within trial, cluster
permutation over time), NOT choosing a better window post hoc.

### Ripple-triggered single units (new: `mc/analyse/swr_units.py`)

24 sessions have both ripples and mPFC units. Row order taken from
`all_cells_region_labels_sub{XX}.txt`, which matches the firing-matrix row count in
every session; `neurons_with_ROI_labels.csv` has FEWER rows (s18 24 vs 25, s45 15 vs
18) and its `cell idx` must not be used for this -- it would mislabel HC cells as mPFC.

| region | units | subjects | mean z | p (units) | p (subjects) |
|---|---|---|---|---|---|
| HC (positive control) | 277 | 19 | +0.131 | 0.0004 | 0.029 |
| mPFC | 93 | 16 | +0.054 | 0.32 | 0.44 |
| PCC | 25 | 3 | +0.515 | 0.0004 | 0.21 |

**Positive control passes** (HC peaks at ripple onset, z~0.28), so spike and LFP
clocks are aligned. **No ripple-locked mPFC modulation** in the pre-declared
0-200 ms window. Caveat: HC's own effect is modest for the strongest ripple
signature in the brain -- if the 25 ms matrices are smoothed rather than raw counts,
fast coupling is blurred and the mPFC null is uninformative. Check before trusting it.

### Rejection bias is not a confound

chi2=10.32, p=0.0057 is significant only on n=66,435. Actual acceptance:
exploration 0.6666, first_correct 0.6638, later_repeats 0.6543 -- a **<=1.9% bias on
counts**, against effects of interest in the tens of percent, and running against H2
(which was null anyway). Artifact rejection differences are not a confound at all:
that is the denominator, already handled by artifact-free exposure.

**NULL / dead ends (do not re-run):**
- H1 as a window-averaged first-D vs later-D contrast, at any window from 0.5 to 5 s.
  The baseline differs; the window cannot fix it.
- H3 with the variable-length pause as both window and covariate.
- H2, H5, H6, H7 as pre-declared -- all null at n=46 with 1000 shifts.
- Peri-ripple mPFC firing in a 0-200 ms window at 25 ms resolution: null, and
  possibly under-powered by smoothing.

## 2026-09-04 — correction: "H1-H7" were never pre-registered

SK pointed out that the seven hypotheses in `swr_hypotheses.py` were written by
Claude from SK's verbal description of the idea. They were never declared in
advance by anyone. Labelling H1 "primary (pre-declared, uncorrected)" and
FDR-correcting H2-H7 as a registered family was therefore wrong on two counts:

* it gave those seven contrasts a confirmatory status they never had, and
* it made a set of exploratory probes read as a failed confirmatory study,
  which is a much stronger negative claim than the data support.

Relabelled to "exploratory" throughout; the q-values are retained but are
descriptive only. **The 2026-09-03 entry should be read with this in mind: the
n=46 nulls mean those particular contrasts do not show an effect, NOT that a
pre-registered prediction failed.**

The project is at the exploration stage: the open questions are when the effect
happens, over what window, and how ripples interact with movement. See
`swr_explore.py` (probes, nothing claimed) and `swr_findings.py` (curated).

N3 (no ripple response at grid onset) is also reframed: under SK's logic that is
the EXPECTED result, since at grid onset no reward has been uncovered and there
is nothing plan-relevant to communicate. It is a confirmation, not a null.

---

## 2026-09-07 — Methods figures for single-unit QC and ROI assignment

`scripts/qc_methods_figure.py` → `data/ephys_humans/derivatives/QC_methods_figure_2026-09-07/`

Two 18 × 4 cm panels (Arial, min 9 pt, 11 pt headings), pdf + png (300 dpi) + svg,
built purely from existing derivatives — no analysis re-run, nothing recomputed:

* `qc_all_sessions_rebuild.mat` (output of `scripts/call_cell_wise_QC.m`)
* `neurons_with_ROI_labels.csv` (output of `scripts/cell_to_roi_july26.py`)
* `abcd_data_08-Sep-2025.mat` — raw spike times, read **only** for the pooled
  ISI histogram; cached to `pooled_isi_histogram.npz` so the 6 GB file is
  touched once.

**Fig 1 — single-unit QC.** (a) spike count/unit, 300-spike cut-off;
(b) firing rate of the 36 spike-excluded vs the 1006 retained units;
(c) pooled ISI over the 984 final units with the 1.5 ms refractory window;
(d) max pairwise correlation (100 ms bins), r = 0.50 dedup threshold.
Panels follow the funnel: a/b cover all 1042 sorted units, d the 1006
base-accepted units, c the 984 final units.

**Fig 2 — from recorded to analysed units.** (a) yield 1042 → 984 → 924;
(b) hippocampal MNI-y with the anterior/mid split at y = −21 mm
(Poppenk & Moscovitch 2013); (c) units per ROI with contributing sessions.

Numbers (all reconcile exactly):

| stage | n |
|---|---|
| sorted units | 1042 |
| excluded, < 300 spikes | 36 |
| excluded, RPV ≥ 1 % | **0** |
| excluded, duplicate r ≥ 0.50 | 22 |
| QC-passed | 984 |
| excluded, ROI < 3 sessions | 60 |
| **analysed** | **924** |

Per ROI: HC_anterior 295 (52 sessions), HC_mid 233 (36), mPFC 158 (33),
mOFC 142 (27), PCC 61 (10), EC 35 (8). Not analysed: Visual 21, PHC 12,
medial_CC 8, Thalamus 8, Insula 7, leftover 4.

Two things worth knowing for the methods text:

1. **The RPV criterion never excluded anything.** Max RPV across all units is
   0.60 %, well under the 1 % threshold (median 0 %). The refractory criterion
   is real but was not binding — the manuscript should say units *satisfied*
   it rather than implying units were removed by it.
2. **No waveforms are stored** in `abcd_data_*.mat` (fields: cellID, channelNum,
   electrodeLabel, excludeCell, psth_trials*, region, regionLabel, roi,
   spikeTimes, spike_rate, unitNum). The Wave_clus waveform criteria
   (consistency, slope, amplitude, trough-to-peak) cannot be plotted from the
   current derivatives; the pooled ISI panel stands in for spike-isolation
   quality. Add waveform panels only if the sorted waveform templates are
   re-exported.

Direction check on the hippocampal split: `mc/analyse/anatomy_atlas.py:273`
assigns `y >= -21 → HC_anterior`, so larger y = anterior. The axis arrow
"posterior → anterior" pointing right is correct as drawn.

### Addendum (same day) — firing-rate descriptives

The 1.58 Hz quoted in Fig 1b is the **median**, not the mean. Panel b now
labels both values "mdn" so they cannot be misread. Firing rate over the whole
session, per unit:

| set | n | mean ± SD | median | IQR | range |
|---|---|---|---|---|---|
| all sorted | 1042 | 2.96 ± 4.06 | 1.50 | 0.52–3.71 | 0.041–32.98 |
| after < 300-spike exclusion | 1006 | 3.06 ± 4.10 | 1.58 | 0.59–3.82 | 0.048–32.98 |
| final QC-passed | 984 | 3.04 ± 4.11 | 1.56 | 0.58–3.82 | 0.048–32.98 |
| the 36 excluded units | 36 | 0.10 ± 0.04 | 0.10 | 0.08–0.13 | 0.041–0.23 |

The distribution is strongly right-skewed (mean ≈ 2× median, max 33 Hz), so
median + IQR is the honest descriptor for the manuscript; quoting the mean
alone overstates the typical unit.

---

## 2026-09-11 — left-hippocampus rewDSR instruction timecourse

Inspected the completed `per_TR_svc_instr_test_full_allTR_2026-08-28` run and
generated a LOSO k=50 plot for `rewDSR_instr` in its existing left Garvert MTL
mask. The broad-mask SVC peak is TR5, MNI [-32, -2, -34], t(31)=8.30,
p_FWE < .0001 (1326 voxels × 12 TRs). This coordinate is not hippocampus in
the Harvard–Oxford 50% atlas, so it should be called left MTL/HC-EC rather than
a hippocampal peak.

The anatomical hypothesis was a representation in hippocampus; its timing was
left open and is corrected across all 12 TRs. Repeated the same SVC and LOSO
pipeline in the Harvard–Oxford max-probability 50% left hippocampus mask.
Results are in
`data/derivatives/group/per_TR_svc_rewDSR_instr_HO50_HC_L_2026-09-11/`, with
the derived mask saved below its `masks/` directory and full parameters in
`settings.json`.

* n=32; 519 in-brain left-hippocampal voxels; 12 TRs; 10,000 sign flips.
* SVC peak: TR4, MNI [-30, -22, -20], t(31)=7.873, p_FWE < .0001, corrected
  jointly over hippocampal voxels × TRs.
* LOSO k=50: peak t at TR4, t(31)=6.743, p_FWE < .0001 across 12 TRs; all 12
  TRs significant. Held-out beta is already positive at TR0, reaches its raw
  maximum at TR3, and then declines. This supports a sustained representation,
  not a representation that first appears at D.
  The raw means are 0.05467 ± 0.00863 SEM at TR3 and 0.05151 ± 0.00764 at TR4.
  Their paired difference is not significant (TR4 - TR3 = -0.00316,
  t(31)=-0.906, two-sided p=.372, 95% CI [-0.01028, 0.00396]). TR4 is the
  inferential peak because its smaller between-subject variance gives the
  largest mean/SEM ratio, not because it has the largest raw mean.
* The original broad-mask results are bilateral (right MTL SVC t(31)=6.09,
  TR2, p_FWE=.0009), so no left-lateralisation claim follows without a direct
  hemisphere contrast.
* The older mPFC result currently stored in `per_TR_mask_stats` is n=32
  (therefore df=31, not 32): MNI [-6, 32, 18], TR4, t(31)=5.079,
  p_FWE=.042 from 2,000 permutations over 4,181 voxels × 12 TRs. A manuscript
  value of .041 does not match this saved JSON and should only be retained if
  it comes from a documented later run.

The plotting helper now supports aggregate models such as `rewDSR_instr` with
a compact legend/title and ROI colour, while retaining the fixed A/B/C/D state
colours for split reward-channel plots.

Plotting convention corrected after review: a TR label is the start of a
one-second interval (TR0 = 0--1 s), so numeric timecourse values are now placed
at interval centres (0.5, 1.5, ... s). The reward schedule is drawn at its
true boundaries: A/B/C/D for 1.5 s each on the first pass, then A/B/C for 1 s
each and D for 1 s.

Follow-up bilateral ROI comparison: repeated `rewDSR_instr` in the bilateral
Harvard--Oxford max-probability 50% hippocampus mask (1,049 in-brain voxels).
The bilateral hippocampal result remains significant, peaking at TR4, MNI
[-30, -22, -20], t(31)=7.873, p_FWE < .0001 (10,000 sign flips; correction
jointly over 1,049 voxels x 12 TRs). The existing bilateral Garvert MTL test is
also significant and has the larger corrected peak: TR5, MNI [-32, -2, -34],
t(31)=8.299, p_FWE < .0001 (2,678 voxels x 12 TRs). Per the pre-specified
selection rule, the final memory timecourse therefore uses the bilateral-MTL
search and correction family; the winning voxel is in the left MTL.

Added `scripts/plot_instruction_main_effects_t.py`, which plots replaceable
memory and plan group-t maps at their ROI peaks. It places conditions at their
one-second interval centres, draws the actual reward presentation strip, and
marks conditions significant in each supplied voxel-wise FWE map. It exports
PDF/SVG/PNG plus the exact plotted values and provenance as CSV/JSON. With the
current inputs, bilateral MTL conditions 1--9 are significant. The supplied
execution-model `rewDSR_instr_t.nii.gz` has no significant mPFC condition
(masked peak TR1, MNI [0, 48, 38], t(31)=3.632, p_FWE=.6243), so the orange
trace correctly has no significance bar pending a replacement final map.

### Correction after source-model audit

The publication mPFC effect was initially paired with the wrong later
execution-run map. Its correct source is the original
`group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped` series and the mPFC
small-volume result in `per_TR_svc_instruction_rewDSR_allTR_2026-08-28`:
`rewDSR` peaks at TR4, MNI [-6, 32, 18], t(31)=5.079, p_FWE=.0407, corrected
jointly over 4,179 mPFC voxels x 12 TRs. Only TR4 is significant at the
selected peak. The publication plot has been regenerated from these inputs.

The previously selected bilateral-MTL `rewDSR_instr` result must not be
reported. It came from the `full_no_diag` RSA without a block nuisance. In
that design, instruction-model similarity is confounded with the systematic
within- versus across-task-half similarity difference. This is visibly a
whole-brain offset rather than a regional memory effect: at TR0 the mean
whole-brain t is 2.547 and 97.1% of brain voxels have positive t, even though
no reward location has yet been shown. Across TR0--TR11 the fraction positive
remains 90.2--97.2%. The earlier bilateral hippocampus/MTL max-t results are
therefore retained only as a record of a failed/confounded analysis.

Ran a post-hoc cluster-mass sensitivity analysis on the corrected
`within_only` model after subtracting each subject's whole-brain mean at each
TR. The one-sided cluster-forming threshold was p_uncorrected < .001
(t(31)>3.3749), using 26-voxel connectivity and 10,000 subject-wise sign
flips; the maximum spatial cluster mass was taken across all 12 TRs within
each bilateral ROI. No cluster survived:

* bilateral Harvard--Oxford 50% hippocampus: largest cluster at TR4, 10
  voxels, peak MNI [-34, -20, -18], peak t=3.806, cluster p_FWE=.186;
* bilateral Garvert MTL: largest cluster at TR4, 11 voxels, peak MNI
  [-34, -20, -18], peak t=3.806, cluster p_FWE=.373.

This cluster test is explicitly a sensitivity analysis selected after looking
at the voxelwise result; it does not provide evidence for a hippocampal memory
effect. Added `scripts/per_TR_cluster_test.py` so the empirical and permuted
cluster statistics use the identical function and the analysis can be rerun.

### Exact-source hippocampus/MTL double-check

Re-read all 12 requested
`group_RSA_within_th_only_intr-vs-exe_glmbase_01-TR{0..11}_cropped/`
`cropped_masked_smooth_fwhm5_rewDSR_instr_beta_std.nii.gz` files and reran
10,000-sign-flip inference from scratch in bilateral Harvard--Oxford 50%
hippocampus and bilateral Garvert MTL. Results reproduce the earlier
within-only outputs exactly and rule out a source-file mismatch:

* no demeaning: positive peak HC t(31)=2.264 at TR10, p_FWE=.8822; positive
  peak MTL t(31)=2.440 at TR1, p_FWE=.9383;
* subject/TR whole-brain demeaning: positive peak HC t(31)=3.806 at TR4,
  MNI [-34, -20, -18], p_FWE=.2854; positive peak MTL t(31)=4.699 at TR5,
  MNI [-28, -22, -32], p_FWE=.1110.

All p values above are one-sided and corrected jointly over the ROI voxels and
all 12 TRs. The demeaned maps therefore contain the visually plausible local
TR4--5 pattern, but it is not FWE significant. The matching non-demeaned
cluster-mass test has no positive voxels even crossing the p<.001
cluster-forming threshold; the demeaned cluster results remain p_FWE=.186 in
HC and .373 in MTL.

Also added `scripts/per_TR_roi_mean_test.py` and tested the entire bilateral
ROI mean, removing spatial selection before inference and correcting only
over 12 TRs. No positive whole-ROI effect survives either preprocessing:
without demeaning, HC peak t=-0.378 (p_FWE=.9260) and MTL peak t=-0.829
(p_FWE=.9640); with demeaning, HC peak t=2.038 at TR10 (p_FWE=.1268) and MTL
peak t=.550 at TR3 (p_FWE=.7103). Thus the apparent TR4--5 signal is local,
not a distributed whole-hippocampus effect, and remains subthreshold under
both peak-voxel and cluster-mass correction.

Restricted the demeaned bilateral-HC cluster correction to the a-priori TR4/5
window on request, while retaining the all-12-TR common brain mask for
whole-brain demeaning so that only the temporal correction family changed.
The TR4 cluster is still not significant: 10 voxels, mass=2.6245, peak
t(31)=3.806 at MNI [-34, -20, -18], p_FWE=.0530 across bilateral-HC space x
TR4/5 (10,000 sign flips; one-sided CFT p<.001, 26-connectivity). TR5 contains
one suprathreshold voxel, p_FWE=.1760. A preliminary run that allowed the
demeaning brain mask itself to change with the two-TR subset gave p=.0536;
that value is not the controlled comparison and should not be reported.

Rendered the requested combined group-t timecourse from the corrected inputs:
the demeaned bilateral-HC peak in blue and the original instruction-per-TR
mPFC plan peak in orange. The mPFC TR4 point carries the only p_FWE<.05 bar;
the HC curve is shown descriptively with no significance bar.

## 2026-09-13 — event-locked cumulative instruction overview

Inventoried the new `group_RSA_instr_cumrew_glmbase_instr_*_cropped` outputs
from `rsa_instruction_cumulative_rew.json`. There are 11 event definitions,
33 subjects and 41 maps per condition (not 42): 21 single-model maps and 20
coefficients from combo models. Added
`scripts/plot_instruction_condition_overview.py` and generated 26 matched
plan-memory comparisons in both raw and subject/condition whole-brain-demeaned
versions (52 two-panel PNGs total). The panels show the eight individual
reward presentations plus empty screen, and collapsed-first / collapsed-second
/ empty-screen. Every trace is a descriptive, uncorrected group-t timecourse
at one fixed ROI peak selected across the nine detailed conditions; it must not
be used for inference.

The 26 comparisons cover every stored map without summing across/within:
14 single-model comparisons (seven reward models x within/across plan), four
concurrent plan+instruction combo comparisons, and eight split-coefficient
comparisons (four rewards x within/across plan). Instruction maps are reused as
the within-only memory counterpart, as no across-half instruction map exists.
The common input mask has 144,237 voxels; the intersected ROIs contain 4,165
mPFC and 1,050 bilateral Harvard--Oxford 50% hippocampal voxels.

Demeaning audit on all 451 map-condition combinations: within-plan whole-brain
means never exceeded |t|=2 (0/121); across-plan did so in 6.6% (8/121), all
positive; within-memory did so in 12.0% (25/209), predominantly negative
(10.0% below -2, 1.9% above +2). These are occasional model/condition offsets,
not the near-whole-brain full-RDM confound seen in the obsolete
`full_no_diag` analysis. For this new `within_only` design, raw betas against
zero should therefore remain the primary estimand; whole-brain demeaning is a
regional-localisation sensitivity analysis, not an automatic requirement for
every within-half map. If across and within estimates are later combined 1:1,
combine/average the subject-level beta maps and recompute the group statistic;
never add t-statistics or mix a raw component with a demeaned component.

Descriptively, the raw single-model across-half ABCD plan effect reproduces the
expected mPFC timing and coordinate: the selected mPFC curve peaks at
`see-D-first`, t=4.880, MNI [-6, 32, 18]. The cumulative instruction-memory
sequence is not clean in hippocampus in this uncorrected screen, so no
inferential claim is made at this stage.

### Correction: across-half panels are plan-only

The initial overview incorrectly displayed each across-half plan contrast next
to the corresponding within-half instruction/memory contrast. Those traces do
not form a matched contrast: across-half estimates exist only for execution
(plan), whereas instruction (memory) was estimated only within task halves.
Updated `scripts/plot_instruction_condition_overview.py` so the 15 within-half
panels retain four ROI-by-role traces (mPFC/hippocampus x plan/memory), while
the 11 across-half panels contain only the two plan traces (mPFC and
hippocampus). The corrected v3 output supersedes the v1/v2 figures; its index
leaves `memory_model` empty for every across-half panel.

## 2026-09-14 — event-locked instruction group-t NIfTI export

Added `scripts/save_instruction_condition_tmaps.py` and exported uncorrected
group one-sample-t maps for all 41 stored event-locked instruction RSA models.
For both raw and subject/condition common-brain-demeaned beta maps, each model
has a 9-volume resolved NIfTI (A--D first, A--D second, empty screen) and a
3-volume collapsed NIfTI (collapsed first, collapsed second, empty screen).
The fourth dimension is a named condition axis rather than seconds. Outputs
use the 144,237-voxel mask shared by all 11 condition folders, contain 33
subjects, and carry NIfTI t-test intent metadata with 32 degrees of freedom.

Also exported 11 equally weighted within/across plan estimates under both
preprocessing choices and both condition orderings. Each was computed as
`0.5 * within beta + 0.5 * across beta` separately for every subject and voxel,
followed by the group t-test; group t-statistics were never averaged. The full
export contains 208 t-map NIfTIs and is indexed by `map_index.csv`, with exact
settings and source mappings in `settings.json`.

Corrected the automatic plan/memory classifier in the overview script. The
four plan coefficients from concurrent `*_exe_vs_instr` models had previously
been mislabeled as memory in the whole-brain-offset audit because the combo
suffix contains `instr`; the plotted traces were explicitly assigned and were
not affected. Correct counts and audit summaries are: across plan 121 maps
(mean whole-brain t=-0.079; 6.6% |t|>2), within plan 165 (mean=0.221; 0.6%),
and within memory 165 (mean=-0.884; 14.5%).

## 2026-09-14 — ABCD reward RSA methods and implementation audit

Wrote a supervisor-facing intermediate methods report at
`data/derivatives/group/instruction_cumulative_eventlocked_methods_audit_2026-09-14/ABCD_reward_RSA_methods_and_implementation_audit.md`.
It documents the event-locked GLMs, neural and model RDM construction,
`ABCD_rew` and `ABCD_rew_instr`, single versus concurrent fits, the exact
within/across cell sets, second-level beta-map tests, sign-flip SVC/LOSO,
whole-brain demeaning, and the subject-level 1:1 within/across plan average.

Focused diagnostics over the nine resolved conditions showed median raw to
demeaned group-t shifts of -0.805 to 0.014 for the single instruction-memory
map and -0.902 to 0.008 for its combo partial coefficient. The single
across-half plan map changed from -0.034 to 0.005. Raw/demeaned t-map
correlations were 0.953, 0.937 and 0.974, respectively. These are descriptive
pooled voxel-condition audits, not inferential tests; the report contains the
condition-wise brain-mean statistics and explicit caveats.

Added one shared 20-condition axis key for the subject-02 ABCD model-RDM
figures, listing each task-half label and its executed versus instructed
A--B--C--D grid locations. This was read from the existing pairing metadata
and `rewDSR` model vectors; no RSA maps or statistics were regenerated.

## 2026-09-14 — subject contribution and task-order audit of the instruction shift

Added `scripts/diagnose_instruction_subject_order.py`. The lightweight audit
uses the already merged raw 4-D maps and the actual behavioural task order for
all 33 subjects. It records each subject's spatial mean beta across the nine
resolved conditions and the exact change in the mean voxelwise group-t shift
when that subject is omitted. It also correlates the instruction-model
dissimilarity with temporal proximity (`exp(-lag/tau)`) for condition pairs;
cross-acquisition-part pairs have zero proximity. Positive values of that
order metric are the direction expected to bias an RSA coefficient negative
if temporally nearby patterns are spuriously similar.

For single-model `ABCD_rew_instr_within`, 20/33 subject means were negative.
The mean voxelwise group-t statistic across the nine conditions was -0.833;
it remained negative in every leave-one-subject-out sample (range -1.024 to
-0.704). Largest negative contributors were sub-27, sub-33, sub-02 and
sub-23, so no single subject generated the shift. The same subjects strongly
tracked the concurrent-model instruction coefficient (subject-mean Pearson
r=0.970).

There was no evidence that actual task order explained the subject shift. The
primary instruction-dissimilarity/temporal-proximity proxy correlated
Spearman rho=0.114 with the raw subject mean (two-sided permutation p=.516)
and rho=-0.086 with exact LOO negative-t contribution (p=.628). Initial
forward/backward direction (p=.784) and initial task set (p=.987) were also
unrelated. Sub-01 was the only subject whose matched task sets were mixed
between acquisition parts; its mean beta was positive and therefore attenuated
rather than caused the negative shift.

Successive instruction epochs were separated by 164.7 s on average, and
matched forward/backward task instructions by 614.5 s on average among the 32
subjects with within-part matched pairs. Thus this audit addresses order and
slow temporal structure, not short-lag HRF overlap. It cannot test whether
participants differ in residual autocorrelation strength because the current
event-locked FEAT design matrices and residuals are not present locally. The
merged NIfTIs also lack a subject-ID manifest, so their fourth-dimension order
is explicitly recorded as the standard 33-subject merge-order assumption.
Outputs and settings are in
`data/derivatives/group/instruction_memory_subject_order_audit_2026-09-14/`.

## 2026-09-14 — sEEG behaviour: all attempts + task stages

`scripts/behaviour_summary.py` extended on the cell-data (sEEG) side; fMRI
side unchanged. New helper `scripts/extract_uncovers_ephys.py` pulls
per-attempt uncover counts out of `abcd_data_08-Sep-2025.mat`
(`pressed_to_uncover == 1 & correct_uncover == 0`) into
`ephys_humans/derivatives/group/ephys_uncovers_per_attempt.csv`.

**Two levels.** Every cell measure now exists for all attempts
(`rep_overall` axis) and for error-free repeats only (`rep_correct` axis,
the subset the ABCD-code analyses use). n = 63 sessions, 18228 attempts,
1489 grids.
- all attempts: A→D 6.51 ± 2.76 s, whole attempt 9.15 ± 3.98 s,
  90.8 ± 3.6 % shortest walks, slope −0.59 s/attempt (t(62) = −14.4,
  p = 5.4e-20)
- correct only: A→D 4.43 ± 1.74 s, 97.4 ± 2.2 % shortest walks,
  slope −0.006 s/repeat (t(62) = −0.65, p = 0.52) — i.e. essentially all
  the speed-up happens before the first error-free loop.

**Stages**, defined by available information, not by performance:
explore = first attempt of a grid; learn = further attempts up to and
including the first error-free loop; execute = attempts after it.
7/1489 grids never reached an error-free loop.

| measure | explore | learn | execute | RM-ANOVA | learn−execute |
|---|---|---|---|---|---|
| attempt time [s] | 27.8 | 12.2 | 6.6 | F(2,124)=209, p=1.7e-40, ηp²=.77 | t(62)=9.1, p_holm=4.5e-13, dz=1.15 |
| A→D time [s] | 20.6 | 7.9 | 4.7 | F=270, p=7.0e-46, ηp²=.81 | t=6.9, p_holm=3.6e-09, dz=0.87 |
| time to A [s] | 7.2 | 4.3 | 1.9 | F=48, p=3.7e-16, ηp²=.44 | t=12.1, p_holm=1.5e-17, dz=1.53 |
| error fraction | 1.00 | 0.29 | 0.07 | F=1259, p=4.3e-83, ηp²=.95 | t=9.9, p_holm=2.1e-14, dz=1.25 |
| wrong uncovers/attempt | 13.7 | 1.84 | 0.17 | F=1249, p=7.0e-83, ηp²=.95 | t=7.1, p_holm=1.2e-09, dz=0.90 |
| % shortest walks | 31.1 | 86.8 | 96.3 | F=1151, p=8.6e-81, ηp²=.95 | t=−7.3, p_holm=6.0e-10, dz=−0.92 |
| extra steps/walk | 3.44 | 0.66 | 0.11 | F=485, p=2.3e-59, ηp²=.89 | t=7.3, p_holm=7.4e-10, dz=0.92 |

CAVEATS, logged so they are not over-claimed: `error_fraction` is ~1 in
explore by construction (the first attempt on a covered grid cannot be
error-free), and explore-stage uncovers are search, not memory failures.
The clean contrast is learn vs. execute — both stages where all four
rewards have already been seen — and it is significant for every measure.

**Learning speed** (per session, mean ± sd): explore duration
27.8 ± 12.8 s; 2.55 ± 0.58 attempts and 48.8 ± 27.5 s to the first
error-free loop; 17.5 ± 6.9 wrong uncovers before it. Across the ten
correct repeats loop time falls only 4.65 → 4.38 s (4.4 ± 9.3 %,
t(62) = 3.77, p = 3.7e-4).

**Data-quality finding.** 12 of 18228 attempts are flagged
`trial_correct = 1` although the move record contains an incorrect uncover.
Those 12 also incremented the correct-repeat counter, which fully explains
the previously unexplained `rep_correct == 10` overflow (same 9 sessions,
same 12 trials). The flag is left untouched; the discrepant attempts are
listed in `derivatives/group/ephys_uncover_flag_discrepancies.csv`.

New 2.5 × 2.5 cm panels (pdf + png) under `plots/`: `ephys_stage_*`
(6 measures + colour legend, phase ramp pastel-pink → bordeaux),
`ephys_loop_time_by_attempt_all`, `ephys_incorrect_uncovers_by_attempt`,
`ephys_attempts_to_criterion`, `ephys_speedup_percent`,
`ephys_explore_duration`. New tables: `ephys_per_stage.csv`,
`ephys_learning.csv`; `ephys_attempts.csv` gains stage / run_index /
uncover columns; `ephys_shortest_paths.csv` now covers all attempts.

### Addendum — time spent per stage

Added the missing "how much time does each stage take" read-out. Per-grid
stage durations partition the time on task (attempt starts are contiguous),
so shares sum to 100 %.

| | explore | learn | execute | RM-ANOVA | learn−execute |
|---|---|---|---|---|---|
| time per grid [s] | 27.8 | 21.1 | 64.7 | F(2,124)=171, p=2.0e-36, ηp²=.73 | t(62)=−15.4, p_holm=2.5e-22, dz=−1.93 |
| % of time on task | 25.3 | 16.7 | 58.0 | F(2,124)=408, p=2.8e-55, ηp²=.87 | t(62)=−22.4, p_holm=4.2e-31, dz=−2.82 |

Note the reversal against the per-attempt measures: execute is by far the
fastest stage per attempt (6.6 s vs 27.8 s) yet consumes most of the time on
task, because it contains ~9 of ~12 attempts per grid. Explore is a single
attempt and still takes a quarter of the time on task.

New panels: `ephys_stage_time_spent`, `ephys_stage_time_share`,
`ephys_stage_time_stacked` (single stacked bar, 25/17/58 %). `duration_per_grid`,
`total_duration` and `duration_percent` added to `ephys_per_stage.csv` and to
the stage tests.

### Bugfix — NaN loop times poisoned the sEEG learning slopes

Ten attempts across the dataset have no `t_D` (recording stopped
mid-attempt). Passing them to `scipy.stats.linregress` returned NaN for the
whole session, so two sessions (s02, s28 — which have 184 and 247 correct
trials each) were silently dropped from the correct-repeat slope, and eight
more from the all-attempt slope. New helper `_loop_time_slope` drops
non-finite pairs first. All 63 sessions now contribute.

- correct-repeat slope: was −0.006 s/repeat, t(60) = −0.65, p = 0.52 (n = 61)
  → now −0.008 s/repeat, t(62) = −0.97, p = 0.34 (n = 63). Still null.
- all-attempt slope: now −0.629 s/attempt, t(62) = −15.9, p = 1.4e-23 (n = 63)

This also corrects a methods claim: the two sessions missing from the slope
analysis were NOT sessions with "fewer than two correct trials".

Other facts checked against the draft methods while reviewing it:
- fMRI: 33 subjects (35 scanned, sub-21/sub-29 excluded), 10 layouts, 5
  repeats each, mean 2.599 steps per subpath — all as written.
- The fMRI gamma jitter (`3x3_fMRI_part1.py: jitter()`) draws the *subpath*
  duration (truncated 3–12 s, shape 5.75), then divides it across the steps
  of that subpath plus the reward wait. It is not a per-step draw.
- Runs per reward-layout within a session range 1–5, not 1–4
  (85/258/225/52/1 layouts at 1/2/3/4/5 runs).
- `fixed_grids` is set in 28 of 63 sessions, not 26.
- The single excluded s23 attempt is a hand-identified 314.6 s interruption,
  not a 3-SD rule: 357 attempts (82 correct ones) exceed mean + 3 SD and are
  retained.
