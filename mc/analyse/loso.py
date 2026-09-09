"""
Per-TR group inference for the instruction-phase RSA.

One implementation of three things, used by `scripts/per_TR_loso.py`:

  1. SVC peak test -- small-volume corrected max-t sign-flip permutation
     (Nichols & Holmes 2002) over all voxels in an a-priori mask AND all
     included TRs jointly, so one null corrects the spatial and the temporal
     search at once. This is the test behind the reported instruction-phase
     result.
  2. LOSO cross-validated timecourse -- top-k voxels chosen on n-1 subjects,
     read out in the held-out subject, repeated for every subject. Removes the
     circularity of selecting and averaging on the same data, so the timecourse
     is an unbiased effect-size estimate.
  3. Whole-brain volumes -- t, FWE p and uncorrected p, for visualisation.

Everything empirical and every permutation goes through `tstat` /
`null_max_t` here, so the two can never drift apart (CLAUDE.md rule 4).
`perm_wholebrain` asserts that its permutation arithmetic reproduces `tstat`
on the identity sign-flip, making that guarantee checkable rather than assumed.

Input layout, one folder per TR:
    {root}/{fmt_level(dir_pattern, level)}/
        cropped_masked_smooth_fwhm5_{model}_beta_std.nii[.gz]   (X,Y,Z,n_subj)
        mask_all_32_subjects.nii[.gz]                           (X,Y,Z)

NOTE ON df: with 32 subjects the one-sample t has df = 31. `tstat` divides by
the sample SD with ddof=1 and multiplies by sqrt(n).
"""
import csv
import json
import os
import re
import sys

import numpy as np
import nibabel as nib

FILE_RE = re.compile(r"^cropped_masked_smooth_fwhm5_(.+)_beta_std\.nii(\.gz)?$")
BETA_STEM = "cropped_masked_smooth_fwhm5_{model}_beta_std.nii"
GROUP_MASK_NAME = "mask_all_32_subjects.nii"

# ------------------------------------------------------- the third axis ----
# Beta maps are stacked along one extra axis whose levels are either the TRs of
# the instruction period (ordered seconds -- the original per-TR analysis) or a
# set of named conditions (the instruction-epoch GLMs: 'instr_see-A-first',
# ...). Everything below treats that axis as a plain list of LABELS and never
# assumes they are numbers. Only the plotting cares which kind it is, because
# a line drawn through unordered conditions asserts an ordering that does not
# exist. The statistics are identical either way: max-t over voxels x levels.
AXIS_KINDS = ("tr", "condition")


def axis_is_numeric(levels):
    """True if the axis levels are numbers, i.e. an ordered (TR) axis."""
    return all(isinstance(v, (int, np.integer)) for v in levels)


def fmt_level(pattern, level):
    """Substitute one axis level into a directory pattern. `{tr}` stays the
    placeholder name so every existing pattern keeps working; `{condition}`
    and `{level}` are accepted too so a condition-wise pattern reads
    naturally."""
    return pattern.format(tr=level, condition=level, level=level)


def _lab(v):
    """One axis label, JSON-safe (numpy ints do not serialise)."""
    return int(v) if isinstance(v, (int, np.integer)) else str(v)


def level_labels(levels):
    return [_lab(v) for v in levels]


# --------------------------------------------------------------- inputs ----
def resolve_nii(path):
    """Accept a .nii path and fall back to its .nii.gz twin (or vice versa).

    Older per-TR group folders store uncompressed .nii, newer ones .nii.gz;
    callers name the file the same way regardless."""
    if os.path.exists(path):
        return path
    alt = path[:-3] if path.endswith(".gz") else path + ".gz"
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"neither {path} nor {alt} exists")


def load_ref(root, dir_pattern, trs):
    """Reference image + brain mask = voxels present in all subjects at every
    included TR that ships one.

    The per-TR group masks differ by a handful of voxels, and a voxel entering
    a max-t search over voxels x TRs has to be valid at every TR in that
    search, so this intersects rather than taking TR0's alone.

    Older group folders only wrote `mask_all_32_subjects` for some TRs (the
    original `instruction_per_TR` run has it for TR0 and TR3 only, which is why
    the script that produced the reported number read TR0's and stopped). TRs
    without one are skipped and the count actually used is printed, so the
    search volume is never silently wrong -- it just falls back to whatever
    masks exist."""
    ref, brain, used = None, None, []
    for tr in trs:
        try:
            p = resolve_nii(os.path.join(root, fmt_level(dir_pattern, tr),
                                         GROUP_MASK_NAME))
        except FileNotFoundError:
            continue
        img = nib.load(p)
        m = img.get_fdata() > 0
        ref, brain = (img, m) if ref is None else (ref, brain & m)
        used.append(tr)
    if ref is None:
        raise FileNotFoundError(
            f"no {GROUP_MASK_NAME} found in any of {list(trs)} under "
            f"{os.path.join(root, dir_pattern)}")
    if len(used) < len(trs):
        print(f"[warn] group mask present for only {len(used)}/{len(trs)} level(s) "
              f"{used}; brain mask is the intersection of those",
              file=sys.stderr)
    print(f"[info] brain mask: {int(brain.sum())} voxels "
          f"(intersection over {len(used)} level mask(s))", file=sys.stderr)
    return ref, brain


def load_mask(mask_path, ref):
    """Binary mask on the reference grid, resampling nearest-neighbour if the
    mask ships on a different grid."""
    img = nib.load(mask_path)
    same_grid = (img.shape[:3] == ref.shape[:3] and
                 np.allclose(img.affine, ref.affine, atol=1e-3))
    if same_grid:
        return img.get_fdata() > 0
    from nilearn.image import resample_img
    print(f"[info] mask grid differs from data grid -- resampling "
          f"(nearest-neighbour) {img.shape[:3]} -> {ref.shape[:3]}", file=sys.stderr)
    return resample_img(img, target_affine=ref.affine, target_shape=ref.shape,
                        interpolation="nearest").get_fdata() > 0


def load_masks(specs, ref, brain):
    """`['name=path', ...]` -> {name: {path, bool, n_vox}}, each & brain."""
    masks = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        m = load_mask(path, ref) & brain
        masks[name] = dict(path=path, bool=m, n_vox=int(m.sum()))
        print(f"[mask] {name:8s} {m.sum():6d} in-brain voxels  ({path})", file=sys.stderr)
    return masks


def discover_models(root, dir_pattern, tr):
    """Every `{model}` with a beta map in this TR's folder, sorted."""
    d = os.path.join(root, fmt_level(dir_pattern, tr))
    return [m.group(1) for m in
            (FILE_RE.match(f) for f in sorted(os.listdir(d))) if m]


def _expected_uncompressed_size(path):
    """Bytes a complete NIfTI should occupy, taken from its own header."""
    hdr = nib.load(path).header
    n_vox = int(np.prod(hdr.get_data_shape()))
    return int(hdr.get_data_offset()) + n_vox * int(hdr.get_data_dtype().itemsize)


def _is_truncated(path):
    """True if `path` is a short or unreadable nifti.

    A gzip trailer records the uncompressed size, so comparing it against what
    the nifti header implies flags a cut-off file in constant time -- a
    truncated .gz still has a perfectly readable header, which is exactly why
    it survives until `get_fdata`. Only the files that fail that cheap test are
    confirmed by really decompressing, so a multi-member gzip (whose trailer
    describes its last member alone) cannot raise a false alarm."""
    try:
        want = _expected_uncompressed_size(path)
    except Exception:
        return True
    if not path.endswith(".gz"):
        return os.path.getsize(path) < want
    with open(path, "rb") as fh:
        fh.seek(-4, os.SEEK_END)
        isize = int(np.frombuffer(fh.read(4), dtype="<u4")[0])
    if isize == want % 2 ** 32:
        return False
    import gzip
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return False
    except Exception:
        return True


def check_inputs(root, dir_pattern, models, trs):
    """Fail fast on missing or truncated inputs, before hours of work.

    An interrupted transfer leaves niftis that open fine and read their header
    fine, and only die deep inside `get_fdata` -- so the run crashes partway,
    after it has already paid for whichever models happened to be intact. This
    costs a second and turns that into a message before anything starts."""
    bad = []
    for model in models:
        for tr in trs:
            p = os.path.join(root, fmt_level(dir_pattern, tr),
                             BETA_STEM.format(model=model))
            try:
                f = resolve_nii(p)
            except FileNotFoundError:
                bad.append(("missing", p))
                continue
            if _is_truncated(f):
                bad.append(("truncated", f))
    n = len(models) * len(trs)
    if bad:
        print(f"[error] {len(bad)} of {n} input files unusable:", file=sys.stderr)
        for why, p in bad:
            print(f"  {why:9s} {p}", file=sys.stderr)
        raise RuntimeError(
            f"{len(bad)} of {n} input files are missing or truncated; "
            "re-transfer them (or pass --skip-input-check to proceed anyway)")
    print(f"[info] input check: all {n} beta maps complete", file=sys.stderr)


def _load_with_retry(path, retries=4, wait=10):
    """nib.load, retried on transient failure.

    These group folders live on sync-backed storage, where a file can briefly
    read as unidentifiable or truncated while it is intact on disk before and
    after. That killed a 3-hour run once; a few seconds of patience is cheaper
    than restarting. A genuinely corrupt file still raises, after `retries`."""
    import time as _time
    for attempt in range(retries):
        try:
            return nib.load(path).get_fdata()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"[warn] read failed ({type(exc).__name__}: {exc}); retry "
                  f"{attempt + 1}/{retries - 1} in {wait}s -- {path}",
                  file=sys.stderr, flush=True)
            _time.sleep(wait)


def read_model_columns(root, dir_pattern, model, trs, ijk):
    """(n_subj, n_vox, n_tr) for the voxels `ijk`, one pass over the TR niftis.

    Reading is the expensive part, so callers extract the widest voxel set they
    need once and take column subsets of it for individual masks."""
    ix, iy, iz = ijk
    out = None
    for j, tr in enumerate(trs):
        f = resolve_nii(os.path.join(root, fmt_level(dir_pattern, tr),
                                     BETA_STEM.format(model=model)))
        data = _load_with_retry(f)
        if data.ndim == 3:
            data = data[..., None]
        if out is None:
            out = np.empty((data.shape[-1], len(ix), len(trs)), dtype=np.float32)
        out[:, :, j] = data[ix, iy, iz, :].T
        del data
    return out


# ----------------------------------------------------------- statistics ----
def tstat(A):
    """A: (n_subj, ...) -> per-column one-sample t against 0."""
    n = A.shape[0]
    m = A.mean(0)
    s = A.std(0, ddof=1)
    return np.where(s > 0, m / np.where(s > 0, s, 1) * np.sqrt(n), 0.0)


def adaptive_pblock(n_cols, max_elements=2e7):
    """Permutations per block, so one block stays near `max_elements` floats."""
    return int(max(1, min(1000, max_elements // max(n_cols, 1))))


def null_max_t(A, n_perm, seed, pblock=1000):
    """Sign-flip max-t null over all columns of A (n_subj, n_cols).

    Under H0 the sign of a subject's whole map is arbitrary. Sign-flipping
    preserves sum(x^2), so the per-permutation t follows from
    var = (S2 - n*M^2)/(n-1) without recomputing anything per permutation.
    Chunked over permutations to bound memory."""
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    S2 = (A ** 2).sum(0)
    out = np.empty(n_perm)
    for s0 in range(0, n_perm, pblock):
        b = min(pblock, n_perm - s0)
        F = rng.choice([-1.0, 1.0], size=(b, n))
        M = (F @ A) / n
        var = (S2[None, :] - n * M ** 2) / (n - 1)
        T = np.where(var > 0, M * np.sqrt(n) / np.sqrt(np.where(var > 0, var, 1)), 0.0)
        out[s0:s0 + b] = T.max(1)
    return out


def voxel_fwe_p(T, nmt):
    """Voxel-wise FWE p from the max-t null: fraction of permutations whose
    max t is >= this voxel's observed t."""
    srt = np.sort(nmt)
    return ((len(srt) - np.searchsorted(srt, T, side="left")) / len(srt)).astype(np.float32)


def perm_wholebrain(A, T_obs, n_perm, seed, pblock, want_neg=False):
    """Sign-flip permutation keeping BOTH the max-t null (for FWE) and a
    per-column exceedance tally (for uncorrected p).

    Only the tallies are kept, never the (n_perm, n_vox) null, so memory is set
    by `pblock` and not by `n_perm`. The identity flip is asserted to reproduce
    the observed t, so the permutation statistic and `tstat` are verifiably the
    same statistic.

    Returns (null_max_t, count_ge, count_le); count_le is None unless want_neg."""
    n, n_cols = A.shape
    S2 = (A ** 2).sum(0)

    def _t_from_flips(F):
        M = (F @ A) / n
        var = (S2[None, :] - n * M ** 2) / (n - 1)
        return np.where(var > 0, M * np.sqrt(n) / np.sqrt(np.where(var > 0, var, 1)), 0.0)

    assert np.allclose(_t_from_flips(np.ones((1, n)))[0], T_obs, atol=1e-4), \
        "permutation t and empirical tstat disagree on the identity sign-flip"

    rng = np.random.RandomState(seed)
    null_max = np.empty(n_perm)
    cnt_ge = np.zeros(n_cols, dtype=np.int32)
    cnt_le = np.zeros(n_cols, dtype=np.int32) if want_neg else None
    for s0 in range(0, n_perm, pblock):
        b = min(pblock, n_perm - s0)
        T = _t_from_flips(rng.choice([-1.0, 1.0], size=(b, n)))
        null_max[s0:s0 + b] = T.max(1)
        cnt_ge += (T >= T_obs[None, :]).sum(0, dtype=np.int32)
        if want_neg:
            cnt_le += (T <= T_obs[None, :]).sum(0, dtype=np.int32)
    return null_max, cnt_ge, cnt_le


# ---------------------------------------------------------------- tests ----
def run_svc(D, ref, ijk, n_perm, seed, trs=None):
    """SVC peak test in one mask, both signs.

    `trs` are the real TR numbers behind D's third axis; peaks are reported in
    those, not in the position along the axis, so a run over a subset of TRs
    still names the TR it actually found.

    The negative peak reuses the SAME null -- sign-flipping makes the null of
    max(-t) identical to the null of max(t) -- as a second one-sided test. It
    is NOT corrected for having looked in both directions; treat the positive
    column as the test and the negative one as descriptive.

    Returns (Tobs, null_max_t, summary_dict)."""
    n, n_vox, n_tr = D.shape
    trs = list(range(n_tr)) if trs is None else list(trs)
    Tobs = tstat(D)
    flat = D.reshape(n, -1)
    nmt = null_max_t(flat, n_perm=n_perm, seed=seed,
                     pblock=adaptive_pblock(flat.shape[1]))
    tcrit = float(np.percentile(nmt, 95))
    ix, iy, iz = ijk

    def _peak(T):
        pk = np.unravel_index(np.argmax(T), T.shape)
        mni = nib.affines.apply_affine(ref.affine, [ix[pk[0]], iy[pk[0]], iz[pk[0]]])
        return pk, float(T[pk]), [int(round(float(v))) for v in mni]

    pk_p, t_p, mni_p = _peak(Tobs)
    pk_n, t_n, mni_n = _peak(-Tobs)
    return Tobs, nmt, dict(
        n_vox=int(n_vox), n_tr=int(n_tr), n_subj=int(n),
        peak_t=t_p, peak_p_FWE=float((nmt >= t_p).mean()),
        peak_TR=_lab(trs[pk_p[1]]), peak_mni=mni_p,
        peak_t_neg=float(-t_n), peak_p_FWE_neg=float((nmt >= t_n).mean()),
        peak_TR_neg=_lab(trs[pk_n[1]]), peak_mni_neg=mni_n,
        t_crit_FWE05=tcrit,
        n_supra_FWE05=int((Tobs >= tcrit).sum()),
        n_supra_FWE05_neg=int((-Tobs >= tcrit).sum()),
        trs=level_labels(trs),
        peak_voxel_t_by_TR=[float(v) for v in Tobs[pk_p[0], :]],
        max_t_by_TR=[float(v) for v in Tobs.max(0)],
        n_perm=int(n_perm), seed=int(seed))


def run_loso(D, k_values, n_perm, seed, trs=None):
    """Leave-one-subject-out cross-validated timecourse, per selection size k.

    Voxels are ranked by their by-TR t on n-1 subjects and the held-out
    subject's mean beta in the top k is read out, so nothing is selected and
    averaged on the same data. The per-TR p is FWE-corrected over the TR axis
    by the same sign-flip max-t null.

    Returns (results_dict, {k: held_out_array (n_subj, n_tr)})."""
    n, n_vox, n_tr = D.shape
    trs = list(range(n_tr)) if trs is None else list(trs)
    out, held_by_k = {}, {}
    for k in k_values:
        kk = min(k, n_vox)
        held = np.zeros((n, n_tr))
        for s in range(n):
            train = np.delete(np.arange(n), s)
            Ttr = tstat(D[train])
            top = np.argsort(-Ttr.max(1))[:kk]
            held[s] = D[s, top, :].mean(0)
        t = tstat(held)
        nm = null_max_t(held, n_perm=n_perm, seed=seed + 100 + kk)
        out[str(kk)] = dict(
            k=int(kk), trs=level_labels(trs),
            mean=[float(v) for v in held.mean(0)],
            sem=[float(v) for v in held.std(0, ddof=1) / np.sqrt(n)],
            t=[float(v) for v in t],
            p_FWE=[float((nm >= v).mean()) for v in t],
            t_crit_FWE05=float(np.percentile(nm, 95)))
        held_by_k[kk] = held
    return out, held_by_k


def run_wholebrain(D, ref, ijk, n_perm, seed, want_neg=False, trs=None):
    """Whole-brain t + FWE p + uncorrected p, over voxels x TRs jointly.

    FWE corrects over the ENTIRE brain mask and all included TRs at once, so it
    is much stricter than the small-volume p from `run_svc` and the two are not
    comparable. Uncorrected p is that voxel-and-TR's own permutation p, for
    visualisation only."""
    n, n_vox, n_tr = D.shape
    trs = list(range(n_tr)) if trs is None else list(trs)
    Tobs = tstat(D)
    A = D.reshape(n, -1)
    nmt, cnt_ge, cnt_le = perm_wholebrain(
        A, Tobs.reshape(-1), n_perm=n_perm, seed=seed,
        pblock=adaptive_pblock(A.shape[1], max_elements=1e7), want_neg=want_neg)

    p_unc = (cnt_ge / n_perm).reshape(n_vox, n_tr).astype(np.float32)
    p_fwe = voxel_fwe_p(Tobs, nmt)
    ix, iy, iz = ijk
    pk = np.unravel_index(np.argmax(Tobs), Tobs.shape)
    mni = nib.affines.apply_affine(ref.affine, [ix[pk[0]], iy[pk[0]], iz[pk[0]]])
    summary = dict(
        n_vox=int(n_vox), n_tr=int(n_tr), n_subj=int(n), n_perm=int(n_perm),
        seed=int(seed), peak_t=float(Tobs[pk]), peak_TR=_lab(trs[pk[1]]),
        trs=level_labels(trs),
        peak_mni=[int(round(float(v))) for v in mni],
        peak_p_FWE=float((nmt >= Tobs[pk]).mean()),
        t_crit_FWE05=float(np.percentile(nmt, 95)),
        # counted off p_fwe so it always equals thresholding the saved
        # _1minusFWEp map at 0.95 (the t_crit percentile and the exceedance
        # proportion drift apart at low n_perm)
        n_p_FWE_lt_05=int((p_fwe < 0.05).sum()),
        n_p_unc_lt_001=int((p_unc <= 0.001).sum()),
        max_t_by_TR=[float(v) for v in Tobs.max(0)])
    p_unc_neg = (cnt_le / n_perm).reshape(n_vox, n_tr).astype(np.float32) if want_neg else None
    return Tobs, nmt, p_fwe, p_unc, p_unc_neg, summary


# -------------------------------------------------------------- volumes ----
def vol_from_cols(vals, fill, ref, ijk, n_tr):
    """Scatter a (n_vox, n_tr) column array back into image space.

    3-D when a single TR was analysed, 4-D (X, Y, Z, TR) when several were, so
    the fsleyes TR slider scrubs through the instruction period."""
    ix, iy, iz = ijk
    v = np.full(ref.shape[:3] + (n_tr,), fill, dtype=np.float32)
    v[ix, iy, iz, :] = vals
    return v[..., 0] if n_tr == 1 else v


def write_mask_maps(out_dir, model, Tobs, nmt, ref, ijk, n_tr):
    """t-map and voxel-wise FWE maps inside one mask, both signs. Outside the
    mask: t = 0, p = 1 (1-p = 0). Threshold `_voxel1minusFWEp` at 0.95."""
    p_pos = voxel_fwe_p(Tobs, nmt)
    p_neg = voxel_fwe_p(-Tobs, nmt)
    hdr = ref.header.copy()
    for suffix, vals, fill in (("t", Tobs, 0.0),
                               ("voxelFWEp", p_pos, 1.0),
                               ("voxel1minusFWEp", 1.0 - p_pos, 0.0),
                               ("voxel1minusFWEp_neg", 1.0 - p_neg, 0.0)):
        nib.save(nib.Nifti1Image(vol_from_cols(vals, fill, ref, ijk, n_tr),
                                 ref.affine, hdr),
                 os.path.join(out_dir, f"{model}_{suffix}.nii.gz"))
    np.save(os.path.join(out_dir, f"{model}_null_max_t.npy"), nmt.astype(np.float32))


def write_wholebrain_maps(out_dir, model, Tobs, p_fwe, p_unc, p_unc_neg, nmt,
                          ref, ijk, n_tr):
    """Whole-brain volumes: t, and both p images in the FSL 1-p convention
    (threshold 0.95 for p < .05, 0.999 for p < .001).

    A voxel no permutation ever beat gets p = 0 (1-p = 1); read that as
    p < 1/n_perm, the resolution limit, not as a real zero."""
    hdr = ref.header.copy()
    vols = [("t", Tobs, 0.0),
            ("1minusFWEp", 1.0 - p_fwe, 0.0),
            ("1minusp_uncorr", 1.0 - p_unc, 0.0)]
    if p_unc_neg is not None:
        vols += [("1minusFWEp_neg", 1.0 - voxel_fwe_p(-Tobs, nmt), 0.0),
                 ("1minusp_uncorr_neg", 1.0 - p_unc_neg, 0.0)]
    for suffix, vals, fill in vols:
        nib.save(nib.Nifti1Image(vol_from_cols(vals, fill, ref, ijk, n_tr),
                                 ref.affine, hdr),
                 os.path.join(out_dir, f"{model}_{suffix}.nii.gz"))
    np.save(os.path.join(out_dir, f"{model}_null_max_t.npy"), nmt.astype(np.float32))


def write_table(path, rows):
    if not rows:
        return None
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ------------------------------------------------- reading results back ----
def summary_row(svc, loso_rec=None):
    """One `summary_table.csv` row from an SVC summary dict (+ optional LOSO)."""
    row = dict(mask=svc["mask"], model=svc["model"], n_vox=svc["n_vox"],
               peak_t=round(svc["peak_t"], 3), peak_TR=svc["peak_TR"],
               peak_mni="/".join(str(c) for c in svc["peak_mni"]),
               p_FWE=svc["peak_p_FWE"], t_crit_FWE05=round(svc["t_crit_FWE05"], 3),
               n_supra=svc["n_supra_FWE05"],
               peak_t_neg=round(svc["peak_t_neg"], 3), peak_TR_neg=svc["peak_TR_neg"],
               peak_mni_neg="/".join(str(c) for c in svc["peak_mni_neg"]),
               p_FWE_neg=svc["peak_p_FWE_neg"], n_supra_neg=svc["n_supra_FWE05_neg"])
    if loso_rec is not None:
        t = np.asarray(loso_rec["t"])
        # the LEVEL of the peak, not its position along the axis -- with a
        # subset like --trs 4,5,6 the position said 0/1/2 rather than 4/5/6.
        i = int(t.argmax())
        levels = loso_rec.get("trs") or list(range(len(t)))
        row.update(loso_k=int(loso_rec["k"]), loso_peak_t=round(float(t.max()), 3),
                   loso_peak_TR=_lab(levels[i]),
                   loso_p_FWE=loso_rec["p_FWE"][i])
    return row


def wholebrain_row(wsum):
    return dict(model=wsum["model"], n_vox=wsum["n_vox"],
                peak_t=round(wsum["peak_t"], 3), peak_TR=wsum["peak_TR"],
                peak_mni="/".join(str(c) for c in wsum["peak_mni"]),
                p_FWE=wsum["peak_p_FWE"],
                t_crit_FWE05=round(wsum["t_crit_FWE05"], 3),
                n_p_FWE_lt_05=wsum["n_p_FWE_lt_05"],
                n_p_unc_lt_001=wsum["n_p_unc_lt_001"])


def model_is_done(out_dir, model, mask_names, cv=True, wholebrain=False):
    """True if every per-model output this run would write already exists."""
    for name in mask_names:
        if not os.path.exists(os.path.join(out_dir, name, f"{model}_svc_summary.json")):
            return False
        if cv and not os.path.exists(os.path.join(out_dir, name,
                                                  f"{model}_loso_results.json")):
            return False
    if wholebrain and not os.path.exists(
            os.path.join(out_dir, "wholebrain", f"{model}_summary.json")):
        return False
    return True


def collect_rows(out_dir, mask_names, models, cv=True, wholebrain=False, k="100"):
    """Rebuild both summary tables from the per-model json already on disk.

    Used at the end of every run so the tables cover every model present,
    including ones finished by an earlier (e.g. interrupted) invocation."""
    rows, wb_rows = [], []
    for model in models:
        for name in mask_names:
            f = os.path.join(out_dir, name, f"{model}_svc_summary.json")
            if not os.path.exists(f):
                continue
            svc = json.load(open(f))
            lo = None
            if cv:
                lf = os.path.join(out_dir, name, f"{model}_loso_results.json")
                if os.path.exists(lf):
                    d = json.load(open(lf))
                    lo = d[k] if k in d else d[sorted(d, key=lambda s: int(s))[0]]
            rows.append(summary_row(svc, lo))
        if wholebrain:
            f = os.path.join(out_dir, "wholebrain", f"{model}_summary.json")
            if os.path.exists(f):
                wb_rows.append(wholebrain_row(json.load(open(f))))
    return rows, wb_rows


def result_masks(out_dir):
    """Mask sub-folders of a finished run, in the order they were written."""
    return [d for d in sorted(os.listdir(out_dir))
            if os.path.isdir(os.path.join(out_dir, d)) and d != "wholebrain"]


def load_loso(out_dir, mask, model, k="100"):
    """One model's LOSO record; falls back to the smallest k if `k` is absent."""
    lo = json.load(open(os.path.join(out_dir, mask, f"{model}_loso_results.json")))
    key = k if k in lo else sorted(lo, key=lambda s: int(s))[0]
    return lo[key]


def load_settings(out_dir):
    return json.load(open(os.path.join(out_dir, "settings.json")))


# ------------------------------------------------------------- plotting ----
# CLAUDE.md state palette -- the four reward channels ARE the A/B/C/D rewards.
# Keyed by the CURRENT split-channel names (fMRI_run_RSA_instruction.py renamed
# curr_rew -> A_rew etc.; see its LEGACY_MODEL_NAMES). Colours are the fixed
# project state colours A/B/C/D, orange -> purple.
CHANNEL_COLOURS = {"a_rew": "#F15A29", "b_rew": "#F7931E",
                   "c_rew": "#C7C6E2", "d_rew": "#6B60AA"}
CHANNEL_LABELS = {"a_rew": "reward A", "b_rew": "reward B",
                  "c_rew": "reward C", "d_rew": "reward D"}
# Result folders written before the rename still hold the old names, so they
# keep their colours instead of silently falling back to grey.
LEGACY_CHANNEL_ALIASES = {"curr_rew": "a_rew", "next_rew": "b_rew",
                          "two_next_rew": "c_rew", "three_next_rew": "d_rew"}
# From mc/latest_experiment/3x3_fMRI_part1.py `show_rewards`: ONE reward on
# screen at a time, a 1.5 s/reward first pass then a 1 s/reward refresh.
REWARD_SCHEDULE = [(0.0, 1.5, "A", "#F15A29"), (1.5, 3.0, "B", "#F7931E"),
                   (3.0, 4.5, "C", "#C7C6E2"), (4.5, 6.0, "D", "#6B60AA"),
                   (6.0, 7.0, "A", "#F15A29"), (7.0, 8.0, "B", "#F7931E"),
                   (8.0, 9.0, "C", "#C7C6E2"), (9.0, 12.0, "D", "#6B60AA")]
ROI_DISPLAY_NAMES = {"mPFC": "mPFC", "MTL": "HC / EC", "MTL_L": "HC / EC left",
                     "MTL_R": "HC / EC right", "visual": "occipital"}
OBSERVED_MARKER_COLOUR = "#0e3d3a"


def set_figure_style():
    """Arial, 9 pt axes / 11 pt titles -- sized for an eighth-of-A4 subpanel."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })


def base_channel(model):
    """Reduce a map name to its reward channel, for colouring.

    'A_REW-split_rew_DSR_combo', 'A_rew' and 'A_rew_instr' all -> 'a_rew', so a
    channel keeps one colour whether it is the execution or the instruction
    variant and whether it stands alone or sits inside a combo. Pre-rename
    names ('curr_rew') map onto the same channel."""
    stem = model.split("-")[0].lower()
    if stem.endswith("_instr"):
        stem = stem[:-len("_instr")]
    return LEGACY_CHANNEL_ALIASES.get(stem, stem)


def _draw_schedule(ax, x_max=12):
    from matplotlib import pyplot as plt
    y0, y1 = ax.get_ylim()
    h = (y1 - y0) * 0.07
    for t0, t1, label, col in REWARD_SCHEDULE:
        if t0 >= x_max:
            continue
        t1 = min(t1, x_max)
        ax.add_patch(plt.Rectangle((t0 - 0.5, y1), t1 - t0, h, facecolor=col,
                                   edgecolor="white", lw=0.5, clip_on=False))
        ax.text((t0 + t1) / 2 - 0.5, y1 + h / 2, label, ha="center", va="center",
                fontsize=7, color="#333333", clip_on=False)
    ax.set_ylim(y0, y1)


def plot_per_TR_timecourses(out_dir, models, masks=None, k="100",
                            out_name="per_TR_timecourses", show=True,
                            axis_kind="tr"):
    """LOSO held-out beta along the third axis, one panel per mask.

    Plots exactly the values `run_loso` wrote -- nothing is refitted, rescaled
    or smoothed. Levels with p_FWE < .05 are ringed. Returns the per-model
    peak rows.

    `axis_kind` decides how that axis is drawn, and the difference is not
    cosmetic:
      'tr'         ordered seconds of the instruction period. Points joined by
                   a line, reward-reveal schedule along the top -- a timecourse.
      'condition'  named conditions (the instruction-epoch GLMs). Points only,
                   no joining line and no schedule: the conditions have no
                   ordering, so a line between them would assert a progression
                   the design does not contain.
    A non-numeric axis is always drawn as 'condition' whatever is asked for."""
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    set_figure_style()
    masks = masks or result_masks(out_dir)

    # Levels come from the first record read; the axis kind is downgraded to
    # categorical if they are not numbers, so a mislabelled call cannot draw a
    # timecourse through condition names.
    _probe = load_loso(out_dir, masks[0], models[0], k)
    _levels = _probe.get("trs") or list(range(len(_probe["mean"])))
    categorical = (axis_kind == "condition") or not axis_is_numeric(_levels)
    width = (3.4 if categorical else 3.0) * len(masks)
    fig, axes = plt.subplots(1, len(masks), figsize=(width, 3.9 if categorical else 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    peak_rows = []
    for ax, mask in zip(axes, masks):
        for mi, model in enumerate(models):
            rec = load_loso(out_dir, mask, model, k)
            mean = np.asarray(rec["mean"]); sem = np.asarray(rec["sem"])
            p = np.asarray(rec["p_FWE"])
            levels = list(rec.get("trs") or range(len(mean)))
            col = CHANNEL_COLOURS.get(base_channel(model), "#666666")
            if categorical:
                # Position along the axis; the labels go on the ticks. Models
                # are nudged apart horizontally so overlapping points stay
                # readable -- the axis is categorical, so the offset carries no
                # meaning and no value is altered.
                dodge = (mi - (len(models) - 1) / 2) * 0.16
                x = np.arange(len(mean)) + dodge
                ax.errorbar(x, mean, yerr=sem, fmt="o", color=col, ms=3.5,
                            lw=0, elinewidth=1.1, capsize=2)
            else:
                x = np.asarray(levels)
                ax.plot(x, mean, "-o", color=col, ms=3, lw=1.4)
                ax.fill_between(x, mean - sem, mean + sem, color=col,
                                alpha=0.18, lw=0)
            sig = p < 0.05
            if sig.any():
                ax.plot(x[sig], mean[sig], "o", color=col, ms=6.5,
                        mec=OBSERVED_MARKER_COLOUR, mew=1.0, zorder=5)
            t = np.asarray(rec["t"])
            peak_rows.append(dict(mask=mask, model=model, k=int(rec["k"]),
                                  peak_TR=_lab(levels[int(np.argmax(t))]),
                                  peak_t=round(float(t.max()), 3),
                                  p_at_peak=float(p[int(np.argmax(t))]),
                                  n_sig_TR=int(sig.sum())))
        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.set_title(ROI_DISPLAY_NAMES.get(mask, mask), pad=14)
        if categorical:
            ax.set_xlabel("")
            ax.set_xticks(np.arange(len(levels)))
            ax.set_xticklabels([str(v).replace("instr_", "") for v in levels],
                               rotation=45, ha="right")
            ax.set_xlim(-0.5, len(levels) - 0.5)
        else:
            ax.set_xlabel("instruction period (s)")
            ax.set_xticks([v for v in x if v % 2 == 0])
            _draw_schedule(ax, x_max=int(x.max()) + 1)
    axes[0].set_ylabel("held-out beta\n(LOSO cross-validated)")
    handles = [Line2D([], [], color=c, marker="o", ms=3, lw=1.4,
                      label=CHANNEL_LABELS[n]) for n, c in CHANNEL_COLOURS.items()]
    handles.append(Line2D([], [], color="none", marker="o", ms=6.5,
                          mec=OBSERVED_MARKER_COLOUR, mew=1.0, label="p$_{FWE}$ < .05"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Reward-channel representation "
                 + ("per instruction epoch" if categorical
                    else "across the instruction period"),
                 y=1.13, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "jpeg"):
        fig.savefig(os.path.join(out_dir, f"{out_name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    write_table(os.path.join(out_dir, f"{out_name}_peaks.csv"), peak_rows)
    print(f"-> {out_dir}/{out_name}.pdf  (+ .jpeg, _peaks.csv)")
    if show:
        plt.show(block=False)
    return peak_rows
