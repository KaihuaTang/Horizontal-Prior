"""
Paired stratified bootstrap 95% CI between TWO methods on per-sample CSV outputs.

The two CSVs are expected to come from `test.py` / `test_hl.py` (the modified
versions that dump `per_sample_rotate_{lo}_{hi}.csv`). Rows are joined on the
key `(dataset, item_id, rotate_range_lo, rotate_range_hi)` so that paired
differences are taken on the SAME test image evaluated by both methods.

For each requested metric, the script reports:
  * mean_A [95% CI]                 (bootstrap CI for method A's mean)
  * mean_B [95% CI]                 (bootstrap CI for method B's mean)
  * Δ = mean_A − mean_B [95% CI]    (paired bootstrap CI for the difference)
  * paired bootstrap p-value        (two-sided, centered)
  * Cohen's d (effect size)         (paired |Δ|/std-of-paired-diff)
  * Significance verdict at α=0.05

Usage examples
--------------
# Compare baseline vs. IDCue-Constraint at the Tipping setting:
python paired_bootstrap.py \\
    --csv_a exp/test_idcue_constraint/per_sample_rotate_0_90.csv \\
    --csv_b exp/test_baseline/per_sample_rotate_0_90.csv         \\
    --name_a "IDCue-Constraint" --name_b "Baseline"              \\
    --metrics d1 abs_rel rmse                                    \\
    --B 1000 --seed 0

# Save a Markdown table to disk:
python paired_bootstrap.py --csv_a A.csv --csv_b B.csv --save_md report.md

Notes
-----
* d1/d2/d3 are higher-is-better; abs_rel/sq_rel/rmse/rmse_log/log10/silog
  are lower-is-better. Δ is reported as A − B; positive Δ on d1 means A is
  better, negative Δ on abs_rel means A is better.
* Stratified resampling preserves the (dataset × rotate_range) marginal
  distribution. Within each stratum we draw |stratum| indices with replacement.
* Paired bootstrap means re-sampling the SAME index for both methods on each
  resample, which is how the CI on Δ is tightened relative to unpaired tests.
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------- I/O & joining ------------------------------ #

NUMERIC_INT_COLS = {
    "sample_idx", "item_id", "rotate_range_lo", "rotate_range_hi",
    "n_valid_pixels",
}


def _cast(col: str, val: str):
    """Cast a CSV cell to int / float / str depending on column."""
    if col == "dataset":
        return val
    if col in NUMERIC_INT_COLS:
        try:
            return int(val)
        except ValueError:
            return int(float(val))
    try:
        return float(val)
    except ValueError:
        return val


def load_csv(path: str) -> List[dict]:
    if not os.path.isfile(path):
        sys.exit(f"[error] CSV not found: {path}")
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"[error] empty CSV: {path}")
        required = {"dataset", "item_id", "rotate_range_lo", "rotate_range_hi"}
        missing = required - set(reader.fieldnames)
        if missing:
            sys.exit(f"[error] {path} missing required columns: {missing}")
        for row in reader:
            rows.append({k: _cast(k, v) for k, v in row.items()})
    return rows


def join_paired(rows_a: List[dict], rows_b: List[dict],
                metrics: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Inner-join the two row lists on (dataset, item_id, rot_lo, rot_hi).

    Returns
    -------
    mat_a, mat_b : (N, K) float arrays of metric values for each joined sample
    strata_ids   : (N,)   int array of stratum ids = unique (dataset, rot_lo, rot_hi)
    strata_names : list of length max_strata+1 mapping id -> "dataset|lo_hi"
    """
    def _key(r):
        return (r["dataset"], int(r["item_id"]),
                int(r["rotate_range_lo"]), int(r["rotate_range_hi"]))

    map_a = {_key(r): r for r in rows_a}
    map_b = {_key(r): r for r in rows_b}
    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_keys) == 0:
        sys.exit("[error] No common (dataset, item_id, rotate_range) keys "
                 "between the two CSVs. Check that both come from the same "
                 "evaluation seed and same rotate_range setting.")

    # validate metrics exist on both sides
    for m in metrics:
        if m not in rows_a[0] or m not in rows_b[0]:
            sys.exit(f"[error] metric '{m}' not present in one of the CSVs. "
                     f"Available: {sorted(set(rows_a[0]) & set(rows_b[0]))}")

    N, K = len(common_keys), len(metrics)
    mat_a = np.zeros((N, K), dtype=np.float64)
    mat_b = np.zeros((N, K), dtype=np.float64)
    name_to_id: Dict[str, int] = {}
    strata_names: List[str] = []
    strata_ids = np.zeros(N, dtype=np.int64)

    for j, key in enumerate(common_keys):
        ds, _item, lo, hi = key
        s = f"{ds}|{lo}_{hi}"
        if s not in name_to_id:
            name_to_id[s] = len(strata_names)
            strata_names.append(s)
        strata_ids[j] = name_to_id[s]
        for k, met in enumerate(metrics):
            mat_a[j, k] = float(map_a[key][met])
            mat_b[j, k] = float(map_b[key][met])

    return mat_a, mat_b, strata_ids, strata_names


# ------------------------------ Statistics -------------------------------- #

def stratified_paired_bootstrap(mat_a: np.ndarray, mat_b: np.ndarray,
                                 strata_ids: np.ndarray, B: int = 1000,
                                 seed: int = 0):
    """Return per-resample means for A, B and paired diff (A-B).

    Each of the B resamples uses the SAME indices for A and B (paired) and
    samples WITH REPLACEMENT within each stratum (stratified).

    Returns
    -------
    boot_a, boot_b, boot_diff : each shape (B, K)
    """
    rng = np.random.default_rng(seed)
    N, K = mat_a.shape
    n_strata = int(strata_ids.max()) + 1
    by_stratum = [np.where(strata_ids == s)[0] for s in range(n_strata)]

    boot_a = np.empty((B, K), dtype=np.float64)
    boot_b = np.empty((B, K), dtype=np.float64)
    boot_d = np.empty((B, K), dtype=np.float64)
    for b in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True)
                              for ix in by_stratum])
        boot_a[b] = mat_a[idx].mean(axis=0)
        boot_b[b] = mat_b[idx].mean(axis=0)
        boot_d[b] = (mat_a[idx] - mat_b[idx]).mean(axis=0)
    return boot_a, boot_b, boot_d


def percentile_ci(boot: np.ndarray, alpha: float = 0.05):
    lo = np.percentile(boot, 100.0 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100.0 * (1 - alpha / 2), axis=0)
    return lo, hi


def two_sided_p_centered(obs_diff: np.ndarray, boot_diff: np.ndarray) -> np.ndarray:
    """Two-sided centered bootstrap p-value:
        p = Pr(|D* - mean(D*)| >= |obs_diff|)
    where D* are bootstrap-mean differences and `obs_diff` is the observed mean diff.
    Uses the maximum over the two one-sided tail proportions. Min value is 1/B.
    """
    centered = boot_diff - boot_diff.mean(axis=0, keepdims=True)
    p = (np.abs(centered) >= np.abs(obs_diff)[None, :]).mean(axis=0)
    p = np.maximum(p, 1.0 / boot_diff.shape[0])
    return p


def cohens_d_paired(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Paired Cohen's d = mean(A-B) / std(A-B), using sample std with ddof=1."""
    diff = mat_a - mat_b
    sd = diff.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return diff.mean(axis=0) / sd


# ----------------------------- Reporting --------------------------------- #

# higher-is-better metrics — used only to colour the verdict text
HIGHER_IS_BETTER = {"d1", "d2", "d3"}


def fmt(x: float, digits: int = 4) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def render_markdown(name_a: str, name_b: str, metrics: List[str],
                     mean_a, ci_a_lo, ci_a_hi,
                     mean_b, ci_b_lo, ci_b_hi,
                     mean_d, ci_d_lo, ci_d_hi,
                     pvals, dvals, n_samples: int, n_strata: int, B: int) -> str:
    lines = []
    lines.append(f"### Paired bootstrap 95% CI ({name_a} vs. {name_b})")
    lines.append(f"")
    lines.append(f"- Joined samples: **N = {n_samples}**, strata: **{n_strata}**, "
                 f"resamples: **B = {B}**, level: **95%**")
    lines.append(f"- Δ is **{name_a} − {name_b}**. For d1/d2/d3 higher is better; "
                 f"for AbsRel/RMSE/etc. lower is better.")
    lines.append("")
    lines.append("| Metric | "
                 f"{name_a} mean [95% CI] | "
                 f"{name_b} mean [95% CI] | "
                 f"Δ [95% CI] | paired p | Cohen's d |")
    lines.append("|---|---|---|---|---|---|")
    for k, met in enumerate(metrics):
        a_str = f"{fmt(mean_a[k], 4)} [{fmt(ci_a_lo[k], 4)}, {fmt(ci_a_hi[k], 4)}]"
        b_str = f"{fmt(mean_b[k], 4)} [{fmt(ci_b_lo[k], 4)}, {fmt(ci_b_hi[k], 4)}]"
        d_str = f"{fmt(mean_d[k], 4)} [{fmt(ci_d_lo[k], 4)}, {fmt(ci_d_hi[k], 4)}]"
        p_str = f"{pvals[k]:.3g}"
        cd_str = f"{fmt(dvals[k], 3)}"
        lines.append(f"| {met} | {a_str} | {b_str} | {d_str} | {p_str} | {cd_str} |")

    # short verdict
    lines.append("")
    lines.append("**Verdict per metric (α = 0.05):**")
    lines.append("")
    for k, met in enumerate(metrics):
        ci_excludes_zero = (ci_d_lo[k] > 0) or (ci_d_hi[k] < 0)
        signif = "**significant**" if (pvals[k] < 0.05 and ci_excludes_zero) else "not significant"
        if met in HIGHER_IS_BETTER:
            better = name_a if mean_d[k] > 0 else name_b
        else:
            better = name_a if mean_d[k] < 0 else name_b
        lines.append(f"- `{met}`: {signif}; better: **{better}** "
                     f"(Δ={fmt(mean_d[k], 4)}, p={pvals[k]:.3g}, "
                     f"CI excludes 0: {ci_excludes_zero}).")
    return "\n".join(lines)


# ---------------------------------- CLI ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--csv_a", required=True, help="Per-sample CSV for method A")
    ap.add_argument("--csv_b", required=True, help="Per-sample CSV for method B (baseline)")
    ap.add_argument("--name_a", default="Method A")
    ap.add_argument("--name_b", default="Method B")
    ap.add_argument("--metrics", nargs="+",
                    default=["d1", "abs_rel"],
                    help="Metric column names present in BOTH CSVs")
    ap.add_argument("--B", type=int, default=1000, help="Number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_md", default=None,
                    help="Optional path to write the Markdown report")
    args = ap.parse_args()

    rows_a = load_csv(args.csv_a)
    rows_b = load_csv(args.csv_b)
    mat_a, mat_b, strata_ids, strata_names = join_paired(rows_a, rows_b, args.metrics)

    print(f"\n[info] Loaded {len(rows_a)} rows from {args.csv_a}")
    print(f"[info] Loaded {len(rows_b)} rows from {args.csv_b}")
    print(f"[info] Joined N = {mat_a.shape[0]} paired samples across "
          f"{len(strata_names)} strata (dataset × rotate_range).")

    boot_a, boot_b, boot_d = stratified_paired_bootstrap(
        mat_a, mat_b, strata_ids, B=args.B, seed=args.seed)

    mean_a, mean_b = mat_a.mean(axis=0), mat_b.mean(axis=0)
    mean_d = mean_a - mean_b
    ci_a_lo, ci_a_hi = percentile_ci(boot_a)
    ci_b_lo, ci_b_hi = percentile_ci(boot_b)
    ci_d_lo, ci_d_hi = percentile_ci(boot_d)
    pvals = two_sided_p_centered(mean_d, boot_d)
    dvals = cohens_d_paired(mat_a, mat_b)

    md = render_markdown(
        args.name_a, args.name_b, args.metrics,
        mean_a, ci_a_lo, ci_a_hi,
        mean_b, ci_b_lo, ci_b_hi,
        mean_d, ci_d_lo, ci_d_hi,
        pvals, dvals,
        n_samples=mat_a.shape[0], n_strata=len(strata_names), B=args.B,
    )
    print()
    print(md)

    if args.save_md:
        with open(args.save_md, "w") as f:
            f.write(md + "\n")
        print(f"\n[info] Markdown report written to {args.save_md}")


if __name__ == "__main__":
    main()
