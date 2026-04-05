"""
radio_analysis.py -- Statistical analysis and visualization for Asimov's Radio experiments.

Reads from:  governed/results/radio_results.tsv
Outputs to:  governed/results/charts/  (PNG charts)
             governed/results/radio_summary.txt  (text summary)

Analysis functions:
  1. analyze_ml_results()      -- Stats for ML training experiments
  2. analyze_coding_results()  -- Stats for coding task experiments
  3. analyze_radio_states()    -- Radio emotional arc state analysis
  4. generate_charts()         -- Box plots, trajectory, crash rates, effect sizes
  5. print_summary()           -- One-page text summary of all findings

Usage:
  python governed/radio_analysis.py              Run everything
  python governed/radio_analysis.py --ml-only    ML training analysis only
  python governed/radio_analysis.py --coding-only  Coding task analysis only
  python governed/radio_analysis.py --charts-only  Charts only

Dependencies: pandas, numpy, scipy, statsmodels, matplotlib
"""

import argparse
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOVERNED_DIR = Path(__file__).parent
RESULTS_DIR = GOVERNED_DIR / "results"
RADIO_RESULTS_TSV = RESULTS_DIR / "radio_results.tsv"
RADIO_STATES_DIR = RESULTS_DIR / "radio_states"
CHARTS_DIR = RESULTS_DIR / "charts"
SUMMARY_PATH = RESULTS_DIR / "radio_summary.txt"

CONDITION_LABELS = {
    "A": "Ungoverned",
    "B": "Governed",
    "C": "Governed + Radio",
}

CONDITION_COLORS = {
    "A": "#e74c3c",  # red
    "B": "#3498db",  # blue
    "C": "#2ecc71",  # green
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(task_type: str | None = None) -> pd.DataFrame:
    """Load the TSV results file into a DataFrame.

    If task_type is specified, filter to that task type only.
    """
    if not RADIO_RESULTS_TSV.exists():
        print(f"ERROR: Results file not found: {RADIO_RESULTS_TSV}")
        sys.exit(1)

    df = pd.read_csv(RADIO_RESULTS_TSV, sep="\t", dtype=str)

    # Convert numeric columns
    numeric_cols = [
        "run_number", "random_seed", "baseline_metric", "result_metric",
        "delta", "peak_vram_mb", "wall_time_s", "agent_tokens_in",
        "agent_tokens_out", "radio_frustration", "radio_injection_count",
        "radio_consecutive_failures", "iteration_count",
        "tests_passed_before", "tests_passed_after", "lines_changed",
        "regressions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if task_type is not None:
        df = df[df["task_type"] == task_type].copy()

    return df


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_ci(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute 95% CI for Cohen's d using the non-central t approach (approximation)."""
    if n1 < 2 or n2 < 2:
        return (0.0, 0.0)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n1 + n2 - 2)
    return (d - t_crit * se, d + t_crit * se)


def interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_adjust(p_values: list[float], n_tests: int = 3) -> list[float]:
    """Apply Bonferroni correction to a list of p-values."""
    return [min(p * n_tests, 1.0) for p in p_values]


# ---------------------------------------------------------------------------
# 1. ML Results Analysis
# ---------------------------------------------------------------------------

def analyze_ml_results(df: pd.DataFrame | None = None) -> dict:
    """Analyze ML training experiment results.

    Returns a dict of summary statistics.
    """
    if df is None:
        df = load_results("ml_training")

    if df.empty:
        print("  No ML training results found.")
        return {}

    print(f"\n{'=' * 70}")
    print("ML TRAINING RESULTS ANALYSIS")
    print(f"{'=' * 70}")

    results = {}

    # Per-condition stats
    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        deltas = cdf["delta"].dropna().values

        n = len(deltas)
        if n == 0:
            results[cond] = {"n": 0}
            continue

        crash_count = len(cdf[cdf["status"] == "crash"])
        improved_count = len(cdf[cdf["status"] == "improved"])
        improvement_rate = improved_count / n if n > 0 else 0.0

        cond_stats = {
            "n": n,
            "mean_delta": np.mean(deltas),
            "std_delta": np.std(deltas, ddof=1) if n > 1 else 0.0,
            "median_delta": np.median(deltas),
            "crash_count": crash_count,
            "improved_count": improved_count,
            "improvement_rate": improvement_rate,
        }

        # Shapiro-Wilk normality test (need at least 3 samples)
        if n >= 3:
            sw_stat, sw_p = stats.shapiro(deltas)
            cond_stats["shapiro_stat"] = sw_stat
            cond_stats["shapiro_p"] = sw_p
            cond_stats["normal"] = sw_p > 0.05
        else:
            cond_stats["shapiro_stat"] = None
            cond_stats["shapiro_p"] = None
            cond_stats["normal"] = None

        results[cond] = cond_stats

    # Print per-condition table
    print(f"\n  {'Metric':<25} {'A (Ungoverned)':>18} {'B (Governed)':>18} {'C (Gov+Radio)':>18}")
    print(f"  {'-' * 25} {'-' * 18} {'-' * 18} {'-' * 18}")

    def _fmt(cond, key, fmt_str="{:.6f}"):
        v = results.get(cond, {}).get(key)
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return fmt_str.format(v)
        return str(v)

    print(f"  {'N':<25} {_fmt('A', 'n', '{:.0f}'):>18} {_fmt('B', 'n', '{:.0f}'):>18} {_fmt('C', 'n', '{:.0f}'):>18}")
    print(f"  {'Mean delta':<25} {_fmt('A', 'mean_delta'):>18} {_fmt('B', 'mean_delta'):>18} {_fmt('C', 'mean_delta'):>18}")
    print(f"  {'Std delta':<25} {_fmt('A', 'std_delta'):>18} {_fmt('B', 'std_delta'):>18} {_fmt('C', 'std_delta'):>18}")
    print(f"  {'Median delta':<25} {_fmt('A', 'median_delta'):>18} {_fmt('B', 'median_delta'):>18} {_fmt('C', 'median_delta'):>18}")
    print(f"  {'Crash count':<25} {_fmt('A', 'crash_count', '{:.0f}'):>18} {_fmt('B', 'crash_count', '{:.0f}'):>18} {_fmt('C', 'crash_count', '{:.0f}'):>18}")
    print(f"  {'Improvement rate':<25} {_fmt('A', 'improvement_rate', '{:.2%}'):>18} {_fmt('B', 'improvement_rate', '{:.2%}'):>18} {_fmt('C', 'improvement_rate', '{:.2%}'):>18}")
    print(f"  {'Shapiro-Wilk p':<25} {_fmt('A', 'shapiro_p', '{:.4f}'):>18} {_fmt('B', 'shapiro_p', '{:.4f}'):>18} {_fmt('C', 'shapiro_p', '{:.4f}'):>18}")

    # Omnibus test
    groups = []
    group_labels = []
    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        deltas = cdf["delta"].dropna().values
        if len(deltas) >= 2:
            groups.append(deltas)
            group_labels.append(cond)

    print(f"\n  Omnibus Test:")
    if len(groups) >= 2:
        all_normal = all(results.get(c, {}).get("normal", False) for c in group_labels)

        if all_normal:
            f_stat, omnibus_p = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
            results["omnibus_test"] = "anova"
            results["omnibus_stat"] = f_stat
        else:
            h_stat, omnibus_p = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis H"
            results["omnibus_test"] = "kruskal"
            results["omnibus_stat"] = h_stat

        results["omnibus_p"] = omnibus_p
        print(f"    {test_name}: stat={results['omnibus_stat']:.4f}, p={omnibus_p:.6f}")
        print(f"    {'Significant (p < 0.05)' if omnibus_p < 0.05 else 'Not significant'}")
    else:
        print("    Not enough groups with data for omnibus test.")

    # Pairwise comparisons with Bonferroni correction
    print(f"\n  Pairwise Comparisons (Bonferroni-corrected):")
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    pairwise_results = []

    for c1, c2 in pairs:
        g1 = df[df["condition"] == c1]["delta"].dropna().values
        g2 = df[df["condition"] == c2]["delta"].dropna().values

        if len(g1) < 2 or len(g2) < 2:
            print(f"    {c1} vs {c2}: Insufficient data")
            pairwise_results.append({"pair": f"{c1}_vs_{c2}", "p": 1.0, "d": 0.0})
            continue

        # Use Mann-Whitney U (non-parametric) for robustness
        u_stat, raw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d(g1, g2)
        ci_lo, ci_hi = cohens_d_ci(d, len(g1), len(g2))

        pairwise_results.append({
            "pair": f"{c1}_vs_{c2}",
            "u_stat": u_stat,
            "raw_p": raw_p,
            "d": d,
            "d_ci_lo": ci_lo,
            "d_ci_hi": ci_hi,
        })

    # Apply Bonferroni correction
    raw_ps = [pr["raw_p"] for pr in pairwise_results if "raw_p" in pr]
    corrected_ps = bonferroni_adjust(raw_ps)

    for i, pr in enumerate(pairwise_results):
        if "raw_p" not in pr:
            continue
        pr["corrected_p"] = corrected_ps[i]
        sig = "*" if corrected_ps[i] < 0.05 else ""
        print(f"    {pr['pair']:>10}: p={corrected_ps[i]:.6f}{sig}  "
              f"d={pr['d']:+.4f} [{pr.get('d_ci_lo', 0):.4f}, {pr.get('d_ci_hi', 0):.4f}] "
              f"({interpret_d(pr['d'])})")

    results["pairwise"] = pairwise_results
    return results


# ---------------------------------------------------------------------------
# 2. Coding Results Analysis
# ---------------------------------------------------------------------------

def analyze_coding_results(df: pd.DataFrame | None = None) -> dict:
    """Analyze coding task experiment results.

    Returns a dict of summary statistics.
    """
    if df is None:
        df = load_results("coding")

    if df.empty:
        print("  No coding task results found.")
        return {}

    print(f"\n{'=' * 70}")
    print("CODING TASK RESULTS ANALYSIS")
    print(f"{'=' * 70}")

    results = {}

    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        n = len(cdf)
        if n == 0:
            results[cond] = {"n": 0}
            continue

        fix_rates = cdf["delta"].dropna().values  # delta stores fix_rate for coding tasks
        iterations = cdf["iteration_count"].dropna().values

        fully_fixed = len(cdf[cdf["status"] == "fully_fixed"])
        partially_fixed = len(cdf[cdf["status"] == "partially_fixed"])
        no_change = len(cdf[cdf["status"] == "no_change"])
        regressed = len(cdf[cdf["status"] == "regressed"])
        fix_failed = len(cdf[cdf["status"] == "fix_failed"])
        parse_error = len(cdf[cdf["status"] == "parse_error"])
        gov_blocked = len(cdf[cdf["status"] == "governance_blocked"])

        regression_rate = regressed / n if n > 0 else 0.0

        cond_stats = {
            "n": n,
            "mean_fix_rate": np.mean(fix_rates) if len(fix_rates) > 0 else 0.0,
            "std_fix_rate": np.std(fix_rates, ddof=1) if len(fix_rates) > 1 else 0.0,
            "median_fix_rate": np.median(fix_rates) if len(fix_rates) > 0 else 0.0,
            "mean_iterations": np.mean(iterations) if len(iterations) > 0 else 0.0,
            "fully_fixed": fully_fixed,
            "partially_fixed": partially_fixed,
            "no_change": no_change,
            "regressed": regressed,
            "fix_failed": fix_failed,
            "parse_error": parse_error,
            "governance_blocked": gov_blocked,
            "regression_rate": regression_rate,
            "success_rate": (fully_fixed + partially_fixed) / n if n > 0 else 0.0,
        }

        # Shapiro-Wilk
        if len(fix_rates) >= 3:
            sw_stat, sw_p = stats.shapiro(fix_rates)
            cond_stats["shapiro_stat"] = sw_stat
            cond_stats["shapiro_p"] = sw_p
            cond_stats["normal"] = sw_p > 0.05
        else:
            cond_stats["shapiro_stat"] = None
            cond_stats["shapiro_p"] = None
            cond_stats["normal"] = None

        results[cond] = cond_stats

    # Print table
    print(f"\n  {'Metric':<25} {'A (Ungoverned)':>18} {'B (Governed)':>18} {'C (Gov+Radio)':>18}")
    print(f"  {'-' * 25} {'-' * 18} {'-' * 18} {'-' * 18}")

    def _fmt(cond, key, fmt_str="{:.4f}"):
        v = results.get(cond, {}).get(key)
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return fmt_str.format(v)
        return str(v)

    rows = [
        ("N", "n", "{:.0f}"),
        ("Mean fix rate", "mean_fix_rate", "{:.4f}"),
        ("Std fix rate", "std_fix_rate", "{:.4f}"),
        ("Median fix rate", "median_fix_rate", "{:.4f}"),
        ("Mean iterations", "mean_iterations", "{:.2f}"),
        ("Fully fixed", "fully_fixed", "{:.0f}"),
        ("Partially fixed", "partially_fixed", "{:.0f}"),
        ("No change", "no_change", "{:.0f}"),
        ("Regressed", "regressed", "{:.0f}"),
        ("Fix failed", "fix_failed", "{:.0f}"),
        ("Parse error", "parse_error", "{:.0f}"),
        ("Gov blocked", "governance_blocked", "{:.0f}"),
        ("Success rate", "success_rate", "{:.2%}"),
        ("Regression rate", "regression_rate", "{:.2%}"),
        ("Shapiro-Wilk p", "shapiro_p", "{:.4f}"),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<25} {_fmt('A', key, fmt):>18} {_fmt('B', key, fmt):>18} {_fmt('C', key, fmt):>18}")

    # Omnibus test on fix rates
    groups = []
    group_labels = []
    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        fix_rates = cdf["delta"].dropna().values
        if len(fix_rates) >= 2:
            groups.append(fix_rates)
            group_labels.append(cond)

    print(f"\n  Omnibus Test (fix rates):")
    if len(groups) >= 2:
        all_normal = all(results.get(c, {}).get("normal", False) for c in group_labels)

        if all_normal:
            stat, omnibus_p = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
            results["omnibus_test"] = "anova"
        else:
            stat, omnibus_p = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis H"
            results["omnibus_test"] = "kruskal"

        results["omnibus_stat"] = stat
        results["omnibus_p"] = omnibus_p
        print(f"    {test_name}: stat={stat:.4f}, p={omnibus_p:.6f}")
        print(f"    {'Significant (p < 0.05)' if omnibus_p < 0.05 else 'Not significant'}")
    else:
        print("    Not enough groups for omnibus test.")

    # Pairwise comparisons
    print(f"\n  Pairwise Comparisons (Bonferroni-corrected):")
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    pairwise_results = []

    for c1, c2 in pairs:
        g1 = df[df["condition"] == c1]["delta"].dropna().values
        g2 = df[df["condition"] == c2]["delta"].dropna().values

        if len(g1) < 2 or len(g2) < 2:
            print(f"    {c1} vs {c2}: Insufficient data")
            pairwise_results.append({"pair": f"{c1}_vs_{c2}", "p": 1.0, "d": 0.0})
            continue

        u_stat, raw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d(g1, g2)
        ci_lo, ci_hi = cohens_d_ci(d, len(g1), len(g2))

        pairwise_results.append({
            "pair": f"{c1}_vs_{c2}",
            "u_stat": u_stat,
            "raw_p": raw_p,
            "d": d,
            "d_ci_lo": ci_lo,
            "d_ci_hi": ci_hi,
        })

    raw_ps = [pr["raw_p"] for pr in pairwise_results if "raw_p" in pr]
    corrected_ps = bonferroni_adjust(raw_ps)

    for i, pr in enumerate(pairwise_results):
        if "raw_p" not in pr:
            continue
        pr["corrected_p"] = corrected_ps[i]
        sig = "*" if corrected_ps[i] < 0.05 else ""
        print(f"    {pr['pair']:>10}: p={corrected_ps[i]:.6f}{sig}  "
              f"d={pr['d']:+.4f} [{pr.get('d_ci_lo', 0):.4f}, {pr.get('d_ci_hi', 0):.4f}] "
              f"({interpret_d(pr['d'])})")

    results["pairwise"] = pairwise_results
    return results


# ---------------------------------------------------------------------------
# 3. Radio State Analysis
# ---------------------------------------------------------------------------

def analyze_radio_states() -> dict:
    """Analyze Radio emotional arc state snapshots from condition C experiments.

    Returns a dict of Radio-specific metrics.
    """
    print(f"\n{'=' * 70}")
    print("RADIO STATE ANALYSIS")
    print(f"{'=' * 70}")

    if not RADIO_STATES_DIR.exists():
        print("  No Radio state directory found.")
        return {}

    state_files = sorted(RADIO_STATES_DIR.glob("*_state.json"))
    if not state_files:
        print("  No Radio state snapshots found.")
        return {}

    print(f"  Total state snapshots: {len(state_files)}")

    states = []
    for sf in state_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            states.append(data)
        except Exception:
            continue

    if not states:
        print("  Could not parse any state files.")
        return {}

    # Filter to condition C only
    c_states = [s for s in states if s.get("condition") == "C"]
    print(f"  Condition C snapshots: {len(c_states)}")

    results = {}

    # Mode transition counts per experiment
    transition_counts = []
    for s in c_states:
        arc = s.get("arc", {})
        mode_history = arc.get("modeHistory", [])
        transition_counts.append(len(mode_history))

    results["n_experiments"] = len(c_states)
    results["mean_transitions"] = np.mean(transition_counts) if transition_counts else 0.0
    results["median_transitions"] = np.median(transition_counts) if transition_counts else 0.0
    results["max_transitions"] = max(transition_counts) if transition_counts else 0

    print(f"\n  Mode Transitions:")
    print(f"    Mean per experiment:   {results['mean_transitions']:.2f}")
    print(f"    Median per experiment: {results['median_transitions']:.1f}")
    print(f"    Max:                   {results['max_transitions']}")

    # Frustration at time of shift activation
    frustration_at_shift = []
    for s in c_states:
        arc = s.get("arc", {})
        mode_history = arc.get("modeHistory", [])
        for entry in mode_history:
            if entry.get("mode") == "shift":
                # The frustration level at the time of this transition
                frust = entry.get("frustrationLevel", arc.get("frustrationLevel", 0))
                frustration_at_shift.append(frust)

    if frustration_at_shift:
        results["mean_frustration_at_shift"] = np.mean(frustration_at_shift)
        results["median_frustration_at_shift"] = np.median(frustration_at_shift)
        print(f"\n  Frustration at Shift Activation:")
        print(f"    Mean:   {results['mean_frustration_at_shift']:.4f}")
        print(f"    Median: {results['median_frustration_at_shift']:.4f}")
        print(f"    N (shift events): {len(frustration_at_shift)}")
    else:
        results["mean_frustration_at_shift"] = None
        print(f"\n  No shift mode activations found.")

    # Time to first shift mode
    time_to_shift = []
    for s in c_states:
        arc = s.get("arc", {})
        mode_history = arc.get("modeHistory", [])
        for i, entry in enumerate(mode_history):
            if entry.get("mode") == "shift":
                time_to_shift.append(i + 1)  # 1-indexed position
                break

    if time_to_shift:
        results["mean_time_to_shift"] = np.mean(time_to_shift)
        results["median_time_to_shift"] = np.median(time_to_shift)
        print(f"\n  Time to First Shift Mode (transition count):")
        print(f"    Mean:   {results['mean_time_to_shift']:.2f}")
        print(f"    Median: {results['median_time_to_shift']:.1f}")
    else:
        results["mean_time_to_shift"] = None

    # Celebration frequency
    celebration_count = 0
    total_events = 0
    for s in c_states:
        arc = s.get("arc", {})
        mode_history = arc.get("modeHistory", [])
        total_events += len(mode_history)
        for entry in mode_history:
            if entry.get("mode") == "celebration":
                celebration_count += 1

    results["celebration_count"] = celebration_count
    results["celebration_rate"] = (
        celebration_count / total_events if total_events > 0 else 0.0
    )
    print(f"\n  Celebration Events:")
    print(f"    Total: {celebration_count}")
    print(f"    Rate:  {results['celebration_rate']:.2%} of all mode transitions")

    # Correlation: injection count vs. delta from TSV
    df = load_results()
    if not df.empty:
        c_df = df[(df["condition"] == "C") & (df["radio_injection_count"] > 0)]
        if len(c_df) >= 3:
            injections = c_df["radio_injection_count"].values
            deltas = c_df["delta"].values
            if np.std(injections) > 0 and np.std(deltas) > 0:
                r, p = stats.pearsonr(injections, deltas)
                results["injection_delta_corr_r"] = r
                results["injection_delta_corr_p"] = p
                print(f"\n  Injection Count vs. Delta Correlation:")
                print(f"    Pearson r: {r:.4f}  (p={p:.6f})")

        # Post-injection vs non-injection delta comparison
        injected = df[(df["condition"] == "C") & (df["radio_injection_count"] > 0)]["delta"].values
        non_injected = df[(df["condition"] == "C") & (df["radio_injection_count"] == 0)]["delta"].values
        if len(injected) >= 2 and len(non_injected) >= 2:
            u, p = stats.mannwhitneyu(injected, non_injected, alternative="two-sided")
            d = cohens_d(injected, non_injected)
            results["injection_vs_noinjection_p"] = p
            results["injection_vs_noinjection_d"] = d
            print(f"\n  Post-injection vs. Non-injection Delta:")
            print(f"    Injected mean:     {np.mean(injected):.6f} (n={len(injected)})")
            print(f"    Non-injected mean: {np.mean(non_injected):.6f} (n={len(non_injected)})")
            print(f"    Mann-Whitney U p: {p:.6f}  d={d:+.4f}")

    return results


# ---------------------------------------------------------------------------
# 4. Chart Generation
# ---------------------------------------------------------------------------

def generate_charts(df: pd.DataFrame | None = None) -> None:
    """Generate all visualization charts."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    if df is None:
        df = load_results()

    if df.empty:
        print("  No results to chart.")
        return

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("GENERATING CHARTS")
    print(f"{'=' * 70}")

    # Determine which task types are present
    task_types = df["task_type"].unique()

    for task_type in task_types:
        tdf = df[df["task_type"] == task_type]
        suffix = "ml" if task_type == "ml_training" else "coding"

        # Chart 1: Box plot -- delta by condition with beeswarm overlay
        _chart_boxplot(tdf, suffix, task_type)

        # Chart 2: Trajectory -- running mean delta over experiment number
        _chart_trajectory(tdf, suffix, task_type)

        # Chart 3: Crash/failure rate bar chart
        _chart_failure_rates(tdf, suffix, task_type)

    # Chart 4: Frustration timeline for condition C (across all task types)
    _chart_frustration_timeline(df)

    # Chart 5: Effect size forest plot
    _chart_effect_sizes(df)

    print(f"\n  All charts saved to: {CHARTS_DIR}")


def _chart_boxplot(df: pd.DataFrame, suffix: str, task_type: str) -> None:
    """Box plot of delta by condition with jittered beeswarm overlay."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    data_by_cond = []
    labels = []
    colors = []

    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        deltas = cdf["delta"].dropna().values
        if len(deltas) > 0:
            data_by_cond.append(deltas)
            labels.append(f"{cond}\n{CONDITION_LABELS[cond]}")
            colors.append(CONDITION_COLORS[cond])

    if not data_by_cond:
        plt.close()
        return

    bp = ax.boxplot(data_by_cond, labels=labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Beeswarm overlay with jitter
    rng = np.random.default_rng(42)
    for i, (deltas, color) in enumerate(zip(data_by_cond, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(deltas))
        ax.scatter(
            np.full_like(deltas, i + 1) + jitter,
            deltas,
            color=color, alpha=0.5, s=15, zorder=3, edgecolors="none",
        )

    metric_label = "Delta (val_bpb improvement)" if "ml" in suffix else "Fix Rate"
    ax.set_ylabel(metric_label)
    ax.set_title(f"{task_type.replace('_', ' ').title()} -- Delta by Condition")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = CHARTS_DIR / f"{suffix}_boxplot_delta.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def _chart_trajectory(df: pd.DataFrame, suffix: str, task_type: str) -> None:
    """Running mean delta over experiment number, one line per condition."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond].sort_values("run_number")
        deltas = cdf["delta"].dropna().values
        if len(deltas) < 2:
            continue

        # Compute running (cumulative) mean
        running_mean = np.cumsum(deltas) / np.arange(1, len(deltas) + 1)
        ax.plot(
            range(1, len(running_mean) + 1),
            running_mean,
            label=f"{cond} ({CONDITION_LABELS[cond]})",
            color=CONDITION_COLORS[cond],
            linewidth=2,
        )

    metric_label = "Running Mean Delta" if "ml" in suffix else "Running Mean Fix Rate"
    ax.set_xlabel("Experiment Number")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{task_type.replace('_', ' ').title()} -- Trajectory")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = CHARTS_DIR / f"{suffix}_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def _chart_failure_rates(df: pd.DataFrame, suffix: str, task_type: str) -> None:
    """Bar chart of crash/failure rates with error bars (binomial CI)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    fail_statuses = {"crash", "fix_failed", "parse_error"}
    conds = []
    rates = []
    errors = []
    bar_colors = []

    for cond in ("A", "B", "C"):
        cdf = df[df["condition"] == cond]
        n = len(cdf)
        if n == 0:
            continue
        n_fail = len(cdf[cdf["status"].isin(fail_statuses)])
        rate = n_fail / n

        # Wilson score interval for binomial proportion
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom

        conds.append(f"{cond}\n{CONDITION_LABELS[cond]}")
        rates.append(rate)
        errors.append(margin)
        bar_colors.append(CONDITION_COLORS[cond])

    if not conds:
        plt.close()
        return

    bars = ax.bar(conds, rates, yerr=errors, color=bar_colors, alpha=0.7,
                  capsize=5, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Failure Rate")
    ax.set_title(f"{task_type.replace('_', ' ').title()} -- Crash/Failure Rate by Condition")
    ax.set_ylim(0, min(1.0, max(rates) * 1.5 + 0.05) if rates else 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    path = CHARTS_DIR / f"{suffix}_failure_rates.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def _chart_frustration_timeline(df: pd.DataFrame) -> None:
    """Frustration level over time for condition C experiments."""
    import matplotlib.pyplot as plt

    cdf = df[df["condition"] == "C"].sort_values("run_number")
    frustrations = cdf["radio_frustration"].dropna().values

    if len(frustrations) < 2:
        print("  Skipping frustration timeline (insufficient data)")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(range(1, len(frustrations) + 1), frustrations,
            color=CONDITION_COLORS["C"], linewidth=1.5, alpha=0.8)
    ax.fill_between(range(1, len(frustrations) + 1), frustrations,
                    alpha=0.15, color=CONDITION_COLORS["C"])

    # Mark mode transitions if injection count changes
    injections = cdf["radio_injection_count"].dropna().values
    for i in range(1, len(injections)):
        if injections[i] > injections[i - 1]:
            ax.axvline(x=i + 1, color="orange", alpha=0.4, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Experiment Number (Condition C)")
    ax.set_ylabel("Frustration Level")
    ax.set_title("Radio Frustration Timeline -- Condition C")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = CHARTS_DIR / "frustration_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


def _chart_effect_sizes(df: pd.DataFrame) -> None:
    """Forest plot of Cohen's d effect sizes for all pairwise comparisons."""
    import matplotlib.pyplot as plt

    comparisons = []
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    task_types = [tt for tt in df["task_type"].unique() if tt in ("ml_training", "coding")]

    for task_type in task_types:
        tdf = df[df["task_type"] == task_type]
        label_prefix = "ML" if task_type == "ml_training" else "Coding"

        for c1, c2 in pairs:
            g1 = tdf[tdf["condition"] == c1]["delta"].dropna().values
            g2 = tdf[tdf["condition"] == c2]["delta"].dropna().values

            if len(g1) < 2 or len(g2) < 2:
                continue

            d = cohens_d(g1, g2)
            ci_lo, ci_hi = cohens_d_ci(d, len(g1), len(g2))

            comparisons.append({
                "label": f"{label_prefix}: {c1} vs {c2}",
                "d": d,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

    if not comparisons:
        print("  Skipping effect size forest plot (insufficient data)")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons) * 0.6 + 1)))

    y_positions = list(range(len(comparisons)))
    labels = [c["label"] for c in comparisons]
    d_values = [c["d"] for c in comparisons]
    ci_los = [c["ci_lo"] for c in comparisons]
    ci_his = [c["ci_hi"] for c in comparisons]

    # Plot CI lines
    for i, comp in enumerate(comparisons):
        ax.plot(
            [comp["ci_lo"], comp["ci_hi"]], [i, i],
            color="#333333", linewidth=2, solid_capstyle="round",
        )
        ax.plot(comp["d"], i, "o", color="#e74c3c", markersize=8, zorder=5)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6)

    # Effect size region shading
    ax.axvspan(-0.2, 0.2, alpha=0.05, color="gray", label="Negligible")
    ax.axvspan(-0.5, -0.2, alpha=0.05, color="yellow")
    ax.axvspan(0.2, 0.5, alpha=0.05, color="yellow")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (with 95% CI)")
    ax.set_title("Effect Size Forest Plot")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    path = CHARTS_DIR / "effect_size_forest.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

def print_summary(
    ml_results: dict | None = None,
    coding_results: dict | None = None,
    radio_results: dict | None = None,
) -> None:
    """Print and save a one-page text summary of all findings."""
    buf = StringIO()

    def w(line=""):
        buf.write(line + "\n")
        print(line)

    w("=" * 70)
    w("ASIMOV'S RADIO EXPERIMENT SUMMARY")
    w("=" * 70)
    w()

    # ML Training
    w("1. ML TRAINING EXPERIMENTS")
    w("-" * 40)
    if ml_results and any(ml_results.get(c, {}).get("n", 0) > 0 for c in ("A", "B", "C")):
        for cond in ("A", "B", "C"):
            cs = ml_results.get(cond, {})
            if cs.get("n", 0) == 0:
                continue
            w(f"  Condition {cond} ({CONDITION_LABELS[cond]}): "
              f"n={cs['n']}, mean delta={cs.get('mean_delta', 0):.6f}, "
              f"improvement rate={cs.get('improvement_rate', 0):.1%}, "
              f"crashes={cs.get('crash_count', 0)}")

        omnibus_p = ml_results.get("omnibus_p")
        if omnibus_p is not None:
            test = ml_results.get("omnibus_test", "unknown")
            w(f"\n  Omnibus ({test}): p={omnibus_p:.6f} "
              f"{'-- SIGNIFICANT' if omnibus_p < 0.05 else '-- not significant'}")

        for pr in ml_results.get("pairwise", []):
            if "corrected_p" in pr:
                sig = " *" if pr["corrected_p"] < 0.05 else ""
                w(f"  {pr['pair']}: p={pr['corrected_p']:.6f}{sig}, "
                  f"d={pr['d']:+.4f} ({interpret_d(pr['d'])})")
    else:
        w("  No ML training data available.")

    w()

    # Coding Tasks
    w("2. CODING TASK EXPERIMENTS")
    w("-" * 40)
    if coding_results and any(coding_results.get(c, {}).get("n", 0) > 0 for c in ("A", "B", "C")):
        for cond in ("A", "B", "C"):
            cs = coding_results.get(cond, {})
            if cs.get("n", 0) == 0:
                continue
            w(f"  Condition {cond} ({CONDITION_LABELS[cond]}): "
              f"n={cs['n']}, mean fix rate={cs.get('mean_fix_rate', 0):.4f}, "
              f"success rate={cs.get('success_rate', 0):.1%}, "
              f"regression rate={cs.get('regression_rate', 0):.1%}")

        omnibus_p = coding_results.get("omnibus_p")
        if omnibus_p is not None:
            test = coding_results.get("omnibus_test", "unknown")
            w(f"\n  Omnibus ({test}): p={omnibus_p:.6f} "
              f"{'-- SIGNIFICANT' if omnibus_p < 0.05 else '-- not significant'}")

        for pr in coding_results.get("pairwise", []):
            if "corrected_p" in pr:
                sig = " *" if pr["corrected_p"] < 0.05 else ""
                w(f"  {pr['pair']}: p={pr['corrected_p']:.6f}{sig}, "
                  f"d={pr['d']:+.4f} ({interpret_d(pr['d'])})")
    else:
        w("  No coding task data available.")

    w()

    # Radio States
    w("3. RADIO EMOTIONAL ARC")
    w("-" * 40)
    if radio_results and radio_results.get("n_experiments", 0) > 0:
        w(f"  Experiments analyzed: {radio_results['n_experiments']}")
        w(f"  Mean mode transitions: {radio_results.get('mean_transitions', 0):.2f}")
        w(f"  Celebration events: {radio_results.get('celebration_count', 0)} "
          f"({radio_results.get('celebration_rate', 0):.1%} of transitions)")

        if radio_results.get("mean_frustration_at_shift") is not None:
            w(f"  Mean frustration at shift: {radio_results['mean_frustration_at_shift']:.4f}")
        if radio_results.get("mean_time_to_shift") is not None:
            w(f"  Mean time to first shift: {radio_results['mean_time_to_shift']:.2f} transitions")
        if radio_results.get("injection_delta_corr_r") is not None:
            w(f"  Injection-delta correlation: r={radio_results['injection_delta_corr_r']:.4f} "
              f"(p={radio_results.get('injection_delta_corr_p', 1):.6f})")
    else:
        w("  No Radio state data available.")

    w()

    # Key Conclusions
    w("4. KEY CONCLUSIONS")
    w("-" * 40)

    conclusions = []

    # Check ML significance
    if ml_results:
        ml_p = ml_results.get("omnibus_p")
        if ml_p is not None and ml_p < 0.05:
            conclusions.append(
                "ML training: Statistically significant differences exist "
                "between conditions (p < 0.05)."
            )
        elif ml_p is not None:
            conclusions.append(
                "ML training: No statistically significant differences "
                "between conditions."
            )

    # Check coding significance
    if coding_results:
        cod_p = coding_results.get("omnibus_p")
        if cod_p is not None and cod_p < 0.05:
            conclusions.append(
                "Coding tasks: Statistically significant differences exist "
                "between conditions (p < 0.05)."
            )
        elif cod_p is not None:
            conclusions.append(
                "Coding tasks: No statistically significant differences "
                "between conditions."
            )

    # Radio effect
    if radio_results and radio_results.get("injection_vs_noinjection_d") is not None:
        d_val = radio_results["injection_vs_noinjection_d"]
        p_val = radio_results.get("injection_vs_noinjection_p", 1)
        conclusions.append(
            f"Radio injection effect: d={d_val:+.4f} ({interpret_d(d_val)}), "
            f"p={p_val:.6f}."
        )

    if not conclusions:
        conclusions.append("Insufficient data for conclusions. Run more experiments.")

    for c in conclusions:
        w(f"  - {c}")

    w()
    w("=" * 70)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\n  Summary saved to: {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asimov's Radio experiment analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                     Run all analyses\n"
            "  %(prog)s --ml-only           ML training analysis only\n"
            "  %(prog)s --coding-only       Coding task analysis only\n"
            "  %(prog)s --charts-only       Generate charts only\n"
        ),
    )

    parser.add_argument("--ml-only", action="store_true", help="ML training analysis only")
    parser.add_argument("--coding-only", action="store_true", help="Coding task analysis only")
    parser.add_argument("--charts-only", action="store_true", help="Generate charts only")

    args = parser.parse_args()

    if args.charts_only:
        generate_charts()
        return

    ml_results = None
    coding_results = None
    radio_results = None

    if args.ml_only:
        ml_results = analyze_ml_results()
        radio_results = analyze_radio_states()
        generate_charts(load_results("ml_training"))
        print_summary(ml_results=ml_results, radio_results=radio_results)
        return

    if args.coding_only:
        coding_results = analyze_coding_results()
        radio_results = analyze_radio_states()
        generate_charts(load_results("coding"))
        print_summary(coding_results=coding_results, radio_results=radio_results)
        return

    # Run everything
    ml_results = analyze_ml_results()
    coding_results = analyze_coding_results()
    radio_results = analyze_radio_states()
    generate_charts()
    print_summary(
        ml_results=ml_results,
        coding_results=coding_results,
        radio_results=radio_results,
    )


if __name__ == "__main__":
    main()
