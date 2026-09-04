#!/usr/bin/env python3
"""Fail-closed Paper-1 analysis, figures, and LaTeX fragments.

Two modes are deliberately separated:

``diagnostic`` renders only the already-completed, non-performance geometry
checks.  ``confirmatory`` reads the frozen 30-seed x 5-bandwidth campaign and
refuses to call an output final unless all 150 cells and their provenance
records validate.

The event is the analysis unit.  Seed is the resampling/pairing cluster.
Thresholds are read from the calibration-only frozen threshold contract and
are never estimated from campaign data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


RUN_RE = re.compile(r"paper1_s(?P<seed>\d{3})_(?P<bandwidth>.+)$")
BANDWIDTH_ORDER = [
    "bw00_floor",
    "bw01_10kbps",
    "bw02_100kbps",
    "bw03_1mbps",
    "bw04_oracle",
]
BANDWIDTH_LABELS = {
    "bw00_floor": "Floor",
    "bw01_10kbps": "10 kb/s",
    "bw02_100kbps": "100 kb/s",
    "bw03_1mbps": "1 Mb/s",
    "bw04_oracle": "Oracle-like",
}
METRIC_LABELS = {
    "s": "Full metric",
    "chi2": r"Estimator-matched $\chi^2$",
    "huber": "Huber",
    "lnr": "LNR",
    "sB1": "Residual + B1",
    "sB2": "Residual + B2",
    "s_gated_lmax": "Coverage-gated",
    "s_delta_lmax": r"Conservative $\delta$ check",
}
METRIC_COLORS = {
    "s": "#0072B2",
    "chi2": "#D55E00",
    "sB1": "#009E73",
    "sB2": "#CC79A7",
    "s_gated_lmax": "#E69F00",
    "huber": "#56B4E9",
    "lnr": "#666666",
    "s_delta_lmax": "#000000",
}
DESIGN_SHA256 = "f4c2b422b3fb113f1e33bd19aafd89f1591f18f1c5272d88a02d84a3d8d154f1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seed_spec(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(x) for x in token.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if len(set(result)) != len(result):
        raise ValueError("seed specification contains duplicates")
    return result


def empirical_auc(y: Sequence[bool], score: Sequence[float]) -> float:
    """Mann-Whitney AUC with half credit for ties."""
    yv = np.asarray(y, dtype=bool)
    sv = np.asarray(score, dtype=float)
    finite = np.isfinite(sv)
    yv, sv = yv[finite], sv[finite]
    n_pos = int(yv.sum())
    n_neg = int((~yv).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(sv, method="average")
    u = float(ranks[yv].sum() - n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


def roc_points(y: Sequence[bool], score: Sequence[float]) -> pd.DataFrame:
    yv = np.asarray(y, dtype=bool)
    sv = np.asarray(score, dtype=float)
    finite = np.isfinite(sv)
    yv, sv = yv[finite], sv[finite]
    positives, negatives = int(yv.sum()), int((~yv).sum())
    if positives == 0 or negatives == 0:
        return pd.DataFrame(columns=["threshold", "fpr", "tpr"])
    order = np.argsort(-sv, kind="mergesort")
    yv, sv = yv[order], sv[order]
    distinct = np.r_[np.flatnonzero(np.diff(sv)), len(sv) - 1]
    tp = np.cumsum(yv)[distinct]
    fp = 1 + distinct - tp
    return pd.DataFrame(
        {
            "threshold": np.r_[np.inf, sv[distinct]],
            "fpr": np.r_[0.0, fp / negatives],
            "tpr": np.r_[0.0, tp / positives],
        }
    )


def bootstrap_mean_ci(
    values: Sequence[float], *, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    means = np.empty(draws, dtype=float)
    for index in range(draws):
        means[index] = np.mean(rng.choice(x, size=len(x), replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min((len(p) - rank) * p[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def safe_wilcoxon(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x, 0.0):
        return 1.0
    return float(wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)


def read_score_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_output_hashes(output: Path) -> None:
    hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "paper1_output_sha256.json":
            hashes[str(path.relative_to(output))] = sha256(path)
    (output / "paper1_output_sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def write_diagnostic_outputs(static_results: Path, output: Path) -> None:
    """Render evidence that is already complete and safe to cite as diagnostic."""
    publication_style()
    figures = output / "figures"
    tables = output / "tables"
    latex = output / "latex"
    for directory in (figures, tables, latex):
        directory.mkdir(parents=True, exist_ok=True)

    summary_path = static_results / "static_geometry_summary.json"
    sweeps_path = static_results / "age_loss_sweeps.csv"
    arm_c_path = static_results / "arm_c_identity_effects.csv"
    influence_path = static_results / "measurement_identity_influence.csv"
    events_path = static_results / "static_geometry_events.csv"
    required = [summary_path, sweeps_path, arm_c_path, influence_path, events_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing diagnostic inputs: " + ", ".join(missing))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sweeps = pd.read_csv(sweeps_path)
    arm_c = pd.read_csv(arm_c_path)
    influence = pd.read_csv(influence_path)
    events = pd.read_csv(events_path)

    # Figure D1: uniform age/loss response.  The endpoint p=1 is retained to
    # expose the observability cliff instead of smoothing it away.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    age = sweeps[sweeps["sweep"].eq("uniform_age")]
    loss = sweeps[sweeps["sweep"].eq("uniform_loss")]
    axes[0].plot(age["level"], age["lam_ratio"], marker="o", color="#0072B2")
    axes[0].set(xlabel="Uniform telemetry age (s)", ylabel=r"Retained $\kappa=\lambda_{\min}(G)/\lambda_{\min}(G_\infty)$")
    axes[0].set_ylim(-0.02, 1.03)
    axes[0].grid(alpha=0.25)
    axes[1].plot(loss["level"], loss["lam_ratio"], marker="o", color="#D55E00", label=r"$\kappa$")
    ax2 = axes[1].twinx()
    ax2.plot(loss["level"], loss["u_lmax"], marker="s", linestyle="--", color="#009E73", label=r"$u^{\lambda}$")
    axes[1].set(xlabel="Uniform packet-loss probability", ylabel=r"Retained $\kappa$")
    ax2.set_ylabel(r"Exposure $u^{\lambda}$")
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].grid(alpha=0.25)
    handles = axes[1].lines + ax2.lines
    axes[1].legend(handles, [line.get_label() for line in handles], loc="center left")
    fig.tight_layout()
    save_figure(fig, figures / "fig_diagnostic_age_loss_response")

    # Figure D2: Arm-C matched-count/age strata.  Log kappa exposes the mixture
    # of well-observed and collapsed configurations hidden by cardinality.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    arm_events = events[events["arm"].astype(str).eq("C")].copy()
    data = [
        np.clip(arm_events.loc[arm_events["stratum"].eq(stratum), "lam_ratio"].to_numpy(float), 1e-13, None)
        for stratum in sorted(arm_events["stratum"].dropna().astype(int).unique())
    ]
    axes[0].boxplot(data, tick_labels=["0", "1", "2"], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Arm-C matched stratum", ylabel=r"Retained $\kappa$ (log scale)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(arm_c["stratum"].astype(str), arm_c["near_unobservable_fraction"], color="#CC79A7")
    axes[1].set(xlabel="Arm-C matched stratum", ylabel="Near-unobservable fraction", ylim=(0, 1))
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures / "fig_diagnostic_matched_identity")

    # Figure D3: single-channel and cumulative weak-direction sensitivity.
    ranked = influence.assign(degradation=1.0 - influence["single_removal_lam_ratio"]).sort_values("degradation", ascending=False).head(12)
    cumulative = summary["identity_effect"]["top_group_removal_lam_ratio"]
    k = np.array(sorted(int(item) for item in cumulative), dtype=int)
    retained = np.array([float(cumulative[str(item)]) for item in k])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    axes[0].bar(np.arange(len(ranked)), ranked["degradation"], color="#56B4E9")
    axes[0].set_xticks(np.arange(len(ranked)), ranked["channel"].astype(int).astype(str), rotation=0)
    axes[0].set(xlabel="Telemetry channel", ylabel=r"Single-removal loss $1-\kappa$")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(k, np.clip(retained, 1e-13, None), marker="o", color="#D55E00")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Top weak-axis channels removed", ylabel=r"Retained $\kappa$ (log scale)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures / "fig_diagnostic_channel_influence")

    # Figure D4: the cardinality/exposure relationship, split by experimental
    # arm.  This is a protocol-identifiability check, not detector performance.
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    for arm, marker, color in (("C", "o", "#0072B2"), ("G", "s", "#D55E00"), ("T", "^", "#009E73")):
        block = events[events["arm"].astype(str).eq(arm)]
        ax.scatter(block["expected_availability"], np.clip(block["u_lmax"], 1e-10, None), s=9, alpha=0.42, marker=marker, color=color, label=f"Arm {arm}")
    ax.set_yscale("log")
    ax.set(xlabel="Expected telemetry availability", ylabel=r"Exposure $u^{\lambda}$ (log scale)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "fig_diagnostic_collinearity")

    table = arm_c[[
        "stratum", "events", "configured_b1", "configured_b2",
        "lam_ratio_min", "lam_ratio_median", "lam_ratio_max",
        "near_unobservable_fraction",
    ]].copy()
    table.to_csv(tables / "table_diagnostic_arm_c.csv", index=False)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Pre-campaign Arm-C geometry diagnostic at matched received fraction and mean age. These are mechanism checks, not confirmatory detector-performance results.}",
        r"\label{tab:diagnostic-arm-c}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Stratum & Events & $B_1$ & $B_2$ (s) & Median $\kappa$ & Near-unobs. \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{int(row.stratum)} & {int(row.events)} & {row.configured_b1:.3f} & "
            f"{row.configured_b2:.3f} & {row.lam_ratio_median:.3g} & "
            f"{100.0 * row.near_unobservable_fraction:.1f}\\% \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (latex / "table_diagnostic_arm_c.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    perfect = summary["perfect_information"]
    top_single_loss = float((1.0 - influence["single_removal_lam_ratio"]).max())
    top_eight_ratio = float(summary["identity_effect"]["top_group_removal_lam_ratio"]["8"])
    availability_spearman = float(
        summary["identity_effect"]["spearman_expected_availability_vs_u_all"]
    )
    diagnostic_tex = rf"""% Auto-generated diagnostic evidence. Safe to cite only as mechanism characterization.
\subsection{{Pre-campaign information-geometry diagnostics}}
The feeder has {summary['dimensions']['measurements']} measurements and
{summary['dimensions']['states']} states.  Under perfect information,
$\lambda_{{\min}}(G_\infty)={perfect['lambda_min']:.6g}$ and the condition
number is {perfect['condition_number']:.4g}.  The numerical implementation
returns $u^\lambda=0$ and $u^{{\mathrm{{tr}}}}=0$ at this limit, as required
by the reduction property.  These results characterize the mechanism and do
not constitute confirmatory detection performance. Removing the most
influential telemetry channel reduces weakest-direction information by
{100.0 * top_single_loss:.1f}\%, while removing the top eight leaves only
{top_eight_ratio:.3g} of the perfect-information floor. Across all static
patterns, expected availability and exposure have Spearman correlation
{availability_spearman:.3f}; this collinearity is reported because it can
make a null incremental-geometry result protocol-induced rather than
substantive.

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{paper1_generated/figures/fig_diagnostic_age_loss_response.pdf}}
\caption{{Static response to uniform age and loss. The abrupt endpoint at total loss is the observability cliff; smooth degradation away from the cliff is weak for this feeder.}}
\label{{fig:diagnostic-age-loss}}
\end{{figure}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{paper1_generated/figures/fig_diagnostic_matched_identity.pdf}}
\caption{{Arm-C matched-count/age diagnostic. Measurement identity produces orders-of-magnitude variation in retained weak-direction information within the same trivial communication stratum.}}
\label{{fig:diagnostic-identity}}
\end{{figure}}

\input{{paper1_generated/latex/table_diagnostic_arm_c.tex}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{paper1_generated/figures/fig_diagnostic_channel_influence.pdf}}
\caption{{Measurement-identity sensitivity of the weakest information direction. Left: single-channel removal loss. Right: retained information after cumulative removal of the most influential channels.}}
\label{{fig:diagnostic-channel-influence}}
\end{{figure*}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{paper1_generated/figures/fig_diagnostic_collinearity.pdf}}
\caption{{Static availability--exposure relationship by experimental arm. The strong but imperfect association motivates reporting collinearity and matched-pair fraction before interpreting F3.}}
\label{{fig:diagnostic-collinearity}}
\end{{figure}}
"""
    (latex / "paper1_diagnostic_results.tex").write_text(diagnostic_tex, encoding="utf-8")

    manifest = {
        "schema": "paper1.publication.diagnostics.v1",
        "status": "diagnostic_not_confirmatory_performance",
        "inputs": {path.name: sha256(path) for path in required},
        "facts": {
            "arm_c_events": int(summary["identity_effect"]["arm_c_events"]),
            "spearman_availability_vs_u": float(summary["identity_effect"]["spearman_expected_availability_vs_u_all"]),
            "condition_number": float(perfect["condition_number"]),
            "lambda_min_perfect": float(perfect["lambda_min"]),
        },
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_contracts(design_path: Path, threshold_path: Path, seeds: list[int], allow_partial: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    if design.get("schema") != "twin.factor.design.paper1.v4":
        raise ValueError("the design is not the frozen Paper-1 v4 contract")
    if sha256(design_path) != DESIGN_SHA256:
        raise ValueError("Paper-1 v4 design hash mismatch")
    if thresholds.get("schema") != "paper1.matched_far.thresholds.v1":
        raise ValueError("wrong matched-FAR threshold schema")
    if thresholds.get("source_split") != "calibration_only":
        raise ValueError("thresholds are not calibration-only")
    if float(thresholds.get("target_far", -1.0)) != 0.01:
        raise ValueError("target FAR is not the frozen 0.01")
    if thresholds.get("quantile_method") != "higher":
        raise ValueError("wrong threshold quantile convention")
    if not {"s", "chi2"}.issubset(thresholds.get("thresholds", {})):
        raise ValueError("frozen thresholds must include s and chi2")
    if not allow_partial:
        if seeds != list(range(2, 32)):
            raise ValueError("final analysis requires seed indices 2..31 exactly")
        if design["seed_policy"]["confirmatory_seed_indices"] != seeds:
            raise ValueError("seed list disagrees with the frozen design")
    return design, thresholds


def validate_cell_provenance(
    run_dir: Path,
    *,
    seed: int,
    bandwidth: str,
    design_hash: str,
    threshold_hash: str,
    expected_bandwidth_cap: float,
    expected_calibration_hash: str,
    expected_gamma_hash: str,
    allow_partial: bool,
) -> dict[str, Any]:
    record_path = run_dir / "cell_record.paper1.v4.json"
    if not record_path.exists():
        if allow_partial:
            return {}
        raise FileNotFoundError(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    required = {
        "schema": "twin.factor.cell.record.paper1.v4",
        "status": "complete",
        "seed_index": seed,
        "physical_seed": 81000 + seed,
        "bandwidth_level": bandwidth,
        "factor_design_sha256": design_hash,
        "matched_far_threshold_sha256": threshold_hash,
        "calibration_sha256": expected_calibration_hash,
        "gamma_sha256": expected_gamma_hash,
        "qualification_seed_excluded": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"{run_dir.name}: cell provenance mismatch for {key}")
    metas: dict[str, dict[str, Any]] = {}
    for service in ("power", "net", "twin", "oracle"):
        meta_path = run_dir / service / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if meta.get("status") != "complete":
            raise ValueError(f"{run_dir.name}: {service} status is not complete")
        metas[service] = meta
    net = metas["net"]
    twin = metas["twin"]
    if net.get("bandwidth_level") != bandwidth or not math.isclose(
        float(net.get("bandwidth_cap_bps", float("nan"))),
        expected_bandwidth_cap,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{run_dir.name}: network bandwidth metadata mismatch")
    if int(net.get("seed", -1)) != 22000 + seed:
        raise ValueError(f"{run_dir.name}: network seed metadata mismatch")
    factor = twin.get("factor_design", {})
    if factor.get("bandwidth_level") != bandwidth or not math.isclose(
        float(factor.get("bandwidth_cap_bps", float("nan"))),
        expected_bandwidth_cap,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{run_dir.name}: twin bandwidth metadata mismatch")
    truth_hash = str(record.get("truth_sha256", "")).lower()
    for service in ("power", "oracle"):
        recorded = str(
            metas[service].get("inputs", {}).get("truth", {}).get("sha256", "")
        ).lower()
        if recorded and recorded != truth_hash:
            raise ValueError(f"{run_dir.name}: {service} truth hash mismatch")
    return record


def q_tie_fraction(y: np.ndarray, values: np.ndarray) -> float:
    """Fraction of positive-negative pairs tied on a discrete control score."""
    y = np.asarray(y, bool)
    values = np.asarray(values)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    frame = pd.DataFrame({"y": y, "value": values})
    grouped = frame.groupby("value", dropna=False)["y"].agg(["sum", "count"])
    tied = float(np.sum(grouped["sum"] * (grouped["count"] - grouped["sum"])))
    return tied / (positives * negatives)


def aggregate_metric_rows(
    frame: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    y = frame["label"].astype(bool).to_numpy()
    positives, negatives = int(y.sum()), int((~y).sum())
    rows: list[dict[str, Any]] = []
    for metric, threshold in thresholds.items():
        if metric not in frame:
            continue
        score = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
        finite = np.isfinite(score)
        if not finite.all():
            continue
        alarm = score > threshold
        tp = int(np.sum(alarm & y)); fp = int(np.sum(alarm & ~y))
        row = dict(group)
        row.update(
            {
                "metric": metric,
                "events": int(len(frame)),
                "positives": positives,
                "negatives": negatives,
                "auc": empirical_auc(y, score),
                "tp": tp,
                "fp": fp,
                "tn": int(negatives - fp),
                "fn": int(positives - tp),
                "recall": float(tp / positives) if positives else float("nan"),
                "false_alarm_rate": float(fp / negatives) if negatives else float("nan"),
                "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            }
        )
        rows.append(row)
    return rows


def analyze_confirmatory(
    *,
    runs_root: Path,
    design_path: Path,
    threshold_path: Path,
    output: Path,
    seeds: list[int],
    draws: int,
    allow_partial: bool,
) -> None:
    design, threshold_contract = validate_contracts(design_path, threshold_path, seeds, allow_partial)
    thresholds = {name: float(record["threshold"]) for name, record in threshold_contract["thresholds"].items()}
    expected = {(seed, bandwidth) for seed in seeds for bandwidth in BANDWIDTH_ORDER}
    found: set[tuple[int, str]] = set()
    event_blocks: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    design_hash, threshold_hash = sha256(design_path), sha256(threshold_path)
    cap_by_bandwidth = {
        item["id"]: float(item["bandwidth_cap_bps"])
        for item in design["bandwidth_levels"]
    }

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        seed = int(match.group("seed")); bandwidth = match.group("bandwidth")
        if (seed, bandwidth) not in expected:
            continue
        if (seed, bandwidth) in found:
            raise ValueError(f"duplicate campaign cell for seed {seed}, {bandwidth}")
        validate_cell_provenance(
            run_dir,
            seed=seed,
            bandwidth=bandwidth,
            design_hash=design_hash,
            threshold_hash=threshold_hash,
            expected_bandwidth_cap=cap_by_bandwidth[bandwidth],
            expected_calibration_hash=design["frozen_inputs"]["calibration_sha256"],
            expected_gamma_hash=design["frozen_inputs"]["gamma_sha256"],
            allow_partial=allow_partial,
        )
        found.add((seed, bandwidth))

        twin_event = read_score_table(run_dir / "twin" / "scores_events.parquet")
        oracle_event = read_score_table(run_dir / "oracle" / "oracle_events.parquet")
        if not allow_partial and (len(twin_event) != 1100 or len(oracle_event) != 1100):
            raise AssertionError(f"{run_dir.name}: expected 1100 event rows")
        oracle_columns = [column for column in ("event_id", "label", "is_nominal", "drift_family", "d") if column in oracle_event]
        merged = twin_event.merge(
            oracle_event[oracle_columns], on="event_id", how="inner", validate="one_to_one"
        )
        if "label" not in merged or len(merged) != len(twin_event):
            raise AssertionError(f"{run_dir.name}: event/label merge failed")
        merged["seed_index"] = seed
        merged["physical_seed"] = 81000 + seed
        merged["bandwidth_level"] = bandwidth
        y = merged["label"].astype(bool).to_numpy()
        chi2_alarm = merged["chi2"].to_numpy(float) > thresholds["chi2"]
        merged["residual_silent"] = y & ~chi2_alarm
        abstain = np.zeros(len(merged), dtype=bool)
        if "held_any" in merged:
            abstain |= merged["held_any"].astype(bool).to_numpy()
        if "residual_available" in merged:
            abstain |= ~merged["residual_available"].astype(bool).to_numpy()
        if "n_rx_telemetry" in merged:
            abstain |= merged["n_rx_telemetry"].astype(int).eq(0).to_numpy()
        merged["abstain"] = abstain

        base_group = {"seed_index": seed, "bandwidth_level": bandwidth, "arm": "ALL", "regime": "ALL", "drift_family": "ALL"}
        rows = aggregate_metric_rows(merged, thresholds=thresholds, group=base_group)
        silence_rate = float(merged.loc[y, "residual_silent"].mean()) if y.any() else float("nan")
        for row in rows:
            row["residual_silence_rate"] = silence_rate
            row["abstention_rate"] = float(abstain.mean())
            row["mean_b1"] = float(pd.to_numeric(merged.get("b1"), errors="coerce").mean()) if "b1" in merged else float("nan")
            row["mean_b2"] = float(pd.to_numeric(merged.get("b2"), errors="coerce").mean()) if "b2" in merged else float("nan")
        cell_rows.extend(rows)

        group_columns = [column for column in ("arm", "regime", "drift_family") if column in merged]
        for column in group_columns:
            merged[column] = merged[column].fillna("NA").astype(str)
        # Emit marginal as well as joint condition summaries.  In particular,
        # Arm G must be available as a pooled F3 verdict; a joint-only table
        # would fragment it into sparse regime/family cells.
        group_specs: list[list[str]] = []
        for candidate in (["arm"], ["regime"], ["drift_family"], ["arm", "regime"], ["arm", "drift_family"], group_columns):
            spec = [column for column in candidate if column in group_columns]
            if spec and spec not in group_specs:
                group_specs.append(spec)
        for spec in group_specs:
            for keys, block in merged.groupby(spec, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                group = {"seed_index": seed, "bandwidth_level": bandwidth, "arm": "ALL", "regime": "ALL", "drift_family": "ALL"}
                group.update(dict(zip(spec, keys)))
                block_rows = aggregate_metric_rows(block, thresholds=thresholds, group=group)
                block_y = block["label"].astype(bool).to_numpy()
                block_silence = float(block.loc[block_y, "residual_silent"].mean()) if block_y.any() else float("nan")
                for row in block_rows:
                    row["residual_silence_rate"] = block_silence
                    row["abstention_rate"] = float(block["abstain"].mean())
                condition_rows.extend(block_rows)

        twin_step = read_score_table(run_dir / "twin" / "scores.parquet")
        oracle_step = read_score_table(run_dir / "oracle" / "oracle_scores.parquet")
        key = "step_index" if "step_index" in twin_step and "step_index" in oracle_step else None
        if key is None:
            twin_step = twin_step.reset_index(drop=True); oracle_step = oracle_step.reset_index(drop=True)
            twin_step["_row"] = np.arange(len(twin_step)); oracle_step["_row"] = np.arange(len(oracle_step)); key = "_row"
        oracle_step_columns = [key, "event_id"] + [column for column in ("label", "d") if column in oracle_step]
        step = twin_step.merge(oracle_step[oracle_step_columns], on=[key, "event_id"], how="inner", validate="one_to_one")
        if not allow_partial and len(step) != 13200:
            raise AssertionError(f"{run_dir.name}: expected 13200 step rows")
        silent_map = merged.set_index("event_id")["residual_silent"].astype(bool).to_dict()
        event_label_map = merged.set_index("event_id")["label"].astype(bool).to_dict()
        event_context = merged.set_index("event_id")
        for event_id, block in step.groupby("event_id", sort=False):
            if not bool(event_label_map.get(event_id, False)):
                continue
            step_positive = block["label"].astype(bool).to_numpy() if "label" in block else np.ones(len(block), dtype=bool)
            onset = int(np.flatnonzero(step_positive)[0]) if step_positive.any() else 0
            for metric, threshold in thresholds.items():
                if metric not in block:
                    continue
                score = pd.to_numeric(block[metric], errors="coerce").to_numpy(float)[onset:]
                hits = np.flatnonzero(score > threshold)
                latency_rows.append(
                    {
                        "seed_index": seed,
                        "bandwidth_level": bandwidth,
                        "metric": metric,
                        "event_id": int(event_id),
                        "arm": str(event_context.at[event_id, "arm"]) if "arm" in event_context else "NA",
                        "regime": str(event_context.at[event_id, "regime"]) if "regime" in event_context else "NA",
                        "drift_family": str(event_context.at[event_id, "drift_family"]) if "drift_family" in event_context else "NA",
                        "residual_silent": bool(silent_map.get(event_id, False)),
                        "detected": bool(len(hits)),
                        "latency_steps": int(hits[0]) if len(hits) else np.nan,
                    }
                )

        net_meta = json.loads((run_dir / "net" / "meta.json").read_text(encoding="utf-8-sig"))
        counts = net_meta.get("counts", {})
        received = int(counts.get("received", 0)); delivered = int(counts.get("delivered", 0))
        network_rows.append(
            {
                "seed_index": seed,
                "bandwidth_level": bandwidth,
                "packets_received": received,
                "packets_delivered": delivered,
                "realized_drop_fraction": float((received - delivered) / received) if received else float("nan"),
                "dropped_random": int(counts.get("dropped_random", 0)),
                "dropped_starved": int(counts.get("dropped_starved", 0)),
                "dropped_queue": int(counts.get("dropped_queue", 0)),
                "mean_b1": float(merged["b1"].mean()) if "b1" in merged else float("nan"),
                "median_b1": float(merged["b1"].median()) if "b1" in merged else float("nan"),
                "mean_b2": float(merged["b2"].mean()) if "b2" in merged else float("nan"),
                "p90_b2": float(merged["b2"].quantile(0.9)) if "b2" in merged else float("nan"),
                "held_rate": float(abstain.mean()),
            }
        )
        event_blocks.append(merged)

    missing = sorted(expected - found)
    if missing and not allow_partial:
        preview = ", ".join(f"s{seed:03d}/{bw}" for seed, bw in missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} confirmatory cells; first: {preview}")
    if not event_blocks:
        raise FileNotFoundError("no Paper-1 campaign cells found")

    output.mkdir(parents=True, exist_ok=True)
    cell = pd.DataFrame(cell_rows)
    conditions = pd.DataFrame(condition_rows)
    latency = pd.DataFrame(latency_rows)
    network = pd.DataFrame(network_rows)
    events = pd.concat(event_blocks, ignore_index=True)
    cell.to_csv(output / "paper1_seed_bandwidth_metrics.csv", index=False)
    conditions.to_csv(output / "paper1_condition_metrics.csv", index=False)
    latency.to_csv(output / "paper1_latency_events.csv", index=False)
    network.to_csv(output / "paper1_network_conditions.csv", index=False)

    rng = np.random.default_rng(51031)
    summary_rows: list[dict[str, Any]] = []
    outcomes = ["auc", "recall", "false_alarm_rate", "residual_silence_rate", "abstention_rate"]
    for (bandwidth, metric), block in cell.groupby(["bandwidth_level", "metric"]):
        for outcome in outcomes:
            values = pd.to_numeric(block[outcome], errors="coerce").to_numpy(float)
            low, high = bootstrap_mean_ci(values, draws=draws, rng=rng)
            summary_rows.append(
                {
                    "bandwidth_level": bandwidth,
                    "metric": metric,
                    "outcome": outcome,
                    "seeds": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "paper1_cluster_summary.csv", index=False)

    condition_summary_rows: list[dict[str, Any]] = []
    if not conditions.empty:
        condition_outcomes = ["auc", "recall", "false_alarm_rate", "residual_silence_rate", "abstention_rate"]
        for keys, block in conditions.groupby(["bandwidth_level", "arm", "regime", "drift_family", "metric"], dropna=False):
            bandwidth, arm, regime, drift_family, metric = keys
            for outcome in condition_outcomes:
                values = pd.to_numeric(block[outcome], errors="coerce").to_numpy(float)
                low, high = bootstrap_mean_ci(values, draws=draws, rng=rng)
                condition_summary_rows.append(
                    {
                        "bandwidth_level": bandwidth,
                        "arm": arm,
                        "regime": regime,
                        "drift_family": drift_family,
                        "metric": metric,
                        "outcome": outcome,
                        "seeds": int(np.isfinite(values).sum()),
                        "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                        "median": float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan"),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    condition_summary = pd.DataFrame(condition_summary_rows)
    condition_summary.to_csv(output / "paper1_condition_cluster_summary.csv", index=False)

    regime_summary_rows: list[dict[str, Any]] = []
    if not conditions.empty:
        regime_only = conditions[
            conditions["arm"].eq("ALL")
            & conditions["regime"].isin(["ample", "moderate", "severe"])
            & conditions["drift_family"].eq("ALL")
        ]
        for (regime, metric, outcome), block in (
            regime_only.melt(
                id_vars=["seed_index", "bandwidth_level", "regime", "metric"],
                value_vars=["auc", "recall", "false_alarm_rate", "residual_silence_rate", "abstention_rate"],
                var_name="outcome",
                value_name="value",
            ).groupby(["regime", "metric", "outcome"], dropna=False)
        ):
            # Collapse five bandwidth cells inside each physical seed before
            # bootstrap resampling; bandwidth replicates are not independent.
            per_seed = block.groupby("seed_index")["value"].mean().to_numpy(float)
            finite = per_seed[np.isfinite(per_seed)]
            low, high = bootstrap_mean_ci(finite, draws=draws, rng=rng)
            regime_summary_rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "outcome": outcome,
                    "seeds": int(len(finite)),
                    "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    pd.DataFrame(regime_summary_rows).to_csv(output / "paper1_regime_cluster_summary.csv", index=False)

    contrast_rows: list[dict[str, Any]] = []
    comparators = [metric for metric in ("chi2", "sB1", "sB2", "s_gated_lmax") if metric in set(cell["metric"])]
    for outcome in ("auc", "recall", "false_alarm_rate"):
        for comparator in comparators:
            raw: list[dict[str, Any]] = []; pvalues: list[float] = []
            for bandwidth in BANDWIDTH_ORDER:
                block = cell[cell["bandwidth_level"].eq(bandwidth)]
                pivot = block.pivot(index="seed_index", columns="metric", values=outcome)
                if "s" not in pivot or comparator not in pivot:
                    continue
                paired = pivot[["s", comparator]].dropna()
                difference = paired["s"].to_numpy(float) - paired[comparator].to_numpy(float)
                low, high = bootstrap_mean_ci(difference, draws=draws, rng=rng)
                pvalue = safe_wilcoxon(difference)
                pvalues.append(pvalue)
                raw.append(
                    {
                        "outcome": outcome,
                        "metric": "s",
                        "comparator": comparator,
                        "bandwidth_level": bandwidth,
                        "paired_seeds": int(len(difference)),
                        "mean_paired_difference": float(np.mean(difference)),
                        "median_paired_difference": float(np.median(difference)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wilcoxon_p_raw": pvalue,
                    }
                )
            for row, adjusted in zip(raw, holm(pvalues)):
                row["wilcoxon_p_holm"] = adjusted
                contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "paper1_paired_metric_contrasts.csv", index=False)

    arm_contrast_rows: list[dict[str, Any]] = []
    arm_only = conditions[
        conditions["arm"].ne("ALL")
        & conditions["regime"].eq("ALL")
        & conditions["drift_family"].eq("ALL")
    ] if not conditions.empty else pd.DataFrame()
    if not arm_only.empty:
        for arm in sorted(arm_only["arm"].unique()):
            for outcome in ("auc", "recall", "false_alarm_rate"):
                for comparator in [item for item in ("chi2", "sB1", "sB2", "s_gated_lmax") if item in set(arm_only["metric"])]:
                    raw = []; pvalues = []
                    for bandwidth in BANDWIDTH_ORDER:
                        block = arm_only[arm_only["arm"].eq(arm) & arm_only["bandwidth_level"].eq(bandwidth)]
                        pivot = block.pivot(index="seed_index", columns="metric", values=outcome)
                        if "s" not in pivot or comparator not in pivot:
                            continue
                        paired = pivot[["s", comparator]].dropna()
                        difference = paired["s"].to_numpy(float) - paired[comparator].to_numpy(float)
                        low, high = bootstrap_mean_ci(difference, draws=draws, rng=rng)
                        pvalue = safe_wilcoxon(difference)
                        pvalues.append(pvalue)
                        raw.append(
                            {
                                "arm": arm,
                                "outcome": outcome,
                                "metric": "s",
                                "comparator": comparator,
                                "bandwidth_level": bandwidth,
                                "paired_seeds": int(len(difference)),
                                "mean_paired_difference": float(np.mean(difference)),
                                "median_paired_difference": float(np.median(difference)),
                                "ci95_low": low,
                                "ci95_high": high,
                                "wilcoxon_p_raw": pvalue,
                            }
                        )
                    for row, adjusted in zip(raw, holm(pvalues)):
                        row["wilcoxon_p_holm"] = adjusted
                        arm_contrast_rows.append(row)
    pd.DataFrame(arm_contrast_rows).to_csv(output / "paper1_arm_paired_contrasts.csv", index=False)

    latency_summary_rows: list[dict[str, Any]] = []
    for (bandwidth, metric, silent), block in latency.groupby(["bandwidth_level", "metric", "residual_silent"]):
        seed_summary = block.groupby("seed_index").agg(
            detection_fraction=("detected", "mean"),
            median_latency_detected=("latency_steps", "median"),
        ).reset_index()
        for outcome in ("detection_fraction", "median_latency_detected"):
            values = seed_summary[outcome].to_numpy(float)
            low, high = bootstrap_mean_ci(values, draws=draws, rng=rng)
            finite_values = values[np.isfinite(values)]
            latency_summary_rows.append(
                {
                    "bandwidth_level": bandwidth,
                    "metric": metric,
                    "residual_silent": bool(silent),
                    "outcome": outcome,
                    "mean": float(np.mean(finite_values)) if len(finite_values) else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                    "seeds": int(np.isfinite(values).sum()),
                }
            )
    pd.DataFrame(latency_summary_rows).to_csv(output / "paper1_latency_summary.csv", index=False)

    roc_rows: list[pd.DataFrame] = []
    for bandwidth in BANDWIDTH_ORDER:
        block = events[events["bandwidth_level"].eq(bandwidth)]
        for metric in [item for item in ("s", "chi2", "sB1", "sB2", "s_gated_lmax") if item in block]:
            points = roc_points(block["label"].astype(bool), block[metric])
            if not points.empty:
                points["bandwidth_level"] = bandwidth; points["metric"] = metric
                roc_rows.append(points)
    roc_frame = pd.concat(roc_rows, ignore_index=True) if roc_rows else pd.DataFrame(
        columns=["threshold", "fpr", "tpr", "bandwidth_level", "metric"]
    )
    roc_frame.to_csv(output / "paper1_roc_points.csv", index=False)

    collinearity_rows: list[dict[str, Any]] = []
    for (seed, bandwidth, arm), block in events.groupby(["seed_index", "bandwidth_level", "arm"], dropna=False):
        if "b1" not in block or "u_lmax" not in block:
            continue
        rho = float(spearmanr(block["b1"], block["u_lmax"], nan_policy="omit").statistic)
        y = block["label"].astype(bool).to_numpy()
        n_rx = block["n_rx_telemetry"].to_numpy() if "n_rx_telemetry" in block else np.rint(block["b1"].to_numpy(float) * 45).astype(int)
        q_b1 = q_tie_fraction(y, n_rx)
        age_bin = np.round(block["b2"].to_numpy(float), 3) if "b2" in block else np.zeros(len(block))
        pair_key = np.array([f"{int(count)}|{age:.3f}" for count, age in zip(n_rx, age_bin)])
        q_b1b2 = q_tie_fraction(y, pair_key)
        collinearity_rows.append(
            {
                "seed_index": seed,
                "bandwidth_level": bandwidth,
                "arm": arm,
                "spearman_b1_u_lmax": rho,
                "q_b1_tied_pos_neg_pairs": q_b1,
                "auc_b1_ceiling": 1.0 - q_b1 / 2.0 if math.isfinite(q_b1) else float("nan"),
                "q_b1_b2_tied_pos_neg_pairs": q_b1b2,
            }
        )
    pd.DataFrame(collinearity_rows).to_csv(output / "paper1_collinearity_and_q.csv", index=False)

    # FAR and miss-rate transport versus realized loss/age.  Fixed bins are
    # predeclared here; aggregation first occurs within seed so five bandwidth
    # cells from one physical realization are not treated as independent.
    communication_rows: list[dict[str, Any]] = []
    metric_names = [name for name in thresholds if name in events]
    dimension_specs = {
        "loss_proxy": (1.0 - events["b1"].to_numpy(float), [-1e-12, 0.10, 0.25, 0.50, 0.75, 1.000001], ["0-.10", ".10-.25", ".25-.50", ".50-.75", ".75-1"]),
        "mean_age": (events["b2"].to_numpy(float), [-1e-12, 0.5, 1.0, 2.0, 5.0, np.inf], ["0-.5", ".5-1", "1-2", "2-5", "5+"]),
    }
    for dimension, (values, edges, labels) in dimension_specs.items():
        tagged = events.copy()
        tagged["condition_value"] = values
        tagged["condition_bin"] = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=True)
        tagged = tagged[tagged["condition_bin"].notna()]
        for (seed, condition_bin), block in tagged.groupby(["seed_index", "condition_bin"], observed=True):
            y = block["label"].astype(bool).to_numpy()
            positives, negatives = int(y.sum()), int((~y).sum())
            for metric in metric_names:
                score = block[metric].to_numpy(float)
                alarm = score > thresholds[metric]
                communication_rows.append(
                    {
                        "seed_index": int(seed),
                        "dimension": dimension,
                        "condition_bin": str(condition_bin),
                        "condition_mean": float(block["condition_value"].mean()),
                        "metric": metric,
                        "events": int(len(block)),
                        "positives": positives,
                        "negatives": negatives,
                        "false_alarm_rate": float(np.sum(alarm & ~y) / negatives) if negatives else float("nan"),
                        "missed_drift_rate": float(np.sum(~alarm & y) / positives) if positives else float("nan"),
                    }
                )
    communication = pd.DataFrame(communication_rows)
    communication.to_csv(output / "paper1_communication_bin_metrics.csv", index=False)
    communication_summary_rows: list[dict[str, Any]] = []
    if not communication.empty:
        for (dimension, condition_bin, metric), block in communication.groupby(["dimension", "condition_bin", "metric"], observed=True):
            for outcome in ("false_alarm_rate", "missed_drift_rate"):
                values = block[outcome].to_numpy(float)
                finite = values[np.isfinite(values)]
                low, high = bootstrap_mean_ci(finite, draws=draws, rng=rng)
                communication_summary_rows.append(
                    {
                        "dimension": dimension,
                        "condition_bin": condition_bin,
                        "condition_mean": float(block["condition_mean"].mean()),
                        "metric": metric,
                        "outcome": outcome,
                        "seeds": int(len(finite)),
                        "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    pd.DataFrame(communication_summary_rows).to_csv(output / "paper1_communication_bin_summary.csv", index=False)

    status = "confirmatory_complete" if not missing and seeds == list(range(2, 32)) else "partial_nonconfirmatory"
    manifest = {
        "schema": "paper1.publication.analysis.v1",
        "status": status,
        "cells_found": len(found),
        "cells_expected": len(expected),
        "confirmatory_seed_indices": seeds,
        "qualification_seed_excluded": 1 not in seeds,
        "design_sha256": design_hash,
        "threshold_sha256": threshold_hash,
        "threshold_source_split": threshold_contract["source_split"],
        "target_far": threshold_contract["target_far"],
        "bootstrap_draws": draws,
        "cluster": "physical seed",
        "multiplicity": "Holm across five bandwidth levels within each predeclared metric contrast",
        "roc_note": "pooled ROC curves are descriptive; inference uses paired seed-level AUC contrasts",
        "missing_cells": [{"seed_index": seed, "bandwidth_level": bw} for seed, bw in missing],
    }
    (output / "paper1_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_errorbar(ax: plt.Axes, frame: pd.DataFrame, metric: str, outcome: str, offset: float) -> None:
    block = frame[frame["metric"].eq(metric) & frame["outcome"].eq(outcome)].copy()
    block["order"] = block["bandwidth_level"].map({name: index for index, name in enumerate(BANDWIDTH_ORDER)})
    block = block.sort_values("order")
    x = block["order"].to_numpy(float) + offset
    y = block["mean"].to_numpy(float)
    err = np.vstack([y - block["ci95_low"].to_numpy(float), block["ci95_high"].to_numpy(float) - y])
    ax.errorbar(x, y, yerr=err, marker="o", capsize=2.5, label=METRIC_LABELS.get(metric, metric), color=METRIC_COLORS.get(metric))


def render_confirmatory(output: Path) -> None:
    manifest_path = output / "paper1_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "confirmatory_complete":
        raise RuntimeError("refusing final figures: analysis status is not confirmatory_complete")
    publication_style()
    figures = output / "figures"; tables = output / "tables"; latex = output / "latex"
    for directory in (figures, tables, latex):
        directory.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(output / "paper1_cluster_summary.csv")
    contrasts = pd.read_csv(output / "paper1_paired_metric_contrasts.csv")
    latency = pd.read_csv(output / "paper1_latency_events.csv")
    latency_summary = pd.read_csv(output / "paper1_latency_summary.csv")
    roc = pd.read_csv(output / "paper1_roc_points.csv")
    network = pd.read_csv(output / "paper1_network_conditions.csv")
    communication = pd.read_csv(output / "paper1_communication_bin_summary.csv")
    regime_summary = pd.read_csv(output / "paper1_regime_cluster_summary.csv")
    collinearity = pd.read_csv(output / "paper1_collinearity_and_q.csv")

    metrics = [metric for metric in ("s", "chi2", "sB1", "sB2", "s_gated_lmax") if metric in set(summary["metric"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    offsets = np.linspace(-0.18, 0.18, len(metrics))
    for metric, offset in zip(metrics, offsets):
        add_errorbar(axes[0], summary, metric, "recall", float(offset))
        add_errorbar(axes[1], summary, metric, "false_alarm_rate", float(offset))
    axes[0].set_ylabel("Recall at frozen threshold")
    axes[1].set_ylabel("False-alarm rate")
    axes[1].axhline(0.01, color="black", linestyle=":", linewidth=1, label="Calibration target")
    for ax in axes:
        ax.set_xticks(range(5), [BANDWIDTH_LABELS[name] for name in BANDWIDTH_ORDER], rotation=25, ha="right")
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_matched_far")

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.7), sharex=True, sharey=True)
    for index, bandwidth in enumerate(BANDWIDTH_ORDER):
        ax = axes.flat[index]
        for metric in [item for item in ("s", "chi2", "sB1") if item in set(roc["metric"])]:
            block = roc[roc["bandwidth_level"].eq(bandwidth) & roc["metric"].eq(metric)]
            ax.plot(block["fpr"], block["tpr"], color=METRIC_COLORS.get(metric), label=METRIC_LABELS.get(metric, metric))
        ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1)
        ax.set_title(BANDWIDTH_LABELS[bandwidth]); ax.grid(alpha=0.2)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False)
    fig.supxlabel("False-positive rate"); fig.supylabel("True-positive rate")
    fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_roc_by_bandwidth")

    silence = summary[(summary["metric"].eq("s")) & (summary["outcome"].eq("residual_silence_rate"))].copy()
    silence["order"] = silence["bandwidth_level"].map({name: index for index, name in enumerate(BANDWIDTH_ORDER)})
    silence = silence.sort_values("order")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = silence["mean"].to_numpy(float)
    ax.errorbar(range(5), y, yerr=np.vstack([y - silence["ci95_low"], silence["ci95_high"] - y]), marker="o", color="#D55E00", capsize=3)
    ax.set_xticks(range(5), [BANDWIDTH_LABELS[name] for name in BANDWIDTH_ORDER], rotation=25, ha="right")
    ax.set(ylabel=r"Residual-silence rate of $\chi^2$", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_residual_silence")

    silent_latency = latency[latency["residual_silent"].astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True)
    selected_bw = ["bw00_floor", "bw02_100kbps"]
    for ax, bandwidth in zip(axes, selected_bw):
        block_bw = silent_latency[silent_latency["bandwidth_level"].eq(bandwidth)]
        for metric in [item for item in ("s", "sB1", "chi2") if item in set(block_bw["metric"])]:
            block = block_bw[block_bw["metric"].eq(metric)]
            values = block["latency_steps"].to_numpy(float)
            curve = [float(np.mean(np.isfinite(values) & (values <= deadline))) for deadline in range(12)]
            ax.step(range(12), curve, where="post", label=METRIC_LABELS.get(metric, metric), color=METRIC_COLORS.get(metric))
        ax.set_title(BANDWIDTH_LABELS[bandwidth]); ax.set_xlabel("Steps after oracle onset"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction detected by deadline")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_silent_latency")

    network_summary = network.groupby("bandwidth_level").agg(
        mean_b1=("mean_b1", "mean"),
        mean_b2=("mean_b2", "mean"),
        realized_drop_fraction=("realized_drop_fraction", "mean"),
        held_rate=("held_rate", "mean"),
    ).reindex(BANDWIDTH_ORDER).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    axes[0].plot(range(5), network_summary["mean_b1"], marker="o", color="#0072B2")
    axes[0].set_ylabel("Mean received fraction")
    axes[1].plot(range(5), network_summary["mean_b2"], marker="o", color="#D55E00")
    axes[1].set_ylabel("Mean telemetry age (s)")
    axes[2].plot(range(5), network_summary["realized_drop_fraction"], marker="o", color="#009E73", label="Drop")
    axes[2].plot(range(5), network_summary["held_rate"], marker="s", color="#CC79A7", label="Abstain/hold")
    axes[2].set_ylabel("Fraction"); axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(range(5), ["Floor", "10k", "100k", "1M", "Oracle"], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_network_conditions")

    # The geometry test is not interpretable until availability--exposure
    # collinearity and the matched-pair fraction q are reported.  Summarize
    # the realistic Arm-G values before rendering any F3 comparison.
    arm_g_col = collinearity[collinearity["arm"].eq("G")].copy()
    col_summary_rows: list[dict[str, Any]] = []
    col_rng = np.random.default_rng(20260831)
    for bandwidth in BANDWIDTH_ORDER:
        block = arm_g_col[arm_g_col["bandwidth_level"].eq(bandwidth)]
        row: dict[str, Any] = {"bandwidth_level": bandwidth}
        for source, prefix in (
            ("spearman_b1_u_lmax", "spearman"),
            ("q_b1_tied_pos_neg_pairs", "q_b1"),
            ("q_b1_b2_tied_pos_neg_pairs", "q_b1_b2"),
        ):
            values = block[source].to_numpy(float)
            values = values[np.isfinite(values)]
            low, high = bootstrap_mean_ci(values, draws=2000, rng=col_rng)
            row[prefix] = float(np.mean(values)) if len(values) else float("nan")
            row[prefix + "_ci95_low"] = low
            row[prefix + "_ci95_high"] = high
        row["auc_b1_ceiling"] = 1.0 - row["q_b1"] / 2.0
        col_summary_rows.append(row)
    col_summary = pd.DataFrame(col_summary_rows)
    col_summary.to_csv(output / "paper1_collinearity_q_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.6, 2.75))
    x = np.arange(len(BANDWIDTH_ORDER))
    for source, label, color, marker in (
        ("spearman", r"Spearman$(B_1,u)$", "#0072B2", "o"),
        ("q_b1", r"Matched-pair fraction $q$", "#D55E00", "s"),
    ):
        y = col_summary[source].to_numpy(float)
        low = col_summary[source + "_ci95_low"].to_numpy(float)
        high = col_summary[source + "_ci95_high"].to_numpy(float)
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker=marker,
                    capsize=2.5, color=color, label=label)
    ax.axhline(0.95, color="#777777", linestyle=":", linewidth=1,
               label=r"$|\rho|=0.95$ warning")
    ax.set_xticks(x, [BANDWIDTH_LABELS[name] for name in BANDWIDTH_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Arm-G diagnostic")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "fig_confirmatory_collinearity_q")

    col_lines = [
        r"\begin{table}[!t]",
        r"\caption{Arm-G pre-interpretation check. Collinearity is reported before the incremental geometry test; $q$ is the B1-tied positive--negative pair fraction and implies the displayed B1 AUC ceiling.}",
        r"\label{tab:collinearity-q}", r"\centering\footnotesize",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Bandwidth & Spearman$(B_1,u)$ & $q$ & B1 ceiling \\", r"\midrule",
    ]
    for _, row in col_summary.iterrows():
        col_lines.append(
            f"{tex_escape(BANDWIDTH_LABELS[row['bandwidth_level']])} & "
            f"{fmt(row['spearman'])} & {fmt(row['q_b1'])} & "
            f"{fmt(row['auc_b1_ceiling'])} \\\\"
        )
    col_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (latex / "table_confirmatory_collinearity_q.tex").write_text(
        "\n".join(col_lines) + "\n", encoding="utf-8"
    )

    if not communication.empty:
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), sharey="row")
        for column, dimension in enumerate(("loss_proxy", "mean_age")):
            dimension_block = communication[communication["dimension"].eq(dimension)]
            for row, outcome in enumerate(("false_alarm_rate", "missed_drift_rate")):
                ax = axes[row, column]
                for metric in [item for item in ("s", "chi2", "sB1") if item in set(dimension_block["metric"])]:
                    block = dimension_block[dimension_block["metric"].eq(metric) & dimension_block["outcome"].eq(outcome)].sort_values("condition_mean")
                    x = block["condition_mean"].to_numpy(float); y = block["mean"].to_numpy(float)
                    err = np.vstack([y - block["ci95_low"].to_numpy(float), block["ci95_high"].to_numpy(float) - y])
                    ax.errorbar(x, y, yerr=err, marker="o", capsize=2.5, color=METRIC_COLORS.get(metric), label=METRIC_LABELS.get(metric, metric))
                ax.grid(alpha=0.25)
                if column == 0:
                    ax.set_ylabel("False-alarm rate" if row == 0 else "Missed-drift rate")
                if row == 1:
                    ax.set_xlabel("Realized loss proxy $1-B_1$" if dimension == "loss_proxy" else "Mean telemetry age (s)")
        axes[0, 0].axhline(0.01, color="#777777", linestyle=":", linewidth=1)
        axes[0, 1].axhline(0.01, color="#777777", linestyle=":", linewidth=1)
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, figures / "fig_confirmatory_far_miss_vs_condition")

    condition_summary_path = output / "paper1_condition_cluster_summary.csv"
    arm_contrast_path = output / "paper1_arm_paired_contrasts.csv"
    if condition_summary_path.exists() and arm_contrast_path.exists():
        condition_summary = pd.read_csv(condition_summary_path)
        arm_contrasts = pd.read_csv(arm_contrast_path)
        arm_g = condition_summary[
            condition_summary["arm"].eq("G")
            & condition_summary["regime"].eq("ALL")
            & condition_summary["drift_family"].eq("ALL")
            & condition_summary["outcome"].eq("auc")
        ]
        if not arm_g.empty:
            fig, ax = plt.subplots(figsize=(3.6, 2.75))
            f3_metrics = [metric for metric in ("s", "chi2", "sB1", "sB2", "s_gated_lmax") if metric in set(arm_g["metric"])]
            offsets = np.linspace(-0.18, 0.18, len(f3_metrics))
            for metric, offset in zip(f3_metrics, offsets):
                block = arm_g[arm_g["metric"].eq(metric)].copy()
                block["order"] = block["bandwidth_level"].map({name: index for index, name in enumerate(BANDWIDTH_ORDER)})
                block = block.sort_values("order")
                y = block["mean"].to_numpy(float)
                err = np.vstack([y - block["ci95_low"].to_numpy(float), block["ci95_high"].to_numpy(float) - y])
                ax.errorbar(block["order"].to_numpy(float) + float(offset), y, yerr=err, marker="o", capsize=2.5, label=METRIC_LABELS.get(metric, metric), color=METRIC_COLORS.get(metric))
            ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
            ax.set_xticks(range(5), [BANDWIDTH_LABELS[name] for name in BANDWIDTH_ORDER], rotation=25, ha="right")
            ax.set(ylabel="Arm-G seed-level AUC", ylim=(0, 1))
            ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            save_figure(fig, figures / "fig_confirmatory_f3_arm_g")

            f3_lines = [
                r"\begin{table*}[!t]",
                r"\caption{Arm-G information-geometry test. The full metric is compared with residual-plus-B1 at the same calibration-frozen FAR rule; differences are paired by seed.}",
                r"\label{tab:ablation}", r"\centering\footnotesize",
                r"\begin{tabular}{lrrrrr}", r"\toprule",
                r"Bandwidth & Full AUC & Residual+B1 AUC & Paired difference & 95\% CI & Holm $p$ \\", r"\midrule",
            ]
            arm_g_pivot = arm_g.pivot(index="bandwidth_level", columns="metric", values="mean").reindex(BANDWIDTH_ORDER)
            for bandwidth in BANDWIDTH_ORDER:
                test = arm_contrasts[
                    arm_contrasts["arm"].eq("G")
                    & arm_contrasts["outcome"].eq("auc")
                    & arm_contrasts["comparator"].eq("sB1")
                    & arm_contrasts["bandwidth_level"].eq(bandwidth)
                ]
                if len(test):
                    tr = test.iloc[0]
                    delta = fmt(tr["mean_paired_difference"]); ci = f"[{fmt(tr['ci95_low'])}, {fmt(tr['ci95_high'])}]"; p = fmt(tr["wilcoxon_p_holm"], 4)
                else:
                    delta = ci = p = "--"
                row = arm_g_pivot.loc[bandwidth]
                f3_lines.append(f"{tex_escape(BANDWIDTH_LABELS[bandwidth])} & {fmt(row.get('s', np.nan))} & {fmt(row.get('sB1', np.nan))} & {delta} & {ci} & {p} \\\\")
            f3_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
            (latex / "table_confirmatory_f3.tex").write_text("\n".join(f3_lines) + "\n", encoding="utf-8")

        silence_by_regime = regime_summary[
            regime_summary["regime"].isin(["ample", "moderate", "severe"])
            & regime_summary["metric"].eq("s")
            & regime_summary["outcome"].eq("residual_silence_rate")
        ]
        if not silence_by_regime.empty:
            # The within-run regime table is shown separately from the
            # cross-run bandwidth factor; rows are seed/bandwidth summaries.
            silence_lines = [
                r"\begin{table}[!t]", r"\caption{Residual-silence rate by within-run communication regime. Bandwidth cells are first collapsed within physical seed; intervals then resample seeds.}",
                r"\label{tab:silence}", r"\centering\footnotesize",
                r"\begin{tabular}{lrr}", r"\toprule", r"Regime & Silence rate & 95\% CI \\", r"\midrule",
            ]
            for regime in ("ample", "moderate", "severe"):
                block = silence_by_regime[silence_by_regime["regime"].eq(regime)]
                if len(block):
                    mean = float(block.iloc[0]["mean"]); low = float(block.iloc[0]["ci95_low"]); high = float(block.iloc[0]["ci95_high"])
                    silence_lines.append(
                        f"{regime.capitalize()} & {fmt(mean)} & [{fmt(low)}, {fmt(high)}]" + r" \\"
                    )
            silence_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            (latex / "table_confirmatory_silence.tex").write_text("\n".join(silence_lines) + "\n", encoding="utf-8")

    # LaTeX AUC table (seed-clustered means and paired tests are kept distinct).
    auc = summary[summary["outcome"].eq("auc")]
    auc_table = auc.pivot(index="bandwidth_level", columns="metric", values="mean").reindex(BANDWIDTH_ORDER)
    auc_table.to_csv(tables / "table_confirmatory_auc.csv")
    lines = [
        r"\begin{table*}[!t]", r"\caption{Confirmatory event-level discrimination across 30 unseen physical seeds. Values are mean seed-level AUCs; inference uses paired seed contrasts.}",
        r"\label{tab:auc}", r"\centering\footnotesize", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Bandwidth & Full & $\chi^2$ & Residual+B1 & $\Delta$Full--$\chi^2$ & 95\% CI & Holm $p$ \\", r"\midrule",
    ]
    for bandwidth in BANDWIDTH_ORDER:
        row = auc_table.loc[bandwidth]
        test = contrasts[(contrasts["outcome"].eq("auc")) & (contrasts["comparator"].eq("chi2")) & (contrasts["bandwidth_level"].eq(bandwidth))]
        if len(test):
            test_row = test.iloc[0]
            delta = fmt(test_row["mean_paired_difference"]); ci = f"[{fmt(test_row['ci95_low'])}, {fmt(test_row['ci95_high'])}]"; p = fmt(test_row["wilcoxon_p_holm"], 4)
        else:
            delta = ci = p = "--"
        lines.append(
            f"{tex_escape(BANDWIDTH_LABELS[bandwidth])} & {fmt(row.get('s', np.nan))} & {fmt(row.get('chi2', np.nan))} & {fmt(row.get('sB1', np.nan))} & {delta} & {ci} & {p} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (latex / "table_confirmatory_auc.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result_tex = r"""% Auto-generated only after the 150-cell confirmatory gate passed.
\subsection{Confirmatory 30-seed factor campaign}
All 150 predeclared cells (30 unseen physical seeds by five bandwidth levels)
completed with the frozen calibration-only thresholds. The event is the unit
of analysis; confidence intervals resample physical seeds, and paired
Wilcoxon tests are Holm-adjusted across bandwidth levels. Pooled ROC curves
are descriptive and are not used as the inferential unit.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{paper1_generated/figures/fig_confirmatory_residual_silence.pdf}
\caption{Estimator-matched $\chi^2$ residual-silence rate across bandwidth. Error bars are 95\% physical-seed cluster-bootstrap intervals.}
\label{fig:residual-silence}
\end{figure}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{paper1_generated/figures/fig_confirmatory_matched_far.pdf}
\caption{Recall and false-alarm transport at thresholds frozen from the same calibration-only nominal/ample population.}
\label{fig:matched-far}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{paper1_generated/figures/fig_confirmatory_far_miss_vs_condition.pdf}
\caption{False-alarm and missed-drift transport versus realized loss and mean telemetry age. Error bars resample physical seeds.}
\label{fig:far}
\end{figure*}

\input{paper1_generated/latex/table_confirmatory_silence.tex}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{paper1_generated/figures/fig_confirmatory_roc_by_bandwidth.pdf}
\caption{Descriptive pooled ROC curves by bandwidth. Statistical conclusions use paired seed-level AUC contrasts in Table~\ref{tab:auc}.}
\label{fig:roc}
\end{figure*}

\input{paper1_generated/latex/table_confirmatory_auc.tex}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{paper1_generated/figures/fig_confirmatory_collinearity_q.pdf}
\caption{Arm-G pre-interpretation diagnostic. Strong $B_1$--$u$ collinearity limits the residual variation available to an incremental geometry test; $q$ quantifies how often B1 is forced to tie positive--negative pairs.}
\label{fig:collinearity-q}
\end{figure}

\input{paper1_generated/latex/table_confirmatory_collinearity_q.tex}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{paper1_generated/figures/fig_confirmatory_f3_arm_g.pdf}
\caption{Arm-G test of whether the full information-geometric score adds discrimination beyond residual-plus-B1/B2 controls. Passing Arm C alone is not treated as evidence for F3.}
\label{fig:f3-arm-g}
\end{figure}

\input{paper1_generated/latex/table_confirmatory_f3.tex}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.88\textwidth]{paper1_generated/figures/fig_confirmatory_silent_latency.pdf}
\caption{Detection-by-deadline curves restricted to oracle-positive events that are silent to estimator-matched $\chi^2$. Non-detections remain in the denominator.}
\label{fig:silent-latency}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{paper1_generated/figures/fig_confirmatory_network_conditions.pdf}
\caption{Realized communication conditions and abstention/hold behavior, verifying that nominal bandwidth factors produced distinct network states.}
\label{fig:realized-network}
\end{figure*}
"""
    (latex / "paper1_confirmatory_results.tex").write_text(result_tex, encoding="utf-8")

    # Headline prose is generated from the same cluster summaries used by the
    # tables.  This prevents manuscript text from drifting away from the
    # sealed numerical evidence.  The prose is intentionally descriptive:
    # F1--F3 are not labelled "passed" unless a separate, predeclared verdict
    # rule exists in the analysis contract.
    def headline(metric: str, outcome: str, bandwidth: str) -> float:
        block = summary[
            summary["metric"].eq(metric)
            & summary["outcome"].eq(outcome)
            & summary["bandwidth_level"].eq(bandwidth)
        ]
        return float(block.iloc[0]["mean"]) if len(block) else float("nan")

    floor = "bw00_floor"
    oracle = "bw04_oracle"
    floor_full_recall = headline("s", "recall", floor)
    floor_chi_recall = headline("chi2", "recall", floor)
    floor_full_far = headline("s", "false_alarm_rate", floor)
    floor_chi_far = headline("chi2", "false_alarm_rate", floor)
    floor_full_auc = headline("s", "auc", floor)
    floor_chi_auc = headline("chi2", "auc", floor)
    oracle_full_auc = headline("s", "auc", oracle)
    oracle_chi_auc = headline("chi2", "auc", oracle)

    severe = regime_summary[
        regime_summary["regime"].eq("severe")
        & regime_summary["metric"].eq("s")
        & regime_summary["outcome"].eq("residual_silence_rate")
    ]
    severe_silence = float(severe.iloc[0]["mean"]) if len(severe) else float("nan")

    floor_f2 = contrasts[
        contrasts["bandwidth_level"].eq(floor)
        & contrasts["outcome"].eq("auc")
        & contrasts["comparator"].eq("chi2")
    ]
    floor_f2_delta = float(floor_f2.iloc[0]["mean_paired_difference"]) if len(floor_f2) else float("nan")
    floor_f2_low = float(floor_f2.iloc[0]["ci95_low"]) if len(floor_f2) else float("nan")
    floor_f2_high = float(floor_f2.iloc[0]["ci95_high"]) if len(floor_f2) else float("nan")
    floor_f2_p = float(floor_f2.iloc[0]["wilcoxon_p_holm"]) if len(floor_f2) else float("nan")

    arm_f3_path = output / "paper1_arm_paired_contrasts.csv"
    arm_f3 = pd.read_csv(arm_f3_path) if arm_f3_path.exists() else pd.DataFrame()
    if not arm_f3.empty:
        floor_f3 = arm_f3[
            arm_f3["arm"].eq("G")
            & arm_f3["bandwidth_level"].eq(floor)
            & arm_f3["outcome"].eq("auc")
            & arm_f3["comparator"].eq("sB1")
        ]
    else:
        floor_f3 = pd.DataFrame()
    floor_f3_delta = float(floor_f3.iloc[0]["mean_paired_difference"]) if len(floor_f3) else float("nan")
    floor_f3_low = float(floor_f3.iloc[0]["ci95_low"]) if len(floor_f3) else float("nan")
    floor_f3_high = float(floor_f3.iloc[0]["ci95_high"]) if len(floor_f3) else float("nan")
    floor_f3_p = float(floor_f3.iloc[0]["wilcoxon_p_holm"]) if len(floor_f3) else float("nan")

    abstract_tex = (
        "In the confirmatory campaign, the severe-regime residual-silence "
        f"rate was {fmt(severe_silence)}. At the floor bandwidth, the full "
        f"metric achieved recall {fmt(floor_full_recall)} at realized FAR "
        f"{fmt(floor_full_far)}, compared with recall {fmt(floor_chi_recall)} "
        f"and FAR {fmt(floor_chi_far)} for estimator-matched $\\chi^2$; "
        f"their mean seed-level AUCs were {fmt(floor_full_auc)} and "
        f"{fmt(floor_chi_auc)}, respectively."
    )
    (latex / "paper1_confirmatory_abstract.tex").write_text(
        abstract_tex + "\n", encoding="utf-8"
    )

    discussion_tex = (
        "The severe-regime residual-silence rate was "
        f"{fmt(severe_silence)}, directly quantifying F1. At the floor "
        f"bandwidth, the paired full-minus-$\\chi^2$ AUC difference was "
        f"{fmt(floor_f2_delta)} (95\\% seed-cluster interval "
        f"[{fmt(floor_f2_low)}, {fmt(floor_f2_high)}], Holm-adjusted "
        f"$p={fmt(floor_f2_p, 4)}$). The corresponding Arm-G "
        f"full-minus-residual+B1 difference was {fmt(floor_f3_delta)} "
        f"(95\\% interval [{fmt(floor_f3_low)}, {fmt(floor_f3_high)}], "
        f"Holm-adjusted $p={fmt(floor_f3_p, 4)}$), which is the direct "
        "incremental-geometry test for F3. In the practically unconstrained "
        f"network cell, the full and $\\chi^2$ mean seed-level AUCs were "
        f"{fmt(oracle_full_auc)} and {fmt(oracle_chi_auc)}. These quantities, "
        "rather than a pooled-timestep test, determine the empirical "
        "interpretation."
    )
    (latex / "paper1_confirmatory_discussion.tex").write_text(
        discussion_tex + "\n", encoding="utf-8"
    )

    conclusion_tex = (
        "Across 30 unseen physical seeds, the floor-bandwidth full metric "
        f"had mean seed-level AUC {fmt(floor_full_auc)} versus "
        f"{fmt(floor_chi_auc)} for estimator-matched $\\chi^2$, with a "
        f"paired difference of {fmt(floor_f2_delta)} (95\\% interval "
        f"[{fmt(floor_f2_low)}, {fmt(floor_f2_high)}]). The Arm-G "
        f"increment over residual+B1 was {fmt(floor_f3_delta)} "
        f"[{fmt(floor_f3_low)}, {fmt(floor_f3_high)}], locating how much "
        "of any communication-aware advantage is specifically attributable "
        "to measurement geometry."
    )
    (latex / "paper1_confirmatory_conclusion.tex").write_text(
        conclusion_tex + "\n", encoding="utf-8"
    )

    headline_json = {
        "schema": "paper1.confirmatory.headline.v1",
        "cells": manifest["cells_found"],
        "seeds": len(manifest["confirmatory_seed_indices"]),
        "severe_residual_silence_rate": severe_silence,
        "floor": {
            "full_recall": floor_full_recall,
            "chi2_recall": floor_chi_recall,
            "full_far": floor_full_far,
            "chi2_far": floor_chi_far,
            "full_auc": floor_full_auc,
            "chi2_auc": floor_chi_auc,
            "paired_auc_difference_full_minus_chi2": floor_f2_delta,
            "paired_auc_difference_ci95": [floor_f2_low, floor_f2_high],
            "paired_auc_difference_holm_p": floor_f2_p,
            "arm_g_paired_auc_difference_full_minus_sB1": floor_f3_delta,
            "arm_g_paired_auc_difference_ci95": [floor_f3_low, floor_f3_high],
            "arm_g_paired_auc_difference_holm_p": floor_f3_p,
        },
        "oracle_like": {"full_auc": oracle_full_auc, "chi2_auc": oracle_chi_auc},
    }
    (output / "paper1_confirmatory_headline.json").write_text(
        json.dumps(headline_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_output_hashes(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--static-results", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)

    confirmatory = subparsers.add_parser("confirmatory")
    confirmatory.add_argument("--runs-root", type=Path, required=True)
    confirmatory.add_argument("--design", type=Path, required=True)
    confirmatory.add_argument("--thresholds", type=Path, required=True)
    confirmatory.add_argument("--static-results", type=Path)
    confirmatory.add_argument("--output", type=Path, required=True)
    confirmatory.add_argument("--confirmatory-seeds", default="2-31")
    confirmatory.add_argument("--bootstrap-draws", type=int, default=2000)
    confirmatory.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "diagnostic":
        write_diagnostic_outputs(args.static_results, args.output)
        write_output_hashes(args.output)
        print("PAPER1_DIAGNOSTIC_PUBLICATION_OUTPUT_OK")
        print("STATUS=diagnostic_not_confirmatory_performance")
        print("OUTPUT=", args.output.resolve())
        return 0

    seeds = parse_seed_spec(args.confirmatory_seeds)
    analyze_confirmatory(
        runs_root=args.runs_root,
        design_path=args.design,
        threshold_path=args.thresholds,
        output=args.output,
        seeds=seeds,
        draws=args.bootstrap_draws,
        allow_partial=args.allow_partial,
    )
    status = json.loads((args.output / "paper1_analysis_manifest.json").read_text())["status"]
    if status == "confirmatory_complete":
        render_confirmatory(args.output)
    if args.static_results:
        write_diagnostic_outputs(args.static_results, args.output)
    write_output_hashes(args.output)
    print("PAPER1_PUBLICATION_ANALYSIS_OK")
    print("STATUS=", status)
    print("OUTPUT=", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
