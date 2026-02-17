from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ABResult:
    metric: str
    control_mean: float
    variant_mean: float
    abs_lift: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_variant: int


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _z_test_proportions(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    denom = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 1.0

    z = (p2 - p1) / denom
    return float(2.0 * (1.0 - _norm_cdf(abs(z))))


def _bootstrap_ci_diff_means(
    control: np.ndarray,
    variant: np.ndarray,
    n_boot: int = 1500,
    alpha: float = 0.05,
    seed: int = 123,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)

    if len(control) == 0 or len(variant) == 0:
        return (float("nan"), float("nan"))

    ctrl = control.astype(float)
    var = variant.astype(float)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        c_s = rng.choice(ctrl, size=len(ctrl), replace=True)
        v_s = rng.choice(var, size=len(var), replace=True)
        diffs[i] = v_s.mean() - c_s.mean()

    low = float(np.quantile(diffs, alpha / 2))
    high = float(np.quantile(diffs, 1 - alpha / 2))
    return low, high


def _perm_pvalue_diff_means(
    control: np.ndarray,
    variant: np.ndarray,
    n_boot: int = 1500,
    seed: int = 123,
) -> float:
    rng = np.random.default_rng(seed)

    if len(control) == 0 or len(variant) == 0:
        return float("nan")

    ctrl = control.astype(float)
    var = variant.astype(float)
    obs = var.mean() - ctrl.mean()

    pooled = np.concatenate([ctrl, var])
    n_ctrl = len(ctrl)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rng.shuffle(pooled)
        c = pooled[:n_ctrl]
        v = pooled[n_ctrl:]
        diffs[i] = v.mean() - c.mean()

    return float((np.abs(diffs) >= abs(obs)).mean())


def run_ab_test(
    df: pd.DataFrame,
    metric: str,
    variant_col: str = "variant",
    control_label: str = "control",
    compare_to: Optional[List[str]] = None,
    metric_type: str = "auto",  # auto | binary | continuous
    n_boot: int = 1500,
    seed: int = 123,
) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe.")
    if variant_col not in df.columns:
        raise ValueError(f"Variant column '{variant_col}' not found in dataframe.")

    work = df[[variant_col, metric]].dropna().copy()

    if metric_type == "auto":
        uniq = set(work[metric].unique().tolist())
        metric_type = "binary" if uniq.issubset({0, 1}) else "continuous"

    control = work.loc[work[variant_col] == control_label, metric].to_numpy()

    if compare_to is None:
        compare_to = sorted([v for v in work[variant_col].unique().tolist() if v != control_label])

    rows = []
    for v in compare_to:
        variant = work.loc[work[variant_col] == v, metric].to_numpy()

        c_mean = float(control.mean()) if len(control) else float("nan")
        v_mean = float(variant.mean()) if len(variant) else float("nan")

        abs_lift = v_mean - c_mean
        rel_lift = (abs_lift / c_mean) if (np.isfinite(c_mean) and c_mean != 0) else float("nan")

        ci_low, ci_high = _bootstrap_ci_diff_means(control, variant, n_boot=n_boot, seed=seed)

        if metric_type == "binary":
            x1, n1 = int(control.sum()), int(len(control))
            x2, n2 = int(variant.sum()), int(len(variant))
            p_val = _z_test_proportions(x1, n1, x2, n2)
        else:
            p_val = _perm_pvalue_diff_means(control, variant, n_boot=n_boot, seed=seed)

        rows.append(
            ABResult(
                metric=metric,
                control_mean=c_mean,
                variant_mean=v_mean,
                abs_lift=float(abs_lift),
                rel_lift=float(rel_lift),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                p_value=float(p_val),
                n_control=int(len(control)),
                n_variant=int(len(variant)),
            ).__dict__
        )

    out = pd.DataFrame(rows)
    out["significant_0_05"] = out["p_value"] < 0.05
    return out.sort_values(["metric", "p_value"], ascending=[True, True]).reset_index(drop=True)
