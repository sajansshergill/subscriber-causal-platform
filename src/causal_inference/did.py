import os
import pandas as pd
import numpy as np
from math import erf, sqrt


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def did_estimate(
    panel_path: str = os.path.join("data", "processed", "geo_panel.parquet"),
    outcome_col: str = "minutes_per_active_user",
):
    """
    Difference-in-Differences:
        outcome ~ treated_geo + post + treated_geo*post

    Returns:
        coefficient, standard error, t-stat, p-value
    """
    df = pd.read_parquet(panel_path)

    needed = {"treated_geo", "post", "did", outcome_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build design matrix
    X = df[["treated_geo", "post", "did"]].astype(float)
    X = np.column_stack([np.ones(len(df)), X.values])  # intercept
    y = df[outcome_col].astype(float).values

    # OLS estimate
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Residuals
    y_hat = X @ beta
    residuals = y - y_hat

    # Variance estimation
    n, k = X.shape
    sigma2 = (residuals @ residuals) / (n - k)

    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))

    # DiD coefficient
    did_coef = beta[3]
    did_se = se[3]
    t_stat = did_coef / did_se if did_se != 0 else np.nan
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

    print("\n📌 Difference-in-Differences Results")
    print(f"Coefficient (DiD effect): {did_coef:.6f}")
    print(f"Standard Error: {did_se:.6f}")
    print(f"t-statistic: {t_stat:.6f}")
    print(f"p-value: {p_value:.6f}")

    return {
        "did_effect": did_coef,
        "standard_error": did_se,
        "t_stat": t_stat,
        "p_value": p_value,
    }


if __name__ == "__main__":
    did_estimate()
