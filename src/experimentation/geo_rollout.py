import os
import numpy as np
import pandas as pd


def build_geo_panel(
    users_path: str = os.path.join("data", "raw", "users.parquet"),
    events_path: str = os.path.join("data", "raw", "events.parquet"),
    out_path: str = os.path.join("data", "processed", "geo_panel.parquet"),
    seed: int = 42,
):
    """
    Build a weekly geo-level panel dataset for DiD:
      geo (country) x week with outcome = avg session minutes and active users.

    We simulate a geo rollout:
      treated geos: subset of countries
      post period: weeks after rollout_date
    """
    rng = np.random.default_rng(seed)

    users = pd.read_parquet(users_path)
    events = pd.read_parquet(events_path)

    # We’ll use country as "geo"
    # If you later want “state/market”, we can generate those too.
    events = events.merge(users[["user_id", "country", "signup_date"]], on="user_id", how="left")

    # Weekly bucket
    events["week"] = events["event_time"].dt.to_period("W").astype(str)

    # Aggregate weekly geo metrics
    panel = (
        events.groupby(["country", "week"], as_index=False)
        .agg(
            total_minutes=("value", "sum"),
            active_users=("user_id", "nunique"),
            sessions=("event_type", "count"),
        )
    )

    panel["minutes_per_active_user"] = panel["total_minutes"] / panel["active_users"].replace(0, np.nan)
    panel["minutes_per_active_user"] = panel["minutes_per_active_user"].fillna(0.0)

    # Choose treated geos (e.g., 2 countries) — deterministic but randomizable
    geos = sorted(panel["country"].unique().tolist())
    treated_geos = rng.choice(geos, size=min(2, len(geos)), replace=False).tolist()

    # Pick a rollout week near the middle of timeline
    weeks = sorted(panel["week"].unique().tolist())
    rollout_week = weeks[len(weeks) // 2]

    panel["treated_geo"] = panel["country"].isin(treated_geos).astype(int)
    panel["post"] = (panel["week"] >= rollout_week).astype(int)
    panel["did"] = panel["treated_geo"] * panel["post"]

    meta = {
        "treated_geos": treated_geos,
        "rollout_week": rollout_week,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.to_parquet(out_path, index=False)

    print("✅ Saved geo panel:", out_path)
    print("Treated geos:", treated_geos)
    print("Rollout week:", rollout_week)

    return panel, meta


if __name__ == "__main__":
    build_geo_panel()
