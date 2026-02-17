import os
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker

from src.common.config import GenConfig


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _choice(rng: np.random.Generator, values: Tuple, probs: Tuple, size: int):
    return rng.choice(np.array(values, dtype=object), p=np.array(probs), size=size)


def generate_users(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    fake = Faker()
    Faker.seed(cfg.seed)

    start = _to_dt(cfg.start_date)
    end = _to_dt(cfg.end_date)
    total_days = (end - start).days

    user_ids = np.arange(1, cfg.n_users + 1, dtype=np.int64)

    # signup dates uniformly distributed
    signup_offsets = rng.integers(0, total_days + 1, size=cfg.n_users)
    signup_dates = [start + timedelta(days=int(x)) for x in signup_offsets]

    countries = _choice(rng, cfg.countries, cfg.country_probs, cfg.n_users)
    channels = _choice(rng, cfg.channels, cfg.channel_probs, cfg.n_users)

    # Experiment assignment: randomized
    variants = _choice(rng, cfg.variants, cfg.variant_probs, cfg.n_users)

    # Simple demographics
    ages = rng.integers(18, 70, size=cfg.n_users)
    has_kids = rng.binomial(1, 0.35, size=cfg.n_users).astype(bool)

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": pd.to_datetime(signup_dates),
            "country": countries,
            "acquisition_channel": channels,
            "experiment_name": cfg.experiment_name,
            "variant": variants,
            "age": ages,
            "has_kids": has_kids,
        }
    )
    return users


def generate_subscriptions(cfg: GenConfig, users: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)

    plans = _choice(rng, cfg.plans, cfg.plan_probs, len(users))

    # baseline monthly prices
    price_map = {"basic": 9.99, "standard": 14.99, "premium": 19.99}
    base_price = np.array([price_map[p] for p in plans], dtype=float)

    # simulate discounts based on channel
    channel = users["acquisition_channel"].to_numpy()
    discount = np.where(channel == "partner_bundle", 0.20, 0.00)  # 20% off for bundle
    discount = np.where(channel == "email", 0.05, discount)       # 5% off email

    monthly_price = base_price * (1.0 - discount)

    subs = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy(),
            "plan": plans,
            "monthly_price": np.round(monthly_price, 2),
            "started_at": users["signup_date"].to_numpy(),
        }
    )
    return subs


def generate_events(cfg: GenConfig, users: pd.DataFrame) -> pd.DataFrame:
    """
    Session-level events, with *true causal effect* baked in:
      - control baseline retention/engagement
      - variant_a: small uplift
      - variant_b: bigger uplift (the 'winner')
    """
    rng = np.random.default_rng(cfg.seed + 2)

    start = _to_dt(cfg.start_date)
    end = _to_dt(cfg.end_date)

    user_ids = users["user_id"].to_numpy()
    signup_dates = pd.to_datetime(users["signup_date"]).dt.to_pydatetime()
    variants = users["variant"].to_numpy()
    countries = users["country"].to_numpy()

    # baseline engagement intensity by country (just for realism)
    country_factor_map = {"US": 1.00, "CA": 0.95, "UK": 0.90, "AU": 0.92, "IN": 0.85}
    country_factor = np.array([country_factor_map[c] for c in countries], dtype=float)

    # causal uplift: variant impacts engagement intensity + churn odds
    # (your causal inference should recover these)
    variant_engagement_uplift = np.where(variants == "control", 1.00,
                                  np.where(variants == "variant_a", 1.03, 1.08))

    # number of sessions per user ~ Poisson with caps
    base_lambda = 18.0  # average sessions/year
    lam = base_lambda * country_factor * variant_engagement_uplift
    n_sessions = rng.poisson(lam=lam).clip(0, cfg.max_session_per_user)

    # build events rows
    rows = []
    for uid, sd, ns, ve in zip(user_ids, signup_dates, n_sessions, variant_engagement_uplift):
        if ns == 0:
            continue

        # each session date after signup
        max_days = max(1, (end - sd).days)
        day_offsets = rng.integers(0, max_days, size=int(ns))
        session_starts = [sd + timedelta(days=int(d)) for d in day_offsets]

        # minutes watched: gamma-like distribution
        # uplift multiplies minutes too (variant effect)
        minutes = rng.gamma(shape=2.0, scale=20.0, size=int(ns)) * ve
        minutes = np.clip(minutes, 1, cfg.max_session_minutes).astype(int)

        for ts, m in zip(session_starts, minutes):
            rows.append((uid, ts, "session", int(m)))

    events = pd.DataFrame(rows, columns=["user_id", "event_time", "event_type", "value"])
    events["event_time"] = pd.to_datetime(events["event_time"])
    return events


def build_labels(cfg: GenConfig, users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Create outcome labels like retention_30d, retention_90d, total_minutes_30d.
    These will be used in A/B testing and causal inference.
    """
    # Join signup date
    base = users[["user_id", "signup_date", "variant", "country", "acquisition_channel"]].copy()

    ev = events.merge(base[["user_id", "signup_date"]], on="user_id", how="left")
    ev["days_since_signup"] = (ev["event_time"] - ev["signup_date"]).dt.days

    # minutes in first 30/90 days
    ev_30 = ev[(ev["days_since_signup"] >= 0) & (ev["days_since_signup"] <= 30)]
    ev_90 = ev[(ev["days_since_signup"] >= 0) & (ev["days_since_signup"] <= 90)]

    mins_30 = ev_30.groupby("user_id")["value"].sum().rename("minutes_30d")
    mins_90 = ev_90.groupby("user_id")["value"].sum().rename("minutes_90d")

    # retention: at least 1 session in window
    ret_30 = ev_30.groupby("user_id").size().gt(0).astype(int).rename("retention_30d")
    ret_90 = ev_90.groupby("user_id").size().gt(0).astype(int).rename("retention_90d")

    labels = base.merge(mins_30, on="user_id", how="left") \
                 .merge(mins_90, on="user_id", how="left") \
                 .merge(ret_30, on="user_id", how="left") \
                 .merge(ret_90, on="user_id", how="left")

    labels[["minutes_30d", "minutes_90d"]] = labels[["minutes_30d", "minutes_90d"]].fillna(0).astype(int)
    labels[["retention_30d", "retention_90d"]] = labels[["retention_30d", "retention_90d"]].fillna(0).astype(int)
    return labels


def main():
    cfg = GenConfig()
    _ensure_dir(cfg.out_dir)

    print("Config:", asdict(cfg))

    print("1) Generating users...")
    users = generate_users(cfg)
    print("Users:", users.shape)

    print("2) Generating subscriptions...")
    subs = generate_subscriptions(cfg, users)
    print("Subscriptions:", subs.shape)

    print("3) Generating events...")
    events = generate_events(cfg, users)
    print("Events:", events.shape)

    print("4) Building outcome labels...")
    labels = build_labels(cfg, users, events)
    print("Labels:", labels.shape)

    # Save parquet (fast, compact)
    users_path = os.path.join(cfg.out_dir, "users.parquet")
    subs_path = os.path.join(cfg.out_dir, "subscriptions.parquet")
    events_path = os.path.join(cfg.out_dir, "events.parquet")
    labels_path = os.path.join(cfg.out_dir, "labels.parquet")

    users.to_parquet(users_path, index=False)
    subs.to_parquet(subs_path, index=False)
    events.to_parquet(events_path, index=False)
    labels.to_parquet(labels_path, index=False)

    print("✅ Saved:")
    print(" -", users_path)
    print(" -", subs_path)
    print(" -", events_path)
    print(" -", labels_path)

    # quick sanity check: retention by variant
    summary = labels.groupby("variant")[["retention_30d", "retention_90d", "minutes_30d"]].mean().sort_index()
    print("\n📊 Sanity Check (mean outcomes by variant):")
    print(summary.round(4))


if __name__ == "__main__":
    main()
