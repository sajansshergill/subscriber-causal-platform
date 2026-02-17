from dataclasses import dataclass

@dataclass(frozen=True)
class GenConfig:
    seed: int = 42
    n_users: int = 200_000 #scale up laster (1M+)
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    
    # Experiment
    experiment_name: str = "perks_variant_test_v1"
    variants: tuple = ("control", "variant_a", "variant_b")
    variant_probs: tuple = (0.34, 0.33, 0.33) # Must sum to 1.0
    
    # Subscription
    plans: tuple = ("basic", "standard", "premium")
    plan_probs: tuple = (0.40, 0.45, 0.15)
    
    # Geo + channels
    countries: tuple = ("US", "UK", "CA", "AU", "IN")
    country_probs: tuple = (0.45, 0.10, 0.15, 0.10, 0.20)
    
    channels: tuple = ("organic", "paid_search", "paid_social", "partner_bundle", "email")
    channel_probs: tuple = (0.35, 0.20, 0.20, 0.15, 0.10)
    
    # Engagement simulation
    max_session_per_user: int = 120
    max_session_minutes: int = 180
    
    # Output
    out_dir: str = "data/raw"