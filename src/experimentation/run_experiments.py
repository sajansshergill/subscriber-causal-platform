import os
import pandas as pd

from src.experimentation.ab_test import run_ab_test

def main():
    labels_path = os.path.join("data", "raw", "labels.parquet")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Missing {labels_path}. Run the generator first: \n"
            "python - m src/data_generator.generate_synthetic_data"
        )
    
    labels = pd.read_parquet(labels_path)
    
    metrics = ["retention_30d", "retention_90d", "minutes_30d", "minutes_90d"]
    
    results = []
    for m in metrics:
        res = run_ab_test(labels, metric=m, metric_type="auto", n_boot=1500)
        results.append(res)
        
    final = pd.concat(results, ignore_index=True)
    out_dir = os.path.join("data", "processed")
    out_path = os.path.join(out_dir, "ab_test_results.csv")
    final.to_csv(out_path, index=False)
    
    print(f"Saved A/B test results to {out_path}")
    
    # Exec-friendly print
    cols = ["metric", "control_mean", "variant_mean", "abs_lift", "rel_lift", "ci_low", "ci_high", "p_value"]
    print("\n Executive Summary (Top rows):")
    print(final[cols].round(5).head(20))
    
if __name__ == "__main__":
    main()