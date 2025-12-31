# src/tabular_baseline.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from config import PROCESSED_DATA_DIR
from preprocessing import DataPreprocessor


def main():
    npz_path = PROCESSED_DATA_DIR / "train_processed.npz"
    data = np.load(npz_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"] 
    X_val = data["X_val"]
    y_val = data["y_val"]  

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train.reshape(-1))

    pred_val_scaled = model.predict(X_val).reshape(-1, 1)

    prep = DataPreprocessor()
    prep.load_preprocessor()

    y_val_price = prep.target_scaler.inverse_transform(y_val)
    pred_val_price = prep.target_scaler.inverse_transform(pred_val_scaled)

    rmse = root_mean_squared_error(y_val_price, pred_val_price)  # RMSE [web:200]
    r2 = r2_score(y_val_price, pred_val_price)

    print("\nTABULAR BASELINE RESULTS")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    # Save outputs for report
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"rmse": float(rmse), "r2": float(r2), "model": "HistGradientBoostingRegressor"}
    (out_dir / "tabular_baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({
        "y_true": y_val_price.reshape(-1),
        "y_pred": pred_val_price.reshape(-1),
    }).to_csv(out_dir / "tabular_baseline_val_preds.csv", index=False)

    print(f"Saved: {out_dir / 'tabular_baseline_metrics.json'}")
    print(f"Saved: {out_dir / 'tabular_baseline_val_preds.csv'}")


if __name__ == "__main__":
    main()
