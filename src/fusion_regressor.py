# src/fusion_regressor.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from config import PROCESSED_DATA_DIR
from preprocessing import DataPreprocessor


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(PROCESSED_DATA_DIR / "train_processed.npz", allow_pickle=True)
    X_train_tab = npz["X_train"]
    y_train = npz["y_train"].reshape(-1)  # scaled
    X_val_tab = npz["X_val"]
    y_val = npz["y_val"].reshape(-1)      # scaled

    emb_train = np.load(PROCESSED_DATA_DIR / "img_emb_train.npy")
    emb_val = np.load(PROCESSED_DATA_DIR / "img_emb_val.npy")

    X_train = np.concatenate([X_train_tab, emb_train], axis=1)
    X_val = np.concatenate([X_val_tab, emb_val], axis=1)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=800,
        random_state=42
    )
    model.fit(X_train, y_train)

    pred_val_scaled = model.predict(X_val).reshape(-1, 1)

    prep = DataPreprocessor()
    prep.load_preprocessor()

    y_val_price = prep.target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    pred_val_price = prep.target_scaler.inverse_transform(pred_val_scaled).reshape(-1)

    rmse = root_mean_squared_error(y_val_price, pred_val_price)
    r2 = r2_score(y_val_price, pred_val_price)

    pd.DataFrame({
    "y_true": y_val_price,
    "y_pred": pred_val_price,
    }).to_csv(out_dir / "fusion_val_preds.csv", index=False)

    print(f"Saved: {out_dir / 'fusion_val_preds.csv'}")

    print("\nFUSION REGRESSOR RESULTS (tabular + resnet18 embeddings)")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    (out_dir / "fusion_metrics.json").write_text(json.dumps(
        {"rmse": float(rmse), "r2": float(r2), "model": "HistGradientBoostingRegressor(tab+imgemb)"},
        indent=2
    ))

    X_test_tab = np.load(PROCESSED_DATA_DIR / "X_test_tab.npy")
    emb_test = np.load(PROCESSED_DATA_DIR / "img_emb_test.npy")
    test_ids = np.load(PROCESSED_DATA_DIR / "test_ids.npy")

    X_test = np.concatenate([X_test_tab, emb_test], axis=1)
    pred_test_scaled = model.predict(X_test).reshape(-1, 1)
    pred_test_price = prep.target_scaler.inverse_transform(pred_test_scaled).reshape(-1)

    sub = pd.DataFrame({"id": test_ids, "predicted_price": pred_test_price})
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"Saved: {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
