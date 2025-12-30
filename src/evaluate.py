import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from pathlib import Path

from models import initialize_model
from train import PropertyDataset, calculate_metrics
from preprocessing import DataPreprocessor
from config import BATCH_SIZE, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_COLUMN


def load_test_data():
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor()

    test_df = preprocessor.load_data('test')
    test_df = preprocessor.clean_data(test_df)
    test_df = preprocessor.engineer_features(test_df)
    X_test, _, image_paths = preprocessor.prepare_features(test_df, is_training=False)

    test_ids = test_df['id'].values

    return X_test, image_paths, test_ids, preprocessor


def generate_predictions(model, test_loader, device, target_scaler):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for images, tabular in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            predictions = model(images, tabular)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_predictions = target_scaler.inverse_transform(all_predictions)

    return all_predictions


def main():
    print("="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)

    print("\nLoading test data...")
    X_test, image_paths, test_ids, preprocessor = load_test_data()

    test_dataset = PropertyDataset(X_test, image_paths, targets=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Test samples: {len(test_dataset)}")

    tabular_input_size = X_test.shape[1]
    model, device = initialize_model(tabular_input_size, 'multimodal')

    print("\nLoading best model checkpoint...")
    checkpoint_path = OUTPUTS_DIR / 'models' / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")

    print("\nGenerating predictions...")
    predictions = generate_predictions(model, test_loader, device, preprocessor.target_scaler)

    predictions_df = pd.DataFrame({
        'id': test_ids,
        'predicted_price': predictions.flatten()
    })

    output_path = OUTPUTS_DIR / 'predictions' / 'test_predictions.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved predictions to {output_path}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: ${predictions_df['predicted_price'].mean():,.2f}")
    print(f"  Median: ${predictions_df['predicted_price'].median():,.2f}")
    print(f"  Min: ${predictions_df['predicted_price'].min():,.2f}")
    print(f"  Max: ${predictions_df['predicted_price'].max():,.2f}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

