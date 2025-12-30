import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
from pathlib import Path
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN,
    TABULAR_FEATURES, VALIDATION_SPLIT, RANDOM_SEED
)


class DataPreprocessor:

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = None
        self.target_scaler = StandardScaler()

    def load_data(self, dataset_type='train'):
        filepath = RAW_DATA_DIR / f"{dataset_type}_with_images.csv"
        df = pd.read_csv(filepath)
        print(f"Loaded {dataset_type} data: {df.shape}")
        return df

    def clean_data(self, df):
        print("\nCleaning data...")
        initial_shape = df.shape

        df = df.dropna(subset=['lat', 'long'])

        if 'image_path' in df.columns:
            df = df[df['image_path'].notna()]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)

        df = df.drop_duplicates()

        print(f"Data cleaned: {initial_shape} -> {df.shape}")
        print(f"Removed {initial_shape[0] - df.shape[0]} rows")

        return df

    def engineer_features(self, df):
        print("\nEngineering features...")

        current_year = 2024
        df['house_age'] = current_year - df['yr_built']
        df['years_since_renovation'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else current_year - x)
        df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)

        if TARGET_COLUMN in df.columns:
            df['price_per_sqft'] = df[TARGET_COLUMN] / df['sqft_living']

        df['living_lot_ratio'] = df['sqft_living'] / df['sqft_lot']
        df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
        df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
        df['basement_ratio'] = df['sqft_basement'] / df['sqft_living']
        df['living_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
        df['lot_vs_neighbors'] = df['sqft_lot'] / (df['sqft_lot15'] + 1)
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['quality_score'] = df['condition'] * df['grade']
        df['lat_long_interaction'] = df['lat'] * df['long']

        print(f"Engineered features. New shape: {df.shape}")

        return df

    def prepare_features(self, df, is_training=True):
        all_features = TABULAR_FEATURES + [
            'house_age', 'years_since_renovation', 'was_renovated',
            'living_lot_ratio', 'bedroom_bathroom_ratio',
            'has_basement', 'basement_ratio',
            'living_vs_neighbors', 'lot_vs_neighbors',
            'total_rooms', 'quality_score', 'lat_long_interaction'
        ]

        available_features = [f for f in all_features if f in df.columns]
        self.feature_names = available_features

        print(f"\nUsing {len(available_features)} tabular features")

        X_tabular = df[available_features].values

        if is_training:
            X_tabular = self.scaler.fit_transform(X_tabular)
        else:
            X_tabular = self.scaler.transform(X_tabular)

        y = None
        if TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN].values.reshape(-1, 1)
            if is_training:
                y = self.target_scaler.fit_transform(y)
            else:
                y = self.target_scaler.transform(y)

        image_paths = df['image_path'].values if 'image_path' in df.columns else None

        return X_tabular, y, image_paths

    def split_data(self, X_tabular, y, image_paths, val_split=VALIDATION_SPLIT):
        indices = np.arange(len(X_tabular))

        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=RANDOM_SEED)

        X_train = X_tabular[train_idx]
        X_val = X_tabular[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        images_train = image_paths[train_idx]
        images_val = image_paths[val_idx]

        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")

        return (X_train, y_train, images_train), (X_val, y_val, images_val)

    def save_preprocessor(self, filepath=None):
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "preprocessor.pkl"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names
            }, f)

        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath=None):
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "preprocessor.pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.target_scaler = data['target_scaler']
            self.feature_names = data['feature_names']

        print(f"Preprocessor loaded from {filepath}")


def main():
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)

    preprocessor = DataPreprocessor()
    train_df = preprocessor.load_data('train')
    train_df = preprocessor.clean_data(train_df)
    train_df = preprocessor.engineer_features(train_df)
    X_tabular, y, image_paths = preprocessor.prepare_features(train_df, is_training=True)
    train_data, val_data = preprocessor.split_data(X_tabular, y, image_paths)
    preprocessor.save_preprocessor()

    np.savez(
        PROCESSED_DATA_DIR / "train_processed.npz",
        X_train=train_data[0],
        y_train=train_data[1],
        images_train=train_data[2],
        X_val=val_data[0],
        y_val=val_data[1],
        images_val=val_data[2]
    )

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

