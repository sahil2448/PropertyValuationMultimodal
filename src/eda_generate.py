# src/eda_generate.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import shutil

IN_CSV = Path("data/raw/train_with_images.csv")
OUT_DIR = Path("outputs/eda")
IMG_SRC_DIR = Path("data/images/train")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {IN_CSV}...")
    df = pd.read_csv(IN_CSV)
    
    print("Generating Price Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price ($)')
    plt.axvline(df['price'].median(), color='red', linestyle='--', label=f'Median: ${df["price"].median():,.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_price_dist.png", dpi=150)
    plt.close()

    print("Generating Geospatial Map...")
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(df['long'], df['lat'], c=np.log1p(df['price']), 
                    cmap='plasma', alpha=0.5, s=3)
    plt.colorbar(sc, label='Log(Price)')
    plt.title('Geospatial Distribution of Property Prices')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_geospatial_map.png", dpi=150)
    plt.close()

    print("Generating Correlation Matrix...")
    cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition']
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_correlation.png", dpi=150)
    plt.close()
    
    print("Selecting sample satellite images...")
    df_sorted = df.sort_values('price')
    cheapest = df_sorted.head(3)
    expensive = df_sorted.tail(3)
    
    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    for i, (_, row) in enumerate(cheapest.iterrows()):
        src = row['image_path']
        dst = samples_dir / f"low_price_{i+1}_{row['price']:.0f}.jpg"
        if Path(src).exists():
            shutil.copy(src, dst)
            
    for i, (_, row) in enumerate(expensive.iterrows()):
        src = row['image_path']
        dst = samples_dir / f"high_price_{i+1}_{row['price']:.0f}.jpg"
        if Path(src).exists():
            shutil.copy(src, dst)

    print(f"Done! Check {OUT_DIR} for plots and {samples_dir} for sample images.")

if __name__ == "__main__":
    main()
