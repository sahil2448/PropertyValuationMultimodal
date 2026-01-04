***Note --> The recommended files have been added to the root of project also...but while building the project, all the files were used from src

# Satellite Imagery-Based Property Valuation

A multimodal regression pipeline that predicts property market value by combining tabular housing data with satellite imagery.

## Project Overview

This project develops a machine learning model that integrates:
- **Tabular Data:** Housing features (sqft, bedrooms, grade, etc.)
- **Visual Data:** Sentinel-2 satellite imagery capturing environmental context

**Results:** Achieved R² = 0.8606 with multimodal fusion, a 1.9% improvement over tabular-only baseline.

## Repository Structure

```
PropertyValuationMultimodal/
├── data/
│   ├── raw/                    # Original CSV files
│   ├── images/                 # Satellite images (train/test)
│   └── processed/              # Preprocessed numpy arrays
├── src/
│   ├── config.py               # Configuration constants
│   ├── data_fetcher.py         # Sentinel Hub API integration
│   ├── preprocessing.py        # Feature engineering & scaling
│   ├── extract_embeddings.py   # ResNet18 feature extraction
│   ├── tabular_baseline.py     # Baseline model training
│   ├── fusion_regressor.py     # Multimodal fusion model
│   ├── gradcam.py              # Explainability visualizations
│   └── eda_generate.py         # EDA plots generation
├── outputs/
│   ├── eda/                    # EDA visualizations
│   ├── gradcam/                # Grad-CAM heatmaps
│   └── submission.csv          # Final predictions
├── preprocessing.ipynb         # Data preprocessing notebook
├── model_training.ipynb        # Model training notebook
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sahil2448/PropertyValuationMultimodal.git
cd PropertyValuationMultimodal
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Sentinel Hub API
Create a `.env` file with your Copernicus Data Space credentials:
```
SENTINEL_CLIENT_ID=your_client_id
SENTINEL_CLIENT_SECRET=your_client_secret
```

### 5. Run the Pipeline

**Step 1: Fetch Satellite Images**
```bash
python src/data_fetcher.py
```

**Step 2: Preprocess Data**
```bash
python src/preprocessing.py
```

**Step 3: Extract Image Embeddings**
```bash
python src/extract_embeddings.py
```

**Step 4: Train Baseline Model**
```bash
python src/tabular_baseline.py
```

**Step 5: Train Fusion Model**
```bash
python src/fusion_regressor.py
```

**Step 6: Generate Visualizations**
```bash
python src/eda_generate.py
python src/gradcam_generate.py
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Handling | Pandas, NumPy, GeoPandas |
| Deep Learning | PyTorch, TorchVision |
| Image Processing | OpenCV, Pillow |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Satellite API | SentinelHub |

## Results

| Model | RMSE ($) | R² Score |
|-------|----------|----------|
| Tabular Baseline | 134,809 | 0.8552 |
| Multimodal Fusion | **132,247** | **0.8606** |

## Key Features

1. **Automated Satellite Image Fetching** - Resumable pipeline with error handling
2. **27 Engineered Features** - From 15 raw housing attributes
3. **ResNet18 Visual Embeddings** - 512-dimensional feature extraction
4. **Grad-CAM Explainability** - Visual interpretation of model predictions

## Author

**Sahil Kamble**  
GitHub: [@sahil2448](https://github.com/sahil2448)

## License

This project is for educational purposes.
