import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SENTINEL_CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID", "")
SENTINEL_CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET", "")
SENTINEL_INSTANCE_ID = os.getenv("SENTINEL_INSTANCE_ID", "")

IMAGE_SIZE = 224

TRAIN_FILE = "train1.csv"
TEST_FILE = "test2.csv"
TARGET_COLUMN = "price"

TABULAR_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
    'sqft_living15', 'sqft_lot15'
]
