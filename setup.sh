echo "================================"
echo "Property Valuation Setup Script"
echo "================================"
echo ""

echo "[1/5] Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/images/train
mkdir -p data/images/test
mkdir -p outputs/models
mkdir -p outputs/predictions
mkdir -p outputs/visualizations
mkdir -p notebooks
echo "✓ Directories created"

echo ""
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "[5/5] Setup complete!"
echo ""
echo "================================"
echo "Next Steps:"
echo "================================"
echo "1. Place train1.csv and test2.csv in data/raw/"
echo "2. Create .env file and add your Google Maps API key"
echo "3. Run: python src/data_fetcher.py"
echo "4. Run: python src/preprocessing.py"
echo "5. Run: python src/train.py"
echo "6. Run: python src/evaluate.py"
echo "7. Run: python src/gradcam.py"
echo "================================"
