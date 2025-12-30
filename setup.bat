@echo off
echo ================================
echo Property Valuation Setup Script
echo ================================
echo.

echo [1/5] Creating directory structure...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\images\train 2>nul
mkdir data\images\test 2>nul
mkdir outputs\models 2>nul
mkdir outputs\predictions 2>nul
mkdir outputs\visualizations 2>nul
mkdir notebooks 2>nul
echo DONE: Directories created

echo.
echo [2/5] Creating virtual environment...
python -m venv venv
echo DONE: Virtual environment created

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo DONE: Virtual environment activated

echo.
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo DONE: Dependencies installed

echo.
echo [5/5] Setup complete!
echo.
echo ================================
echo Next Steps:
echo ================================
echo 1. Place train1.csv and test2.csv in data/raw/
echo 2. Create .env file and add your Google Maps API key
echo 3. Run: python src/data_fetcher.py
echo 4. Run: python src/preprocessing.py
echo 5. Run: python src/train.py
echo 6. Run: python src/evaluate.py
echo 7. Run: python src/gradcam.py
echo ================================
pause