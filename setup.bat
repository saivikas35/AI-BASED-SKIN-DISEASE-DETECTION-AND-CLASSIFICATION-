@echo off
echo ===== Skin Disease Detection Auto Setup (Windows + GPU) =====

:: Step 0 - Initialize conda
CALL conda init >nul 2>&1

:: Step 1 - Create environment with Python 3.9
echo Creating Python 3.9 environment...
conda create -y -n skin-disease python=3.9

:: Step 2 - Activate environment
echo Activating environment...
call conda activate skin-disease

:: Step 3 - Install CUDA + cuDNN (for GPU)
echo Installing CUDA 11.2 and cuDNN 8.1...
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

:: Step 4 - Install Python dependencies
echo Installing project requirements...
pip install --upgrade pip
pip install -r requirements.txt

:: Step 5 - Create GPU test file
echo Creating GPU test file...
echo import tensorflow as tf > gpu_test.py
echo print("GPU Available:", tf.config.list_physical_devices('GPU')) >> gpu_test.py
echo print("TensorFlow Version:", tf.__version__) >> gpu_test.py

:: Step 6 - Run GPU test
echo Running GPU test...
python gpu_test.py

:: Step 7 - Run Flask app
echo Starting application...
python app.py

echo.
echo ✅ DONE! Open browser: http://localhost:5000
pause