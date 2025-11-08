@echo off
REM Quick Launch Scripts for Phishing Detection Apps
REM Choose your app by uncommenting the desired line

echo ================================
echo Phishing Detection App Launcher
echo ================================
echo.
echo 1. Quick Prediction (Fastest)
echo 2. XAI Explanations (Detailed)
echo 3. Unified Interface (Recommended)
echo.

choice /c 123 /n /m "Select app (1-3): "

if errorlevel 3 goto unified
if errorlevel 2 goto xai
if errorlevel 1 goto quick

:quick
echo.
echo Starting Quick Prediction App...
streamlit run src/app/streamlit_multimodel.py
goto end

:xai
echo.
echo Starting XAI Explanations App...
streamlit run src/app/streamlit_xai.py
goto end

:unified
echo.
echo Starting Unified Interface...
streamlit run src/app/streamlit_unified.py
goto end

:end
