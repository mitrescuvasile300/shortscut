@echo off
echo === Shortscut VPS Deploy Script ===
echo.
echo Installing paramiko if needed...
pip install paramiko >nul 2>&1
echo.
python "%~dp0deploy_transcript.py" %*
if errorlevel 1 (
    echo.
    echo Deploy failed! Check the error above.
    pause
) else (
    echo.
    echo Deploy successful!
    pause
)
