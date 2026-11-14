
@echo off
cd /d "%~dp0"

echo ======================================
echo    TNIL - Compilacion ONEFILE
echo ======================================

.venv\Scripts\python.exe -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --distpath dist ^
    --workpath build ^
    TNIL.spec

if errorlevel 1 (
    echo.
    echo ERROR: No se pudo generar el ejecutable.
    pause
    exit /b 1
)

echo.
echo Compilacion finalizada correctamente.
echo.
echo Ejecutable: dist\TNIL.exe
echo.

pause
