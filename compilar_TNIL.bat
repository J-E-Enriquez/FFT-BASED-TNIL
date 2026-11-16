
@echo off
setlocal

REM Ubicarse en la carpeta del proyecto
cd /d "%~dp0"

REM Definir la ruta del entorno virtual
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

REM Verificar si el entorno existente funciona
if exist "%VENV_PY%" (
    "%VENV_PY%" --version >nul 2>&1
    if errorlevel 1 goto recrear
) else (
    if exist ".venv\" goto recrear
    goto crear
)

goto instalar

:recrear
echo El entorno virtual no funciona.
echo Eliminando el entorno anterior...

rmdir /s /q ".venv"
if exist ".venv\" goto error

:crear
echo Creando un nuevo entorno virtual...

py -3 -m venv .venv >nul 2>&1

if errorlevel 1 (
    python -m venv .venv
    if errorlevel 1 goto error
)

if not exist "%VENV_PY%" goto error

:instalar
echo Instalando dependencias...

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto error

REM Instalar PyInstaller
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 goto error

REM Compilar TNIL
echo Compilando aplicacion TNIL...

"%VENV_PY%" -m PyInstaller --noconfirm TNIL.spec
if errorlevel 1 goto error

echo.
echo Compilacion completada correctamente.
pause
exit /b 0

:error
echo.
echo ERROR: No se pudo completar la compilacion.
echo Comprueba Python y las dependencias.
pause
exit /b 1
