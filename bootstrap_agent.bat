@echo off
setlocal
cd /d "%~dp0"

if "%AFAC_PYTHON%"=="" (
  set "PYTHON=python"
) else (
  set "PYTHON=%AFAC_PYTHON%"
)
set "ANCHOR=%CD%\artifacts\A1_v53q1_transition_stable_edge_h2_SAFE.csv"

"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not available: %PYTHON%
  echo Set AFAC_PYTHON to your Python executable, or activate the environment first.
  pause
  exit /b 2
)

echo [INFO] Running doctor...
"%PYTHON%" -m afac_agent.doctor --project_root "%CD%"
if errorlevel 1 (
  echo [ERROR] Doctor failed. Fix the reported M0/M1 preflight issues first.
  pause
  exit /b 3
)

"%PYTHON%" -m afac_agent.auto_loop ^
  --project_root "%CD%" ^
  --anchor_csv "%ANCHOR%" ^
  --max_steps 10

echo.
echo Agent stopped at the first missing input or unbound tool.
pause
