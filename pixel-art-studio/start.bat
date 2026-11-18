@echo off
cd /d "%~dp0"
where git >nul 2>nul || (echo Install Git for Windows first.& pause & exit /b 1)
where py >nul 2>nul || (echo Install Python 3.11+ first.& pause & exit /b 1)
where bash >nul 2>nul || (echo Git Bash is required. Reinstall Git for Windows with Git Bash selected.& pause & exit /b 1)
bash ./start.sh
pause
