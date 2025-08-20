@echo off
title Camera Test Landing Page - Local Server
color 0A

echo.
echo ========================================
echo    🎥 Camera Test Landing Page
echo ========================================
echo.
echo Starting local server...
echo.

python deploy_server.py

echo.
echo Server stopped. Press any key to exit...
pause >nul
