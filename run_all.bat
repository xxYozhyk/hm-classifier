@echo off
chcp 65001 > nul
echo ========================================
echo ЗАПУСК H&M CLASSIFIER
echo ========================================
echo.
echo Запуск сервера...
start /B D:\ANACONDAV4\python.exe app.py
echo Ждем 3 секунды для запуска сервера...
timeout /t 3 /nobreak > nul
echo.
echo Запуск тестов...
D:\ANACONDAV4\python.exe test_api.py
echo.
echo ========================================
echo ТЕСТИРОВАНИЕ ЗАВЕРШЕНО
echo ========================================
pause