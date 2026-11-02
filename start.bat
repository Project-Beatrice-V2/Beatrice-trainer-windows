@echo off
rem Convenience wrapper: same as double-clicking start_windows.bat.
call "%~dp0start_windows.bat" %*
exit /b %errorlevel%
