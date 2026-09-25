uv python install 3.12.10 --managed-python --no-registry --no-bin --no-config || exit /b 1
for /f "delims=" %%P in ('uv python find 3.12.10 --managed-python --no-project --no-config') do set "REVIEWED_PYTHON=%%P"
if not defined REVIEWED_PYTHON exit /b 1
"%REVIEWED_PYTHON%" -c "import sys; assert sys.version_info[:3] == (3, 12, 10), sys.version" || exit /b 1
echo REVIEWED_PYTHON=%REVIEWED_PYTHON%>>"%GITHUB_ENV%"
