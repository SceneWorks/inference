uv python install 3.12.10 --managed-python --no-bin --no-config
PYTHON="$(uv python find 3.12.10 --managed-python --no-project --no-config)"
"$PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,12,10), sys.version'
printf 'REVIEWED_PYTHON=%s\n' "$PYTHON" >> "$GITHUB_ENV"
