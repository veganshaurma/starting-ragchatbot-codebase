#!/bin/bash
# Development quality checks: formatting and tests

set -e

echo "=== Checking formatting (black) ==="
uv run black backend/ --check

echo ""
echo "=== Running tests ==="
cd backend && uv run pytest tests/ -v
