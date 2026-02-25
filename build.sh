#!/usr/bin/env bash
set -e

echo "==> Python version: $(python --version)"
echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies (binary wheels only — no compilation)..."
pip install --only-binary=:all: -r requirements.txt

echo "==> Build complete!"
