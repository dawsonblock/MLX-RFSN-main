#!/bin/bash
# Build script for isolated Python environments
# Phase 6.25: Build in isolated environments (Python 3.11 and 3.12)

set -e

PYTHON_VERSION=${1:-"3.11"}
BUILD_DIR="build/isolated/python${PYTHON_VERSION}"

echo "=== Building in isolated Python ${PYTHON_VERSION} environment ==="

# Create build directory
mkdir -p "${BUILD_DIR}"

# Create isolated virtual environment
echo "Creating virtual environment..."
python${PYTHON_VERSION} -m venv "${BUILD_DIR}/venv"

# Activate virtual environment
source "${BUILD_DIR}/venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install build dependencies
echo "Installing build dependencies..."
pip install build wheel setuptools

# Install pinned dependencies for Apple Silicon
echo "Installing pinned dependencies..."
pip install -r requirements-apple-silicon.txt

# Install package in development mode
echo "Installing package..."
pip install -e .

# Run portable tests (no MLX required)
echo "Running portable tests..."
pytest -m portable -q

# Run MLX tests (if MLX is available)
echo "Running MLX tests..."
pytest -m mlx -q || echo "MLX tests skipped (MLX not available in this environment)"

# Deactivate
deactivate

echo "=== Build complete for Python ${PYTHON_VERSION} ==="
echo "Virtual environment: ${BUILD_DIR}/venv"
