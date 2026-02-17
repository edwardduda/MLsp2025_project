#!/bin/bash

echo "=========================================="
echo "KAN Network Visualizer - Flask App"
echo "=========================================="
echo ""

echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

echo "Checking dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Checking model file..."
if [ -f "saved_models/kan_cnn.pt" ]; then
    echo "✓ Trained model found at saved_models/kan_cnn.pt"
else
    echo "⚠ No trained model found. Running in demo mode with untrained model."
    echo "  To use a trained model, place your model file at: saved_models/kan_cnn.pt"
fi

echo ""
echo "Starting Flask application..."
echo "Access the app at: http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

python3 -m src.web.app
