#!/bin/bash

# Startup script for Phishing Detection API
echo "🚀 Starting Phishing Detection API..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Check if model file exists
if [ ! -f "XGBoostClassifier.pickle.dat" ]; then
    echo "❌ XGBoostClassifier.pickle.dat not found!"
    echo "Please make sure the model file is in the same directory as this script."
    exit 1
fi

# Install requirements if needed
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📊 API Documentation available at http://localhost:8000/docs"
echo "🔍 Health check available at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 backend.py
