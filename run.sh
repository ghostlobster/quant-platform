#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting Quant Platform at http://localhost:8501"
streamlit run app.py --server.port 8501
