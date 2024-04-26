#!/usr/bin/env bash

python -m streamlit run gesture_classification/monitoring/streamlit/app.py --server.port 8001 --server.headless true --server.address=0.0.0.0 &
python -m uvicorn gesture_classification.main:application --host 0.0.0.0 --port 8000