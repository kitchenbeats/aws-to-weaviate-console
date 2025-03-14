#!/bin/bash
# Startup script for FDA Consultant AI
source ../venv/bin/activate
streamlit run fda-ai-ui.py --server.port 8501 --server.address 0.0.0.0
