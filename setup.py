#!/bin/bash
# setup.sh
echo "Setting up environment..."
python -m venv quantum_env
source quantum_env/bin/activate
pip install -r environment/requirements.txt
