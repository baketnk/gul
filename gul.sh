#!/bin/bash
cd ~/workspace/gul
source .venv/bin/activate
python -m src.main "$@"
