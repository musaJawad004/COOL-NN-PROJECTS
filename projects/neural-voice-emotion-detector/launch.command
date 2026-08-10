#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import torch, numpy, scipy, sounddevice, tkinter' >/dev/null 2>&1; then
  PYTHON=.venv/bin/python
elif python3 -c 'import torch, numpy, scipy, sounddevice, tkinter' >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Setup is required. Follow README.md."
  read -r "?Press Return to close..."
  exit 1
fi
"$PYTHON" -m app.gui
