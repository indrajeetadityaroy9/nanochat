#!/bin/bash
# The production run on one H100 80 GB SXM5, training through evaluation (scripts/production.py), in the image.
#
#   tmux new -d -s h100 runs/h100.sh    (from the checkout)
exec "$(cd "$(dirname "$0")" && pwd)"/container.sh python -m scripts.production
