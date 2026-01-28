#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
cd "$ROOT_DIR"

echo "ROOT_DIR=$ROOT_DIR"

echo "Running Generator.py with Fire..."
python tasks/minimal_task/Generator_wop.py --p1_grid=[0.01] --p2_grid=[0.1] --modulus=97

echo "Running Main.py with Fire..."
python arithmetic_experiments/minimal_task_experiment/Main_wop.py --p1=[0.01] --p2=[0.1] --num_epochs=16 --modulus=97

echo "All done!"