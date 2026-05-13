#!/usr/bin/env bash
# run_env.sh - Helper script to run commands inside the torch251 environment

if [ $# -eq 0 ]; then
    echo "Usage: ./run_env.sh <command...>"
    echo "Example: ./run_env.sh python script.py"
    exit 1
fi

mamba run -n torch251 "$@"
