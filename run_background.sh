#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment .venv not found in $SCRIPT_DIR"
  exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate virtual environment"
  exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
  echo "Error: main.py file not found in $SCRIPT_DIR"
  exit 1
fi

# Run python main.py in the background
nohup python main.py > whisperer.log 2>&1 &

echo "Groq Whisperer started in the background."
echo "Output is being redirected to whisperer.log"
echo "To stop the script, you can use 'pkill -f main.py'"

# Deactivate virtual environment
deactivate