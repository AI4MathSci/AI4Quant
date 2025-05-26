#!/bin/bash
# run_servers.sh
# Script to run both FastAPI servers simultaneously without blocking each other

# Change to the project directory
cd /content/experiments/ResearchTrader

# Check if .env file exists with OpenAI API key
if [ ! -f .env ]; then
    echo "Creating .env file with placeholder for OPENAI_API_KEY"
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo "Please update the .env file with your actual OpenAI API key"
fi

# Run both servers with & to put each in background
echo "Starting main server..."
uv run python research_trader/main.py &
main_pid=$!

echo "Starting app server..."
uv run python research_trader/app.py &
app_pid=$!

echo "Main server running with PID: $main_pid"
echo "App server running with PID: $app_pid"

# Trap Ctrl+C to properly terminate both servers
trap "kill $main_pid $app_pid; exit" INT TERM

# Keep script running
echo "Servers are running. Press Ctrl+C to stop both servers."
wait