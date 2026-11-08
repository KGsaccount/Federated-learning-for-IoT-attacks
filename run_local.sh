#!/usr/bin/env bash
# Runs one Flower server and N clients on this machine.
# Usage: ./run_local.sh [clients=3] [rounds=10]
set -euo pipefail
CLIENTS=${1:-3}
ROUNDS=${2:-10}

${PYTHON:-python3} server.py --rounds "$ROUNDS" --clients "$CLIENTS" &
SERVER=$!
sleep 5
for i in $(seq 1 "$CLIENTS"); do
  ${PYTHON:-python3} client.py --id "$i" > "client$i.log" 2>&1 &
done
wait "$SERVER"
