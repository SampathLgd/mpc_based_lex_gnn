#!/bin/bash
set -e

# --- ADD THIS LINE ---
echo "--- Running Offline Phase (precompute.py) ---"
python precompute.py
echo "--- Offline Phase Complete ---"
# --- END ADD ---

echo "🧹 Killing any old MPC node processes..."
ps aux | egrep 'mpc_node_worker_batched_fixed_v2.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
sleep 0.2

echo " Starting MPC nodes..."
python mpc_node_worker_batched_fixed_v2.py --id 0 > node0.log 2>&1 &
PID0=$!
python mpc_node_worker_batched_fixed_v2.py --id 1 > node1.log 2>&1 &
PID1=$!
python mpc_node_worker_batched_fixed_v2.py --id 2 > node2.log 2>&1 &
PID2=$!

sleep 0.5
echo "Nodes launched (PIDs: $PID0, $PID1, $PID2). Checking ports..."
ss -ltnp | egrep ':9000|:9001|:9002' || echo "⚠️ Nodes not yet listening, waiting a bit more..."
sleep 0.5

echo " Running GNN Layer v2 Coordinator (Private Attention)..."
python mpc_gnn_coordinator_v2.py

echo " MPC logs (tail):"
tail -n 10 node0.log || true
tail -n 10 node1.log || true
tail -n 10 node2.log || true

echo " Cleaning up nodes..."
kill -9 $PID0 $PID1 $PID2 2>/dev/null || true
echo "All done!"