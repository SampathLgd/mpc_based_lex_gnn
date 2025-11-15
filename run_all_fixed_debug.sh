#!/bin/bash
set -e
echo "cleanup old workers..."
ps aux | egrep 'mpc_node_worker_batched_fixed_v2.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
sleep 0.1
echo "start nodes in background..."
python mpc_node_worker_batched_fixed_v2.py --id 0 > node0.log 2>&1 &
PID0=$!
python mpc_node_worker_batched_fixed_v2.py --id 1 > node1.log 2>&1 &
PID1=$!
python mpc_node_worker_batched_fixed_v2.py --id 2 > node2.log 2>&1 &
PID2=$!
sleep 0.5
echo "running debugger coordinator..."
python mpc_coordinator_beaver_batched_fixed_debug_inspect.py
echo "tail node logs (last 20 lines each):"
tail -n 20 node0.log || true
tail -n 20 node1.log || true
tail -n 20 node2.log || true
echo "cleaning up nodes..."
kill -9 $PID0 $PID1 $PID2 2>/dev/null || true
echo "done"
