#!/bin/bash
set -x

APP="$*"

echo "$NUM_NODES nodes from `hostname`"
worker_num=$NUM_NODES
hostfile=$(pwd)/mpi.hf
if command -v COMMAND &> /dev/null; then
    scontrol show hostnames $NODE_LIST > $hostfile
else
    echo $NODE_LIST | sed -e "s/ \+/\n/g" > $hostfile
fi

for i in $(seq 1 $worker_num); do
    echo "$RUNNING on $i nodes"
    cmd="mpirun -n $i -f $hostfile -ppn 1 $APP --no-nodes=$i"
    echo "$cmd"
    time $cmd
    echo "return code: $?"
done

echo Done
