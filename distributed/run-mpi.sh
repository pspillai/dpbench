#!/bin/bash
set -x

APP="$*"

echo "$NUM_NODES nodes from `hostname`"
worker_num=$NUM_NODES
hostfile=$(pwd)/mpi.hf
echo $NODE_LIST | sed -e "s/ \+/\n/g" > $hostfile

for i in $(seq 1 $worker_num); do
    echo "$RUNNING on $i nodes"
    if (( $i == 1 )); then
	cmd="mpirun -n $i -f $hostfile -ppn 1 $APP --no-nodes=$i"
    else
	nw=$(( $i-1 ))
	cmd="mpirun -f $hostfile -ppn 1 -n 1 $APP --no-nodes=$i : -n $nw $APP --no-nodes=$i --worker"
    fi
    echo "$cmd"
    time $cmd
    echo "return code: $?"
done

echo Done
