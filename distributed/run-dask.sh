#!/bin/bash
set -x
APP="$*"

echo "$NUM_NODES nodes from `hostname`"
worker_num=$NUM_NODES
head=$(hostname)
nodes=$(echo $NODE_LIST | sed -e "s/ \+/\n/g" | grep -v $head) # Getting the node names exept head

export DASK_CFG=$(pwd)/dask.$$
port=41041

echo "STARTING DASK SCHEDULER at $head:$port $DASK_CFG"
dask-scheduler --port $port --scheduler-file $DASK_CFG &
sleep 10

dawo=`which dask-worker`
i=1
for node in $head $nodes; do
  echo "STARTING DASK WORKER at node $node $DASK_CFG";
  ssh -f $node $dawo --scheduler-file=$DASK_CFG --nprocs=auto --local-directory=/tmp/$USER
  sleep 10
  echo "$RUNNING on $i nodes"
  cmd="$APP --no-nodes=$i"
  echo "$cmd"
  time $cmd
  echo "return code: $?"
  i=$((i+1))
done

echo Done
killall dask-scheduler
killall ssh
