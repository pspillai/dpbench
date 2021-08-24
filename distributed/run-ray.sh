#!/bin/bash
set -x
APP="$*"

export REDIS_PASSWORD="5241590000000000"  #$(uuidgen)

echo "$SLURM_JOB_NUM_NODES nodes from `hostname`"
worker_num=$(($NUM_NODES - 1)) #number of nodes other than the head node

head=$(hostname)
nodes=$(echo $NODE_LIST | sed -e "s/ \+/\n/g" | grep -v $head)
ip=$(hostname --ip-address) # making redis-address
port=6377
export ip_head=$ip:$port
echo "IP Head: $ip_head"

echo "STARTING HEAD at $head"
raycmd=`which ray`
for node in $NODE_LIST; do
    ssh $node $raycmd stop --force
done
ray start --head --node-ip-address=$ip --port=$port --redis-password=$REDIS_PASSWORD

export RAY_HEAD=$ip_head

RWPN=$(( $OMP_NUM_THREADS / 4 ))
export RAMBA_NUM_THREADS=$(( $OMP_NUM_THREADS / $RWPN ))

i=1
for node in $head $nodes; do
    export RAMBA_WORKERS=$(( $i * $RWPN ))
    if [[ "$head" != "$node" ]]; then
	echo "STARTING WORKER at ${node}"
	ssh ${node} ${raycmd} start --address $ip_head --redis-password=$REDIS_PASSWORD
	sleep 5
    fi
    cmd="$APP --no-nodes=$i"
    echo "$cmd"
    time $cmd
    echo "return code: $?"
    i=$((i+1))
done

echo Done
for node in $NODE_LIST; do
    ssh $node $raycmd stop --force
done
