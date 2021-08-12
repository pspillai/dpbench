#/bin/bash
set -x

# Configuration is done through env vars:
# - NODE_LIST: list of cluster nodes (names, ips) to use
# - SLURM_JOB_NODELIST: Evaluated if NODE_LIST is not set
# - NODE_FILE: if none of the above are set, reads cluster nodes from this file
# - DISTS: list whitespace-separated frameworks to use, defaults to all
# - BENCHS: list of whitespace-separated benchmarks to run, defaults to all ("jstencil linreg nbody mandelbrot lbfgs")

# eval list of nodes
if [ -n "${NODE_LIST}" ]; then
    export NODE_LIST
else
    if [ -n "${SLURM_JOB_NODELIST}" ]; then
        export NODE_LIST=${SLURM_JOB_NODELIST}
    else
        if [ -n "${NODE_FILE}" ]; then
       	    export NODE_LIST=$(cat ${NODE_FILE})
        else
            echo "Need one of NODE_LIST, SLURM_JOB_NODELIST, or NODE_FILE being set"
            exit 1
        fi
    fi
fi
_tmp=(${NODE_LIST})
export NUM_NODES=${#_tmp[@]}

dists=${DISTS-"heat dask numpy torch"}
benchs=${BENCHS-"jstencil linreg nbody mandelbrot lbfgs"}

bargs="-b 5"
for bench in $benchs; do
    # the benchmark configuration is hard-coded here as well
    if [[ "$bench" == "jstencil" ]]; then
	app="python jstencil_run.py -n 6000 -i 3 $bargs"
    elif [[ "$bench" == "nbody" ]]; then
	app="python nbody_run.py -n 200 -t 10 -d 0.1 $bargs"
    elif [[ "$bench" == "linreg" ]]; then
	app="python linreg_run.py -n 100 -i 1000 $bargs"
    elif [[ "$bench" == "mandelbrot" ]]; then
	app="python mandelbrot_run.py -d 5000,5000 -i 200 $bargs"
    elif [[ "$bench" == "lbfgs" ]]; then
	app="python lbfgs_run.py $bargs"
    fi
    # now we run this app for all the frameworks
    # the run-scripts start and tear-down dask/ray cluster each time to make sure
    # there are no left-overs, like in the in-memroy object store or whatever.
    for dist in $dists; do
	if [[ "$dist" == "heat" ]]; then
	    ./run-mpi.sh $app -u $dist
	elif [[ "$dist" == "ramba" || "$dist" == "nums" ]]; then
	    ./run-ray.sh $app -u $dist
	elif [[ "$dist" == "numpy" || "$dist" == "torch" ]]; then
	    $app -u $dist
	else
	    ./run-${dist}.sh $app -u $dist
	fi
    done
done

# now we run 
for dist in $dists; do
    $apps_now=${apps//_run.py/_run.py --u $dist/}
    if "$dist" == "heat"; then
	run-mpi.sh $apps_now
    elif [[ "$dist" == "ramba" || "$dist" == "nums" ]]; then
	run-ray.sh $apps_now -u $dist
    else
	run-${dist}.sh $apps_now -u $dist
    fi
done
