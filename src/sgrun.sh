#!/bin/bash

PYTHON=python3
MAIN=sg_main.py

# VERBOSE=-v
VERBOSE=
# RANDOM_SEED=1
RANDOM_SEED=

if [[ -n "$RANDOM_SEED" ]]; then
  SEED_ARG="--seed $RANDOM_SEED"
else
  SEED_ARG=
fi

# basic experiment
# generate sg_topology.png (--output-topology-max-level)
$PYTHON $MAIN $VERBOSE $SEED_ARG --exp basic -n 10 -a 2 --fast-join --output-topology-max-level=3

# simple unicast experiment
# generate sg_hops_hist.png, sg_msgs_hist.png
# generate unicast-greedy-#.png (--output-hop-graph)
$PYTHON $MAIN $VERBOSE $SEED_ARG --exp unicast -n 10 -a 2 --fast-join --unicast-algorithm greedy --output-hop-graph
# generate unicast-original-#.png (--output-hop-graph)
$PYTHON $MAIN $VERBOSE $SEED_ARG --exp unicast -n 10 -a 2 --fast-join --unicast-algorithm original --output-hop-graph

# if you prefer diagonal lines
# $PYTHON $MAIN $VERBOSE $SEED_ARG --exp unicast -n 10 -a 2 --fast-join --output-hop-graph --diagonal

# do unicast, varying number of nodes
# generate sg_hops_vs_n.png
$PYTHON $MAIN $VERBOSE $SEED_ARG --exp unicast_vary_n --fast-join
