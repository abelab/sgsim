# sgsim: A Simple Skip Graph Simulator and Visualizer

## Overview
*sgsim* is a simple skip graph simulator and visualizer.

sgsim can:

- construct a skip graph overlay network based on a random membership vector,
- perform some simple experiments, 
- render routing paths over a skip graph topology, and
- compute some statistics and output graphs.

sgsim is written in Python and uses [pandas](https://pandas.pydata.org/) for statistics.


*sgsim* is developed by Kota Abe at Osaka City University, Japan
[[Link](https://www.media.osaka-cu.ac.jp/~k-abe/)]

![Unicast Image](https://github.com/abelab/sgsim/raw/main/images/unicast.png)

![Graph](https://github.com/abelab/sgsim/raw/main/images/sg_hops_vs_n.png)

![Histogram](https://github.com/abelab/sgsim/raw/main/images/sg_hops_hist.png)

## License
MIT License

## Installation
Tested with Python3.9.

Install dependencies:
```
% pip install -r requirements.txt
```

## Usage
```
% python sg_main.py -h
usage: sg_main.py [-h] [-n N] [-a ALPHA] [--exp {basic,unicast,unicast_vary_n}]
                  [--unicast-algorithm {greedy,original}] [--fast-join] [--seed SEED] [--interactive]
                  [--output-topology-max-level OUTPUT_TOPOLOGY_MAX_LEVEL] [--output-hop-graph] [--diagonal]
                  [-v]

sgsim: Skip Graph Simulator and Visualizer

optional arguments:
  -h, --help            show this help message and exit
  -n N                  number of nodes (default: 8)
  -a ALPHA, --alpha ALPHA
                        base of membership vector (default: 2)
  --exp {basic,unicast,unicast_vary_n}
                        experiment type
  --unicast-algorithm {greedy,original}
                        unicast algorithm
  --fast-join           use fast (cheat) join
  --seed SEED           give a random seed
  --interactive         display graphs on a window rather than save to files
  --output-topology-max-level OUTPUT_TOPOLOGY_MAX_LEVEL
                        render a topology from level 0 to the specified level (use with --exp basic)
  --output-hop-graph    render a hop graph (use with --exp unicast)
  --diagonal            draw diagonal line (use with --output-hop-graph)
  -v, --verbose         verbose output
```

Currently, a real (event-based) node insertion algorithm is not implemented.  
You have to give `--fast-join` option to instruct statistically constructing a skip graph.

### Basic options:

- `--exp basic` Do statistics about routing tables.  
- `--exp basic --output-topology-max-level=3` Output a topology graph from level 0 to 3 (sg_topology.png).
- `--exp unicast` Do simple unicast experiments.  Output histograms of the number of hops (sg_hops_hist.png) and of messages (sg_msgs_hist.png).
- `--exp unicast --output-hop-graph` Output hop graphs (unicast-(algorithm)-#.png).
- `--exp unicast_vary_n` Compute an average number of hops, varying number of nodes.  Output a graph (sg_hops_vs_n.png). 
- `--unicast-algorithm greedy` Use a greedy algorithm for unicast
- `--unicast-algorithm original` Use an algorithm based on the article ["Skip Graphs"](https://dl.acm.org/doi/10.1145/1290672.1290674) for unicast

## Examples

- `python sg_main.py  --fast-join --exp unicast --output-hop-graph -n 15 --unicast-algorithm original`
