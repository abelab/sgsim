from __future__ import annotations

import argparse
import itertools
import math
import random
from typing import TypeVar, Type, cast

import matplotlib.pyplot as plt
import pandas as pd

import sg
import sg_draw as draw
from discrete_ev_sim import SchedEvent, EventExecutor
from sg import SGNode, MembershipVector, UnicastGreedy, UnicastOriginal, UnicastBase


class SGArguments:
    """
    This class is for a type hint of an instance returned by argparse.parse_args()
    """
    def __init__(self):
        self.n = 0
        self.alpha = 0
        self.exp = ''
        self.unicast_algorithm = ''
        self.fast_join = False
        self.seed = 0
        self.interactive = False
        self.output_topology_max_level = 0
        self.output_hop_graph = False
        self.hop_graph_diagonal = False
        self.verbose = False

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="sgsim: Skip Graph Simulator and Visualizer")
        parser.add_argument('-n', help=f'number of nodes (default: {sg.DEFAULT_N})', default=sg.DEFAULT_N, type=int)
        parser.add_argument('-a', '--alpha', help=f'base of membership vector (default: {sg.DEFAULT_ALPHA})',
                            default=sg.DEFAULT_ALPHA, dest='alpha', type=int)
        parser.add_argument('--exp', help=f'experiment type', choices=['basic', 'unicast', 'unicast_vary_n'],
                            type=str, default='basic')
        algorithms = list(cls.unicast_algorithms_map().keys())
        parser.add_argument('--unicast-algorithm', help='unicast algorithm', choices=algorithms,
                            type=str, default=algorithms[0], dest='unicast_algorithm')
        parser.add_argument('--fast-join', help=f'use fast (cheat) join', action='store_true', dest='fast_join')
        parser.add_argument('--seed', help=f'give a random seed', type=int)
        parser.add_argument('--interactive', help='display graphs on a window rather than save to files',
                            action='store_true')
        parser.add_argument('--output-topology-max-level',
                            help=f'render a topology from level 0 to the specified level (use with --exp basic)',
                            default=0, type=int, dest='output_topology_max_level')
        parser.add_argument('--output-hop-graph', help=f'render a hop graph (use with --exp unicast)',
                            action='store_true', dest='output_hop_graph')
        parser.add_argument('--diagonal', help=f'draw diagonal line (use with --output-hop-graph)',
                            action='store_true', dest='hop_graph_diagonal')
        parser.add_argument('-v', '--verbose', help='verbose output', action='store_true', dest='verbose')
        return parser

    @classmethod
    def unicast_algorithms_map(cls) -> dict[str, Type[UnicastBase]]:
        return {
            'greedy': UnicastGreedy,
            'original': UnicastOriginal
        }


class SGMain:
    def __init__(self):
        self.unicast_class = None

    def main(self) -> None:
        args = self.init_from_arguments(SGArguments)
        if not self.unicast_class:
            raise Exception("unknown unicast algorithm")
        # the authentic join algorithm is not implemented...
        if not args.fast_join:
            raise Exception("use --fast-join (for now)")
        self.do_exp(args)

    def do_exp(self, args: SGArguments) -> None:
        exptype = args.exp
        if exptype == 'basic':
            self.basic(args)
        elif exptype == 'unicast':
            self.unicast(args)
        elif exptype == 'unicast_vary_n':
            self.unicast_vary_n(args)
        else:
            raise Exception(f"unknown experiment: {exptype}")

    def unicast_experiment_factory(self) -> UnicastExperiment:
        return UnicastExperiment(self, unicast_class=self.unicast_class)

    def init_from_arguments(self, clazz: Type[SGArguments]) -> SGArguments:
        parser = clazz.get_parser()
        args = cast(SGArguments, parser.parse_args())
        # print(args)
        sg.VERBOSE = args.verbose
        sg.ALPHA = args.alpha
        print(f"alpha={sg.ALPHA}")
        if args.seed is not None:
            random.seed(args.seed)
            print(f"random.seed={args.seed}")
        self.unicast_class = clazz.unicast_algorithms_map().get(args.unicast_algorithm)
        if not self.unicast_class:
            raise Exception("unknown unicast algorithm")
        return args

    @classmethod
    def output_file_prefix(cls):
        return "sg"

    def basic(self, args: SGArguments) -> None:
        number_of_nodes = args.n
        fast_join = args.fast_join
        nodes = self.construct_overlay(number_of_nodes, fast_join=fast_join)
        self.do_basic_stat(nodes)
        if args.output_topology_max_level > 0:
            if args.interactive:
                filename = None
            else:
                filename = f"{self.output_file_prefix()}_topology.png"
            draw.output_topology(nodes, args.output_topology_max_level, filename=filename)

    def unicast(self, args: SGArguments) -> None:
        number_of_nodes = args.n
        fast_join = args.fast_join
        exp = self.unicast_experiment_factory()
        df = exp.unicast_exp(number_of_nodes, fast_join=fast_join)
        if args.interactive:
            filenames = (None, None)
        else:
            filenames = (f"{self.output_file_prefix()}_hops_hist.png",
                         f"{self.output_file_prefix()}_msgs_hist.png")
        exp.output_results(df, filenames=filenames)
        if args.output_hop_graph:
            exp.render_hop_graphs(diagonal=args.hop_graph_diagonal, interactive=args.interactive)

    def unicast_vary_n(self, args: SGArguments) -> None:
        results = []
        ntrials = 3
        nlist = [100, 200, 400, 800]
        for n in nlist:
            for i in range(ntrials):
                exp = self.unicast_experiment_factory()
                EventExecutor.reset()
                df = exp.unicast_exp(n, fast_join=True)
                df['n'] = n
                df['hop'] = df['nhops'].apply(lambda h: min(h))
                results.append(df)
        merged = pd.concat(results)
        print(merged.to_string())
        hops_vs_n_mean = merged.groupby('n')['hop'].mean()
        print()
        print("Average Hops")
        print(hops_vs_n_mean)
        fig = plt.figure(figsize=(10, 5))
        ax = hops_vs_n_mean.plot(fig=fig, style='ob-', logx=True, grid=True)
        ax.set_xticks(nlist)
        ax.set_xticklabels(nlist)
        plt.title("average hops vs n", size=20)
        plt.xlabel("# of nodes", size=20)
        plt.ylabel("# of hops", size=20)
        if args.interactive:
            plt.show()
        else:
            plt.savefig(f"{self.output_file_prefix()}_hops_vs_n.png")
        plt.close('all')

    NODE_INDEX_TO_KEY_FACTOR = 10
    T = TypeVar('T', bound=SGNode)

    def construct_overlay(self, number_of_nodes: int, fast_join=False, node_class: Type[T] = SGNode) -> list[T]:
        """
        construct an overlay network
        :param number_of_nodes
        :param fast_join: use fast join method rather than join()
        :param node_class: class of a node
        :return an array of SGNode that has been joined
        """
        nodes = []
        for i in range(number_of_nodes):
            mv = MembershipVector()
            # if you want to use regular membership vectors...
            # mv = MembershipVector(i)
            nodes.append(node_class(i * self.NODE_INDEX_TO_KEY_FACTOR, mv))
        dump_nodes_mv(nodes)

        if fast_join:
            node_class.fast_join_all(nodes)
        else:
            self.join_nodes_all(nodes)

        dump_nodes_routing_table(nodes)
        return nodes

    @classmethod
    def do_basic_stat(cls, nodes: list[SGNode]) -> pd.DataFrame:
        data = []
        max_length = 0
        for cur in nodes:
            s = cur.routing_table_size_per_level()
            data.append([cur.key, cur.routing_table_height(), cur.number_of_unique_nodes_in_routing_table()] + s)
            max_length = max(max_length, len(s))
        tuples = [('key', ''), ('height', ''), ('uniq', '')]
        tuples += itertools.product(['table_size'], range(0, max_length))

        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(tuples))
        df.set_index('key')
        print("Routing Table Statistics (raw)")
        print(df.to_string(index=False))
        print()
        print("Routing Table Statistics (mean)")
        m = df[['height', 'uniq', 'table_size']].mean()
        print(m.to_string())
        return df

    @classmethod
    def join_nodes_all(cls, nodes: list[SGNode]) -> None:
        introducer = nodes[0]
        introducer.initialize_as_introducer()
        for i, n in enumerate(nodes):
            if i == 0:
                continue
            ev = SchedEvent(lambda _n=n: _n.join(introducer))
            EventExecutor.register_event(ev, i * 1000)
        EventExecutor.sim(len(nodes) * 1000)
        EventExecutor.reset()


class UnicastExperiment:
    def __init__(self, main: SGMain, unicast_class: Type[UnicastBase]):
        self.nodes: list[SGNode] = []
        self.number_of_trials = 0
        self.msgs: list[UnicastBase] = []
        self.main = main
        self.unicast_class = unicast_class

    def unicast_exp(self, number_of_nodes: int, *, fast_join=False) -> pd.DataFrame:
        """
        Perform unicast experiments.
        :param number_of_nodes
        :param fast_join
        :return results
        """
        print("Unicast Experiment:")
        nodes = self.main.construct_overlay(number_of_nodes, fast_join=fast_join)

        number_of_nodes = len(nodes)
        self.nodes = nodes
        # number_of_trials = 100
        self.number_of_trials = number_of_nodes * 4
        self.msgs: list[UnicastBase] = []
        for i in range(self.number_of_trials):
            src = random.randint(0, number_of_nodes - 1)
            dst = random.randint(0, number_of_nodes * self.main.NODE_INDEX_TO_KEY_FACTOR)
            msg = self.unicast_class(nodes[src], target=dst)
            self.msgs.append(msg)
            # perform a unicast every 1000 abstract time
            EventExecutor.register_event(msg, latency=i * 1000)

        EventExecutor.sim(self.number_of_trials * 1000 * 2, verbose=sg.VERBOSE)

        data = []
        for i, msg in enumerate(self.msgs):
            if sg.VERBOSE:
                print(f"{i}: Unicast {msg.source_node}->{msg.target}"
                      f": #msgs={msg.number_of_messages}"
                      f", path lengths={msg.path_lengths}")
            data.append({"no": i,
                         "from": msg.source_node.key,
                         "to": msg.target,
                         "nhops": msg.path_lengths,
                         "nmsgs": msg.number_of_messages})
        df = pd.DataFrame(data)
        df.set_index("no")
        return df

    def output_results(self, df: pd.DataFrame, filenames: tuple[str, str], mean_columns=None) -> None:
        if mean_columns is None:
            mean_columns = ['nmsgs', 'min_hops']
        # extract min and max from 'nhops' (which is a list)
        df_min = df['nhops'].apply(lambda h: min(h))
        df_min.name = "min_hops"
        df_max = df['nhops'].apply(lambda h: max(h))
        df_max.name = "max_hops"
        # append min and max to the right
        merged = pd.concat([df, df_min, df_max], axis=1)
        print(merged.to_string(index=False))
        print("Means")
        means = merged[mean_columns].mean()
        print(means.to_frame().T.to_string(index=False))

        # generate a histogram of # of hops
        df_nhops = merged["min_hops"]
        fig = plt.figure(figsize=(10, 5))
        df_nhops.plot.hist(fig=fig, histtype='step', color="grey",
                           bins=range(0, math.ceil(df_nhops.max()) + 1), title="# of hops", density=True)
        plt.xticks(list(range(0, math.ceil(df_nhops.max()) + 1)))
        if filenames[0] is None:
            plt.show()
        else:
            plt.savefig(filenames[0])
        plt.close('all')

        # generate a histogram of # of messages
        df_nmsgs = df["nmsgs"]
        fig = plt.figure(figsize=(10, 5))
        df_nmsgs.plot.hist(fig=fig, histtype='step', color="grey",
                           bins=range(0, df_nmsgs.max()), title="# of msgs", density=True)
        # plt.xticks(list(range(0, math.ceil(df_nmsgs.max()))))
        if filenames[1] is None:
            plt.show()
        else:
            plt.savefig(filenames[1])
        plt.close('all')

    hop_graph_number = 0

    def render_hop_graphs(self, diagonal=False, interactive=False) -> None:
        for i, m in enumerate(self.msgs):
            print(f"{i}: {m.source_node}->{m.target}")
            if interactive:
                filename = None
            else:
                filename = f"unicast-{m.short_name()}-{self.hop_graph_number}.png"
                self.hop_graph_number += 1
            draw.render_hop_graph(m, self.nodes, diagonal=diagonal, filename=filename)


def dump_nodes_mv(nodes: list[SGNode]) -> None:
    for i, n in enumerate(nodes):
        print(f"node[{i}]={repr(n)}")


def dump_nodes_routing_table(nodes: list[SGNode]) -> None:
    for n in nodes:
        print(f"{n}: {n.mv}")
        print("  ", "\n  ".join(n.routing_table_string()), sep='')
        print(f"  # of unique nodes: {n.number_of_unique_nodes_in_routing_table()}")
    print()


if __name__ == "__main__":
    SGMain().main()
