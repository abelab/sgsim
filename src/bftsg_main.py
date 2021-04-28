from __future__ import annotations
import argparse
import random
import sys
from typing import cast, Type

import pandas as pd

import bftsg
from bftsg import (DEFAULT_K, BFTNode, BFTUnicastBase, BFTUnicastMulti, BFTUnicastSingle)
from sg import MembershipVector
from sg_main import SGMain, UnicastExperiment, dump_nodes_mv, SGArguments

DEFAULT_FAILURE_RATE = 1 / 3


class BFTArguments(SGArguments):
    def __init__(self):
        super().__init__()
        self.k = 0
        self.failure_rate = 0.0
        self.reachability = False
        self.precise = False
        self.asymmetric = False

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = super().get_parser()
        parser.add_argument('-k', help=f'the value of K (default: {DEFAULT_K})', default=DEFAULT_K, type=int)
        parser.add_argument('-f', '--failure-rate', help=f'failure rate (default: {DEFAULT_FAILURE_RATE})',
                            dest='failure_rate', default=DEFAULT_FAILURE_RATE, type=float)
        parser.add_argument('--reachability', help=f'compute message reachability',
                            action='store_true', dest='reachability')
        parser.add_argument('--precise', help=f'compute precisely (do not use Monte Carlo)',
                            action='store_true', dest='precise')
        parser.add_argument('--asymmetric', help=f'use old asymmetric routing table (must be used with --fast-join)',
                            action='store_true', dest='asymmetric')
        return parser

    @classmethod
    def unicast_algorithms_map(cls) -> dict[str, Type[BFTUnicastBase]]:
        return {
            'single': BFTUnicastSingle,
            'multi': BFTUnicastMulti
        }


class BFTMain(SGMain):
    def __init__(self):
        super().__init__()
        self.failure_rate = 0.0
        self.compute_reachability = False
        self.compute_precise = False

    def main(self):
        args = cast(BFTArguments, self.init_from_arguments(BFTArguments))
        self.failure_rate = args.failure_rate
        self.compute_reachability = args.reachability
        self.compute_precise = args.precise
        fast_join = args.fast_join
        bftsg.init_k(args.k)
        bftsg.SYMMETRIC_ROUTING_TABLE = not args.asymmetric
        if not bftsg.SYMMETRIC_ROUTING_TABLE and not fast_join:
            raise Exception("You cannot use asymmetric routing table without --fast-join")

        self.do_exp(args)

    def unicast_experiment_factory(self) -> BFTUnicastExperiment:
        return BFTUnicastExperiment(self,
                                    unicast_class=self.unicast_class,
                                    failure_rate=self.failure_rate,
                                    compute_reachability=self.compute_reachability,
                                    compute_precise=self.compute_precise)

    @classmethod
    def output_file_prefix(cls):
        return "bftsg"

    # override
    def construct_overlay(self, number_of_nodes: int,
                          node_class=BFTNode,
                          fast_join=False) -> list[BFTNode]:
        return super().construct_overlay(number_of_nodes, fast_join=fast_join, node_class=node_class)


class BFTUnicastExperiment(UnicastExperiment):
    msgs: list[BFTUnicastBase]  # type hint

    def __init__(self, main: BFTMain,
                 unicast_class: Type[BFTUnicastBase],
                 failure_rate=0.0,
                 compute_reachability=False,
                 compute_precise=False):
        super().__init__(main, unicast_class)
        self.failure_rate = failure_rate
        self.compute_reachability = compute_reachability
        self.compute_precise = compute_precise

    def unicast_exp(self, number_of_nodes: int, *, fast_join=False, unicast_class=None) -> pd.DataFrame:
        df = super().unicast_exp(number_of_nodes, fast_join=fast_join)

        # count the number of duplicated (redundantly-received) messages
        for i, m in enumerate(self.msgs):
            df.loc[(df['from'] == m.source_node.key) & (df['to'] == m.target), 'ndup'] = m.number_of_duplicated_messages

        #
        # Compute a message reachability
        #
        if not self.compute_reachability:
            return df
        for i, m in enumerate(self.msgs):
            if i % 100 == 0 and i > 0:
                cur = df['reachability'].mean()
                print(f"#{i}/{self.number_of_trials}: reachability={cur}", file=sys.stderr)
            if self.compute_precise:
                prob = m.compute_probability_precise(self.failure_rate)
            else:
                prob = m.compute_probability_monte_carlo(self.failure_rate)
            df.loc[(df['from'] == m.source_node.key) & (df['to'] == m.target), 'reachability'] = prob

        return df

    def output_results(self, df: pd.DataFrame, filenames=tuple[str, str], mean_columns: list[str] = None) -> None:
        mean_columns = ['nmsgs', 'ndup', 'min_hops', 'max_hops']
        if 'reachability' in df:
            mean_columns.append('reachability')
        super().output_results(df, filenames=filenames, mean_columns=mean_columns)


def compare_join_test() -> None:
    """ fast_join と join の結果を比較 """
    number_of_nodes = 100
    fast_nodes = []
    normal_nodes = []
    random.seed(0)
    for i in range(number_of_nodes):
        mv = MembershipVector()
        fast_nodes.append(BFTNode(i, mv))
        normal_nodes.append(BFTNode(i, mv))
    BFTNode.fast_join_all(fast_nodes)
    BFTMain.join_nodes_all(normal_nodes)
    dump_nodes_mv(fast_nodes)
    for a, b in zip(fast_nodes, normal_nodes):
        a_str = "\n".join(a.routing_table_string())
        b_str = "\n".join(b.routing_table_string())
        print(f"{a}:{' (DIFFERS)' if a_str != b_str else ''}")
        print("cheat")
        print("  ", "\n  ".join(a.routing_table_string()), sep="")
        print("ordinal")
        print("  ", "\n  ".join(b.routing_table_string()), sep="")


if __name__ == "__main__":
    BFTMain().main()
