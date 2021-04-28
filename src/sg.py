from __future__ import annotations

import asyncio
import copy
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import cast, Optional, TypeVar

import networkx as nx

from discrete_ev_sim import Event, AbstractNode
from utils import list_str, verbose

DEFAULT_N = 8
DEFAULT_ALPHA = 2

# base of a membership vector
ALPHA = 2
MEMBERSHIP_VECTOR_DIGITS = 32
VERBOSE = False


class MembershipVector:
    def __init__(self, value: Optional[int] = None):
        self.val: list[int] = []
        if value is None:
            for i in range(MEMBERSHIP_VECTOR_DIGITS):
                self.val.append(random.randint(0, ALPHA - 1))
        else:
            for i in range(MEMBERSHIP_VECTOR_DIGITS):
                self.val.append(value % ALPHA)
                value //= ALPHA

    def __str__(self):
        return "".join(map(str, self.val))

    def common_prefix_length(self, another: MembershipVector):
        for i in range(MEMBERSHIP_VECTOR_DIGITS):
            if self.val[i] != another.val[i]:
                return i
        return MEMBERSHIP_VECTOR_DIGITS

    def reverse_prefix(self, number_of_digits: int) -> None:
        """
        Reverse the first 'number_of_digits' elements in the membership vector
        :param number_of_digits
        """
        if number_of_digits > 0:
            self.val = self.val[number_of_digits-1::-1] + self.val[number_of_digits::]


class Direction(IntEnum):
    RIGHT = 0
    LEFT = 1

    def step(self) -> int:
        return 1 if self == Direction.RIGHT else -1

    def flip(self) -> Direction:
        return Direction.LEFT if self == Direction.RIGHT else Direction.RIGHT


RIGHT = Direction.RIGHT
LEFT = Direction.LEFT


class RoutingTableSingleLevel:
    def __init__(self, owner: SGNode, level: int):
        self.owner = owner
        # [[RIGHT], [LEFT]]
        self.neighbors: list[list[SGNode]] = [[], []]
        self.level = level

    def add(self, d: Direction, u: SGNode) -> None:
        lst = self.neighbors[d]
        if u in lst:
            return
            # raise Exception(f"{str(u)} is already added!: {lst}")
        lst.append(u)
        sort_circular(self.owner.key, lst)
        if d == LEFT:
            lst.reverse()


class SGNode(AbstractNode):
    """
    Simple skip graph node
    """
    def __init__(self, key: int, mv: MembershipVector):
        self.routing_table: list[RoutingTableSingleLevel] = []
        self.key = key
        self.mv = mv

    def routing_table_factory(self, level: int) -> RoutingTableSingleLevel:
        return RoutingTableSingleLevel(self, level)

    def __str__(self):
        return f"N{self.key}"

    def __repr__(self):
        return f"Node(key={self.key}, mv={self.mv})"

    def initialize_as_introducer(self) -> None:
        self.extend_routing_table(0)

    @staticmethod
    def fast_join_all(nodes: list[SGNode]) -> None:
        number_of_nodes = len(nodes)

        # initialize level 0 ring
        for _n in nodes:
            _n.extend_routing_table(0)
        for i, p in enumerate(nodes):
            q = nodes[(i + 1) % number_of_nodes]
            p.routing_table[0].neighbors[RIGHT].append(q)
            q.routing_table[0].neighbors[LEFT].append(p)

        # Algorithm overview:
        # - loop over all levels (terminates when all nodes become singleton)
        #  - loop over all prefixes in the current level (terminates when all prefixes are done)
        #    - loop over all nodes that share the current prefix (terminates when circulated on the current prefix ring)
        unfinished = set(nodes)   # nodes that are not singleton yet
        for level in range(MEMBERSHIP_VECTOR_DIGITS):
            # print(f"*** level={level}, unfinished={list_str(unfinished)}")
            nodes_at_current_level = set(unfinished)
            while len(nodes_at_current_level) > 0:  # loop for all prefixes
                # start from any node from nodes_at_current_level
                p = start = next(iter(nodes_at_current_level))
                # print(f"start from {p}")
                while True:
                    # from 'start', 'p' moves on the ring rightward at level 'level',
                    # initializing the routing table at level 'level+1'
                    nodes_at_current_level.remove(p)
                    # traverse rightward until we find a node whose MV matches at least level+1 digits with 'p'
                    q = p.routing_table[level].neighbors[RIGHT][0]
                    while q is not p and p.mv.common_prefix_length(q.mv) <= level:
                        # print(f"q={q}")
                        q = q.routing_table[level].neighbors[RIGHT][0]
                    if q is not p:
                        p.extend_routing_table(level + 1)
                        p.routing_table[level + 1].neighbors[RIGHT] = [q]
                        q.extend_routing_table(level + 1)
                        q.routing_table[level + 1].neighbors[LEFT] = [p]
                    else:
                        # print("removed from unfinished")
                        unfinished.remove(p)
                    # forward rightward
                    p = q
                    if p is start:
                        break
            if len(unfinished) == 0:
                break
        if len(unfinished) != 0:
            raise Exception("insufficient membership vector digits!")

    async def join(self, introducer: SGNode) -> None:
        raise NotImplementedError()

    def extend_routing_table(self, level: int) -> None:
        """ make sure that routing_table has index 'level'"""
        while len(self.routing_table) <= level:
            max_level = len(self.routing_table) - 1
            new_level = max_level + 1
            s = self.routing_table_factory(new_level)
            self.routing_table.append(s)
            # XXX: BFT only
            if max_level >= 0:
                # 経路表のレベルが i から i+1 になるとき，レベルiでMVがi+1桁以上一致しているノードをレベルi+1にコピーする
                for n in self.routing_table[max_level].neighbors[RIGHT] + self.routing_table[max_level].neighbors[LEFT]:
                    if n.mv.common_prefix_length(self.mv) >= new_level:
                        s.add(RIGHT, n)
                        s.add(LEFT, n)

    def routing_table_string(self) -> list[str]:
        buf = []
        for i, single_level in enumerate(self.routing_table):
            s = f"Level {i}: "
            s += f"LEFT={['N' + str(s.key) for s in reversed(single_level.neighbors[LEFT])]}, "
            s += f"RIGHT={['N' + str(s.key) for s in single_level.neighbors[RIGHT]]}"
            buf.append(s)
        return buf

    def routing_table_height(self) -> int:
        return len(self.routing_table)

    def routing_table_size_per_level(self) -> [int]:
        sizes = []
        for i in range(self.routing_table_height()):
            s = len(self.routing_table[i].neighbors[RIGHT]) + len(self.routing_table[i].neighbors[LEFT])
            sizes.append(s)
        return sizes

    def number_of_unique_nodes_in_routing_table(self) -> int:
        nodes = set()
        for t in self.routing_table:
            nodes |= set(t.neighbors[RIGHT])
            nodes |= set(t.neighbors[LEFT])
        return len(nodes)

    def highest_level_in_routing_table(self, node: SGNode) -> Optional[int]:
        for i, t in reversed(list(enumerate(self.routing_table))):
            if node in t.neighbors[RIGHT] or node in t.neighbors[LEFT]:
                return i
        return None

    def handle_unicast(self, msg: UnicastBase) -> None:
        next_nodes = msg.find_next_hops()

        for n, sub_msg in next_nodes:
            if n is self:
                verbose(f"I'm one of the destination node, path={list_str(msg.path)}")
                # TODO: implement UnicastReply
                msg.root.results.append(self)
                if len(msg.root.results) == msg.root.expected_number_of_results:
                    msg.root.future.set_result(None)
                msg.root.destinations.append(self)
                path_length = len(msg.path) - 1
                msg.root.path_lengths.append(path_length)
                continue
            else:
                self.send_event(sub_msg)


class UnicastBase(Event, ABC):
    next_message_id = 1

    def __init__(self, receiver: SGNode, target: int):
        super().__init__(receiver)
        self.source_node = receiver
        self.target = target
        self.message_id = UnicastBase.next_message_id
        UnicastBase.next_message_id += 1
        self.path = [receiver]  # nodes that forward this message
        self.render_level = 0
        self.hop = 0
        self.children: list[UnicastBase] = []
        self.root = self

        # the following part is used by a root message only
        self.expected_number_of_results = 1
        self.results: list[SGNode] = []
        self.future = asyncio.get_event_loop().create_future()
        self.destinations: list[SGNode] = []     # targetの近隣Kノードの配列
        self.number_of_messages = 0
        self.path_lengths: list[int] = []
        self.graph = nx.DiGraph()
        self.graph.add_node(receiver.key, hop=self.hop)

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}(receiver={self.receiver}, target={self.target}, " +\
               f"msgid={self.message_id}, path={list_str(self.path)})"

    def short_name(self) -> str:
        return self.__class__.__name__.replace("Unicast", "").lower()

    def create_sub_message(self, next_hop: SGNode, render_level=0, **kwargs) -> UnicastBase:
        sub = copy.copy(self)
        sub.receiver = next_hop
        sub.path = copy.copy(self.path)
        sub.path.append(next_hop)
        sub.hop = self.hop + 1
        sub.render_level = render_level
        sub.children = []
        self.root.number_of_messages += 1
        self.children.append(sub)
        # for drawing graphs
        g = self.root.graph
        g.add_edge(self.receiver.key, next_hop.key)
        return sub

    async def run(self, node: SGNode):
        return node.handle_unicast(self)

    @abstractmethod
    def find_next_hops(self) -> list[tuple[SGNode, Optional[UnicastBase]]]:
        raise Exception("subclass must implement find_next_hops()")


class UnicastGreedy(UnicastBase):
    # override
    def find_next_hops(self) -> list[tuple[SGNode, Optional[UnicastGreedy]]]:
        my_node = self.receiver
        # find the closest node to the target key from all levels
        nodes = [k for table in my_node.routing_table for d in [LEFT, RIGHT] for k in table.neighbors[d]]
        nodes.append(my_node)
        next_node = closest_node(self.target, nodes)
        render_level = my_node.highest_level_in_routing_table(next_node) or 0
        verbose(f"next hop for target {self.target} is {next_node}")
        if next_node is my_node:
            next_nodes = [(my_node, None)]
        else:
            next_nodes = [(next_node, self.create_sub_message(next_node, render_level=render_level))]
        return next_nodes


class UnicastOriginal(UnicastBase):
    """
    search algorithm based on the following paper:
    James Aspnes and Gauri Shah. 2007. Skip graphs. ACM Trans. Algorithms 3, 4 (November 2007), pp. 1-25,
    DOI:https://doi.org/10.1145/1290672.1290674
    """
    def __init__(self, receiver: SGNode, target: int):
        super().__init__(receiver, target)
        self.level = self.render_level = receiver.routing_table_height() - 1

    # override
    def find_next_hops(self) -> list[tuple[SGNode, Optional[UnicastOriginal]]]:
        def next_msg(_node: SGNode, _level: int) -> tuple[SGNode, UnicastOriginal]:
            return _node, self.create_sub_message(_node, level=_level, render_level=_level)
        my_node = self.receiver
        level = self.level
        if my_node.key <= self.target:
            while level >= 0:
                right_node = my_node.routing_table[level].neighbors[RIGHT][0]
                if my_node.key < right_node.key <= self.target:
                    return [next_msg(right_node, level)]
                level -= 1
            return [(my_node, None)]
        else:
            left_node = None
            while level >= 0:
                left_node = my_node.routing_table[level].neighbors[LEFT][0]
                if self.target <= left_node.key < my_node.key:
                    return [next_msg(left_node, level)]
                level -= 1
            return [next_msg(left_node, 0)]

    def create_sub_message(self, next_hop: SGNode, render_level=0, **kwargs) -> UnicastOriginal:
        sub = cast(UnicastOriginal, super().create_sub_message(next_hop, render_level=render_level))
        sub.level = kwargs['level']
        return sub


T = TypeVar('T', bound=SGNode)


def sort_circular(base: int, nodes: list[T]) -> None:
    """
    Sort 'nodes' destructively in a circular key space so that 'base' is the maximum element
    :param base
    :param nodes
    """
    max_node = max(nodes, key=lambda x: x.key)
    nodes.sort(key=lambda x: x.key + max_node.key + 1 if x.key <= base else x.key)


def closest_node(target: int, nodes: list[T]) -> T:
    """
    Returns the closest node to 'target' from 'nodes' in a circular key space
    :param target
    :param nodes
    """
    sorted_nodes = nodes.copy()
    sort_circular(target, sorted_nodes)
    return sorted_nodes[-1]
