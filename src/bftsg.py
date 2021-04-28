from __future__ import annotations

import random
from abc import ABC
from typing import Optional, TypeVar

import networkx as nx

import sg
from sg import (MembershipVector, sort_circular, RoutingTableSingleLevel, Direction, LEFT, RIGHT,
                SGNode, UnicastBase, MEMBERSHIP_VECTOR_DIGITS)
from space import is_ordered
from utils import list_str, verbose

DEFAULT_K = 2
K = 0
LEFT_HALF_K = 0
RIGHT_HALF_K = 0

# 対称な経路表を使うかどうか
# 対称の場合:
# - レベルiの経路表では，要素ノードのMVのi+1桁目の値(0〜α-1)をカウントし，それぞれについてK-1個以上存在するまでvisitする．
# - 経路表はsingleton になるまで構築する．
SYMMETRIC_ROUTING_TABLE = True


def init_k(k: int):
    def halves_of_k() -> tuple[int, int]:
        """
        Kを左右に2分割した値を返す．奇数のときは左側が右側より1多い．
        """
        return (k // 2, k // 2) if k % 2 == 0 else ((k + 1) // 2, (k + 1) // 2 - 1)

    global K, LEFT_HALF_K, RIGHT_HALF_K
    K = k
    LEFT_HALF_K, RIGHT_HALF_K = halves_of_k()


init_k(DEFAULT_K)


class BFTRoutingTableSingleLevel(RoutingTableSingleLevel):
    owner: BFTNode                  # type hint
    neighbors: list[list[BFTNode]]

    # override
    def add(self, d: Direction, u: BFTNode) -> None:
        super().add(d, u)
        # trim the list beyond the (K-1)th node whose MV matches at least self.level+1 digits
        i = self.satisfaction_index(d)
        if i is not None:
            self.neighbors[d] = self.neighbors[d][0:i+1]

    def has_sufficient_nodes(self, d: Direction) -> bool:
        """ MVのlevel桁目 ∈ α をカウントし，すべてのαについてK-1個以上含まれているかを判定 """
        return self.satisfaction_index(d) is not None

    def satisfaction_index(self, d: Direction) -> Optional[int]:
        """ d方向の経路表で，経路表の条件を満足するノードが何番目のインデックスにあるかを返す．"""
        lst = self.neighbors[d]
        counts = [0] * sg.ALPHA
        next_digit = self.owner.mv.val[self.level]
        for i, n in enumerate(lst):
            digit = n.mv.val[self.level]
            counts[digit] += 1
            if SYMMETRIC_ROUTING_TABLE and all(c >= K - 1 for c in counts) or\
               not SYMMETRIC_ROUTING_TABLE and counts[next_digit] >= K - 1:
                return i
        return None

    def has_duplicates_in_lefts_and_rights(self) -> bool:
        return not set(self.neighbors[RIGHT]).isdisjoint(self.neighbors[LEFT])

    def pickup_k_nodes(self, target: int) -> Optional[list[BFTNode]]:
        """ targetの近傍k個を返す """
        nodes = self.concatenate()
        if self.has_duplicates_in_lefts_and_rights():
            return closest_k_nodes(target, nodes)
        else:
            # if LEFT_HALF_K == RIGHT_HALF_K == 2 and nodes=[0, 1, 2, 3, 4, 5] (len=6),
            # i moves between [1, 3]
            verbose(f"pickup_k_nodes: target={target}, nodes={list_str(nodes)}")
            for i in range(LEFT_HALF_K-1, len(nodes)-RIGHT_HALF_K):
                cur_node = nodes[i].key
                next_node = nodes[i + 1].key
                if is_ordered(cur_node, True, target, next_node, False):
                    return nodes[i-LEFT_HALF_K+1:i+RIGHT_HALF_K+1]
            return None

    def concatenate(self) -> list[T]:
        return list(reversed(self.neighbors[LEFT])) + [self.owner] + self.neighbors[RIGHT]


class BFTNode(SGNode):
    routing_table: list[BFTRoutingTableSingleLevel]     # type hint

    def __init__(self, key: int, mv: MembershipVector):
        super().__init__(key, mv)
        self.query_seen = set()

    # override
    def routing_table_factory(self, level: int) -> BFTRoutingTableSingleLevel:
        return BFTRoutingTableSingleLevel(self, level)

    @staticmethod
    def fast_join_all(nodes: list[T]) -> None:
        def init():
            number_of_nodes = len(nodes)
            for _n in nodes:
                _n.extend_routing_table(0)
            for i, p in enumerate(nodes):
                q = nodes[(i + 1) % number_of_nodes]
                p.routing_table[0].neighbors[RIGHT].append(q)
                q.routing_table[0].neighbors[LEFT].append(p)

        def fast_join_all_sub(_dir: Direction, _nodes: list[T]) -> None:
            # アルゴリズムの概略:
            # - すべてlevelに関する繰り返し (すべてのノードがsingletonになったら終了)
            #   - 当該levelのすべてのprefixに関する繰り返し (すべてのprefixが処理し終わったら終了)
            #     - 当該prefixのすべてのノードに関する繰り返し (当該prefixで一周したら終了)
            unfinished = set(_nodes)
            # Invariants: unfinishedの各ノードは，最大レベルでd方向のノードが最低1つは設定済み
            for level in range(MEMBERSHIP_VECTOR_DIGITS):
                # print(f"*** level={level}, unfinished={unfinished}")
                nodes_at_current_level = set(unfinished)
                while len(nodes_at_current_level) > 0:  # loop for all prefixes
                    # start from any node from nodes_at_current_level
                    buf: list[list[BFTNode]] = [[] for _ in range(sg.ALPHA)]
                    p = start = next(iter(nodes_at_current_level))
                    q = p.routing_table[level].neighbors[_dir][0]
                    # print(f"start from {p}")
                    while True:  # pからレベル=levelでd方向にトラバース
                        # print(f"p={p}")
                        nodes_at_current_level.remove(p)
                        try:
                            buf[p.mv.val[level]].remove(p)
                        except ValueError:
                            pass
                        while q is not p and not all(len(lst) >= K - 1 for lst in buf):
                            d = q.mv.val[level]
                            # print(f"d={d}, q={q}, buf={buf}")
                            buf[d].append(q)
                            q = q.routing_table[level].neighbors[_dir][0]
                        # print([f'buf[{i}]={simple_list_string(buf[i])}' for i in range(ALPHA)])
                        merged = list(set([node for b in buf for node in b]))
                        if not SYMMETRIC_ROUTING_TABLE:
                            # asymmetricの場合，bufに余分なノードが入っている可能性があるため，mergedから削除
                            if len(buf[p.mv.val[level]]) > K-2:
                                last = buf[p.mv.val[level]][K-2]
                                if _dir == RIGHT:
                                    merged = list(filter(
                                        lambda _n: is_ordered(p.key, False, _n.key, last.key, True), merged))
                                else:
                                    merged = list(filter(
                                        lambda _n: is_ordered(last.key, True, _n.key, p.key, False), merged))
                            # print(f"last={last.key}, merged={merged}")
                        sort_circular(p.key, merged)
                        if _dir == LEFT:
                            merged.reverse()
                        p.routing_table[level].neighbors[_dir] = merged
                        upper_buf = buf[p.mv.val[level]]
                        if len(upper_buf) > 0:
                            p.extend_routing_table(level + 1)
                            p.routing_table[level + 1].neighbors[_dir] = upper_buf.copy()
                            # print(p.routing_table_string())
                        else:
                            # print("removed from unfinished")
                            unfinished.remove(p)
                        # forward rightward
                        p = p.routing_table[level].neighbors[_dir][0]
                        if p is start:
                            break
                if len(unfinished) == 0:
                    break
            if len(unfinished) != 0:
                raise Exception("insufficient membership vector digits!")

        init()
        fast_join_all_sub(RIGHT, nodes)
        fast_join_all_sub(LEFT, nodes)
        if not SYMMETRIC_ROUTING_TABLE:
            for n in nodes:
                n.trim_routing_table()
        # print("FINISHED!")
        # dump_nodes_routing_table(nodes)

    # override
    async def join(self, introducer: T) -> None:
        """
        join this node using 'introducer'.
        :param introducer: introducer
        """
        async def set_result():
            if not msg.future.done():
                msg.future.set_result(None)

        def may_contain(_level: int, _d: Direction, _u: BFTNode) -> bool:
            """ 経路表のレベル '_level', 方向 '_d' が，'_u' を含む可能性があるかどうかを判定 """
            self.extend_routing_table(_level)
            lst = self.routing_table[_level].neighbors[_d]
            if not self.routing_table[_level].has_sufficient_nodes(_d):
                debug(f"may_contain: level={_level}, d={_d}, u={_u}: True (1)")
                return True
            if _d == RIGHT and is_ordered(self.key, False, _u.key, lst[-1].key, False) or \
               _d == LEFT and is_ordered(lst[-1].key, False, _u.key, self.key, False):
                debug(f"may_contain: level={_level}, d={_d}, u={_u}: True (2)")
                return True
            debug(f"may_contain: level={_level}, d={_d}, u={_u}: {lst}, False")
            return False

        def possibly_be_used(_d: Direction, _u: BFTNode) -> bool:
            """ 経路表の方向 _d のすべてのレベルで，_u を含む可能性があるかどうかを判定 """
            _common_prefix_len = self.mv.common_prefix_length(_u.mv)
            for _i in range(_common_prefix_len + 1):
                if may_contain(_i, _d, _u):
                    return True
            debug(f"possibly_be_used({_d}, {_u})=False")
            return False

        def debug(*msgs, **kwargs) -> None:
            # print(*msgs, **kwargs)
            pass

        # 自分のキーの近傍K個のノードを得る
        msg = BFTUnicastSingle(introducer, self.key)    # XXX: or Multi?
        self.send_event(msg)
        unicast_timeout = 100               # K個のメッセージが得られない場合のタイムアウト値
        self.sched(unicast_timeout, set_result)
        await msg.future                    # Unicastの結果を待つ
        debug(f"better_join: results={msg.results}")

        candidates = set(msg.results)       # 候補ノード．unicastの結果で初期化する．
        done = set()                        # processed nodes
        accessed = set()                    # nodes that are accessed to fetch routing table (just for statistics)
        failed = set()                      # nodes that are failed (not used for now)
        # 右と左を交互に処理する
        d = RIGHT                           # current direction
        while len(candidates) > 0:          # 候補ノードが残っている限り繰り返し
            ordered = list(candidates)
            sort_circular(self.key, ordered)    # 自分のキーが最大となるようにソート ([5 6 7 1 2 3] if self.key==4)
            # d方向のときはd方向で最もキーが近いノードを選ぶ (d=RIGHT or LEFT)
            u = ordered[0] if d == RIGHT else ordered[-1]
            debug(f"d={d}, ordered={list_str(ordered)}, u={u}")
            # d方向の経路表でuがどこかのレベルに入る?
            if possibly_be_used(d, u):
                # uを経路表に入れる（入るならば左右両方に）．またuから経路表を取得しcandidatesに入れる
                self.add_node_to_routing_table(u, trim=False)
                debug(f"{u} is added")
                # fetch routing table from u (TODO: use message!)
                remote_candidates = set(sum([s.concatenate() for s in u.routing_table], []))
                remote_candidates -= done   # 処理済みのノードは削除
                remote_candidates -= failed
                candidates |= remote_candidates    # candidatesに追加
                # update u's routing table (TODO: use message!)
                u.add_node_to_routing_table(self, trim=not SYMMETRIC_ROUTING_TABLE)
                accessed.add(u)
                delete_u = True
            else:
                # 反対方向で経路表に載る可能性があればcandidatesに保持．なければ破棄．
                delete_u = not possibly_be_used(d.flip(), u)
            if delete_u:
                done.add(u)
                candidates.remove(u)
            d = d.flip()                    # 方向を反対に
        if not SYMMETRIC_ROUTING_TABLE:
            self.trim_routing_table()           # 左右で重複するエントリが存在する最低レベルより上を削除
        debug(f"joined {repr(self)}")
        debug("\n".join(self.routing_table_string()))
        uniq = self.number_of_unique_nodes_in_routing_table()
        if uniq != len(accessed):
            # この条件が成立することがあるのか不明...
            debug(f"join: not optimal! unique nodes={uniq}, "
                  f"accessed={list_str(sorted(accessed, key=lambda x: x.key))}")

    def add_node_to_routing_table(self, u: BFTNode, trim=True) -> None:
        """
        Add node 'u' to the routing table
        :param u: the node to be added
        :param trim: true to trim unnecessary levels in the routing table
        """
        common_prefix_len = self.mv.common_prefix_length(u.mv)
        self.extend_routing_table(common_prefix_len)
        for i in range(common_prefix_len + 1):
            self.routing_table[i].add(RIGHT, u)
            self.routing_table[i].add(LEFT, u)
        if trim:
            self.trim_routing_table()
            if not self.routing_table[self.routing_table_height()-1].has_duplicates_in_lefts_and_rights():
                print(f"{self}: add_node_to_routing_table(u={u})")
                print("after")
                print("\n".join(self.routing_table_string()))
                raise Exception("Err!")
        # print(f"add_node_to_routing_table@{str(self)}: added {str(u)}, {self.routing_table_string()}")

    def trim_routing_table(self) -> None:
        """ 左右で重複するエントリがある最低レベルより上のレベルを削除する """
        for i, t in enumerate(self.routing_table):
            if t.has_duplicates_in_lefts_and_rights():
                del self.routing_table[i:]
                return

    # override
    def handle_unicast(self, msg: BFTUnicastBase) -> None:
        if msg.message_id in self.query_seen:
            verbose(f"N{self.key}: handle_unicast: already seen")
            msg.root.number_of_duplicated_messages += 1
            return
        self.query_seen.add(msg.message_id)
        super().handle_unicast(msg)


class BFTUnicastBase(UnicastBase, ABC):
    root: BFTUnicastBase    # type hint
    results: list[BFTNode]
    destinations: list[BFTNode]

    def __init__(self, receiver: BFTNode, target: int):
        super().__init__(receiver, target)
        # the following part is used by a root message only
        self.expected_number_of_results = K
        self.number_of_duplicated_messages = 0
        # targetの近傍Kノードにおける，キーkに対する受信確率
        self.receive_probabilities: dict[int, float] = {}

    # override
    def short_name(self) -> str:
        return self.__class__.__name__.replace("BFTUnicast", "").lower()

    def next_msg(self, _node: BFTNode, render_level: int) -> tuple[BFTNode, Optional[BFTUnicastBase]]:
        if _node is self.receiver:
            return _node, None
        else:
            return _node, self.create_sub_message(_node, render_level=render_level)

    def compute_probability_monte_carlo(self, failure_rate: float) -> float:
        """
        モンテカルロ法によりメッセージ受信確率を求める
        :param failure_rate: 想定するノードの故障率
        :return: 受信確率
        """
        # 方針:
        # - すべてのエッジからグラフを作る．
        # - failure_rateに従って故障させる (グラフからノードを削除)
        # - 起点ノードから目的ノードに到達できるかを判定
        src = self.source_node.key
        dest_keys = [n.key for n in self.destinations]
        number_of_success = 0
        nodes = list(self.graph.neighbors)
        number_of_trials = min(2**len(nodes), 1000)
        for i in range(number_of_trials):
            g = self.graph.copy()
            for n in nodes:
                if n != src and random.random() < failure_rate:
                    g.remove_node(n)
            if any(g.has_node(d) and nx.algorithms.shortest_paths.generic.has_path(g, src, d) for d in dest_keys):
                number_of_success += 1
        probability = number_of_success / number_of_trials
        if sg.VERBOSE:
            print(f"Computing {src}->{dest_keys}")
            print(f"  edges: {list(self.graph.edges)}, loop={number_of_trials}, reachability={probability}")
            # connectivity = nx.node_connectivity(self.graph, src, dest_keys[0])
            # print(f"connectivity = {connectivity}")
        return probability

    def compute_probability_precise(self, failure_rate: float) -> float:
        """
        正確にメッセージ受信確率を求める
        :param failure_rate: 想定するノードの故障率
        :return: 受信確率
        """
        # 考え方: 検索に失敗するすべてのノード組み合わせを列挙し，その組み合わせが発生する確率の合計を求める．これが到達失敗率となる．
        # nodes[0]が始点ノード，nodes[1]... はその他のノードとする．
        # S: 始点ノード，O: 正常ノード，X: 故障ノード，*: 未定ノードとする．
        # - nodes[1]から順番に故障させた場合の到達失敗率の和を求める．すなわち，SX***, SOX**, SOOX*, SOOOX の場合の合計．
        # - SX*** の計算:
        #   - nodes[1]を故障させた場合に目的ノード群に経路が存在するかを判定．
        #   - 経路が1つも存在しない場合，SX***が発生する確率を返す(この場合F)．
        #   - 経路が1つでも存在する場合，nodes[2]から順番に故障させた場合の到達失敗率の和を求める．すなわち，SXX**, SXOX*, SXOOX の場合の合計．
        # - SOX** の計算
        #   - nodes[2]を故障させた場合に目的ノード群に経路が存在するかを判定．
        #   - 経路が1つも存在しない場合，SOX***が発生する確率を返す(この場合(1-F)*F)．
        #   - 経路が1つでも存在する場合，nodes[3]から順番に故障させた場合の到達失敗率の和を求める．すなわち，SOXX*, SOXOX の場合の合計．
        def compute(g: nx.Graph, i: int, prob: float) -> float:
            nonlocal count
            count += 1
            c = g.copy()
            if i > 0:
                c.remove_node(nodes[i])
                prob *= failure_rate
                if not any(c.has_node(d) and nx.algorithms.shortest_paths.generic.has_path(c, src, d)
                           for d in dest_keys):
                    return prob
                # shortest_paths.generic.has_path() is faster than this:
                # connected = set(nx.node_connected_component(c, src))
                # if connected.isdisjoint(dest_keys):
                #     return prob
            s = 0
            for j in range(i+1, len(nodes)):
                s += compute(c, j, prob)
                prob *= 1-failure_rate
            return s

        count = 0
        src = self.source_node.key
        nodes = list(self.graph.neighbors())
        nodes.remove(src)
        nodes = [src] + nodes
        dest_keys = [n.key for n in self.destinations]
        p = compute(self.graph, 0, 1.0)
        # p = compute(self.graph.to_undirected(), 0, 1.0)
        if sg.VERBOSE:
            print(f"edges={self.graph.edges}")
            print(f"nodes={nodes}, src={src}, dest_keys={dest_keys}, reachability={1-p}, "
                  f"count={count}/{2**(len(nodes)-1)}")
        return 1-p


class BFTUnicastSingle(BFTUnicastBase):
    """
    single: targetの近傍ノードをK個含む最低のレベルを使ってルーティング
    """
    # override
    def find_next_hops(self) -> list[tuple[BFTNode, Optional[BFTUnicastSingle]]]:
        my_node = self.receiver
        for i, single_level in enumerate(my_node.routing_table):
            k_nodes = single_level.pickup_k_nodes(self.target)
            if k_nodes is not None:
                level = i
                break
        else:
            k_nodes = my_node.routing_table[0].concatenate()
            level = 0
        k_nodes = list(filter(lambda n: n is my_node or n not in self.path, k_nodes))
        verbose(f"next hops for target {self.target} are {list_str(k_nodes)} (level {level})")
        next_nodes = [self.next_msg(n, level) for n in k_nodes]
        if self.root is self:
            self.render_level = level + 1
        return next_nodes


class BFTUnicastMulti(BFTUnicastBase):
    """
    multi: すべてのレベルからtargetの近傍ノードをK個選ぶ
    """
    # override
    def find_next_hops(self) -> list[tuple[SGNode, Optional[BFTUnicastMulti]]]:
        def render_level(n: BFTNode):
            if n is my_node:
                return 0
            for _i, s in enumerate(my_node.routing_table):
                if n in s.neighbors[RIGHT] or n in s.neighbors[LEFT]:
                    return _i
            return -1
        my_node = self.receiver
        nodes = [k for table in my_node.routing_table for d in [LEFT, RIGHT] for k in table.neighbors[d]]
        nodes.append(my_node)
        k_nodes = closest_k_nodes(self.target, nodes)
        k_nodes = list(filter(lambda n: n is my_node or n not in self.path, k_nodes))
        verbose(f"next hops for target {self.target} are {list_str(k_nodes)}")
        next_nodes = [self.next_msg(n, render_level(n)) for n in k_nodes]
        if self.root is self:
            self.render_level = max(msg.render_level if msg else 0 for _, msg in next_nodes) + 1
        return next_nodes


T = TypeVar('T', bound=SGNode)


def closest_k_nodes(target: int, nodes: list[T]) -> list[T]:
    """
    nodesからtargetに最も近い最大K個のノードをピックアップし，リストで返す．
    結果は [L2, L1, R1, R2] のような形式．
    L1がtargetに最も近い (L1.key <= target < R1.key)．
    nodesが十分大きければLxはLEFT_HALF_K個，RxはRIGHT_HALF_K個ある．
    """
    sorted_nodes = list(set(nodes))
    sort_circular(target, sorted_nodes)
    # print(f"nodes={nodes}")
    # print(f"max_key={max_key}")
    # nodes=[0, 10, 20, 100, 110, 120] のとき:
    # target | return(K=2)              | return(K=4)
    #      5 | [0, 10]                  | [120, 0, 10, 20]
    #     50 | [20, 100]                | [10, 20, 100, 110]
    #    -10 | [120, 0]                 | [110, 120, 0, 10]
    #    150 | [120, 0]                 | [110, 120, 0, 10]
    #      0 | [0, 10] (not [120, 0])   | [120, 0, 10, 20]
    #     10 | [10, 20] (not [0, 10])   | [0, 10, 20, 100]
    #    120 | [120, 0] (not [110, 120])| [110, 120, 0, 10]
    # targetが最大になるように環状キー空間でソートし，配列の右端からLEFT_HALF_K個，左端からRIGHT_HALF_K個を取り出す．
    # target=-10 -> [0, 10, 20, 100, 110, 120] -> [110, 120, 0, 10] (K=2)
    # target=50  -> [100, 110, 120, 0, 10, 20] -> [10, 20, 100, 110] (K=2)
    # target=120 -> [0, 10, 20, 100, 110, 120] -> [110, 120, 0, 10] (K=2)
    # target=10  -> [20, 100, 110, 120, 0, 10] -> [0, 10, 20, 100] (K=2)
    # print(f"sorted nodes={[n.key for n in nodes]}")
    left_len = min(len(sorted_nodes), LEFT_HALF_K)
    right_len = min(len(sorted_nodes) - left_len, RIGHT_HALF_K)
    lefts = sorted_nodes[-left_len:]
    rights = sorted_nodes[:right_len]
    return lefts + rights
