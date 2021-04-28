import sys

import networkx as nx
from matplotlib import axes
from matplotlib import pyplot as plt

import sg
from sg import SGNode, MembershipVector, UnicastBase


FIG_SIZE = (10, 7.5)


def node_with_level(node: SGNode, level: int) -> str:
    return f"{node.key}@{level}"


def ingredients(nodes: list[SGNode], max_level: int) -> \
        tuple[nx.Graph, nx.Graph, dict[str, int], dict[str, tuple[int, int]]]:
    """
    Generate ingredients for rendering a skip graph
    :param nodes
    :param max_level the max level (inclusive) for drawing a skip graph.
    :return tuple of (the base graph, left-side legends, node labels, positions of each object).
    """
    g = nx.Graph()
    g_aux = nx.Graph()
    labels: dict[str, int] = {}
    pos: dict[str, tuple[int, int]] = {}
    y = 0
    x_coords = {
        "level": 0,
        "mv": 2,
        "nodes": 4
    }
    done = {}
    for level in range(0, max_level + 1):
        prefix = 0
        level_string = f"lv {level}"
        g_aux.add_node(level_string)
        pos[level_string] = (x_coords["level"], y)
        if level != 0:
            plt.axhline(y=y-0.5, xmin=0, xmax=1)    # horizontal lines between levels
        for i in range(sg.ALPHA**level):
            mv = MembershipVector(prefix)
            mv.reverse_prefix(level)                # これがないと不自然な並び方になる
            u = None
            exists = False
            edge_drawn = False
            for ind, j in enumerate(range(len(nodes))):
                w = nodes[j]
                if w.key in done:
                    continue
                if w.mv.common_prefix_length(mv) >= level:
                    exists = True
                    w_string = node_with_level(w, level)
                    g.add_node(w_string)
                    pos[w_string] = (x_coords["nodes"] + ind, y)
                    labels[w_string] = w.key
                    if u is not None:
                        u_string = node_with_level(u, level)
                        g.add_edge(u_string, w_string)
                        edge_drawn = True
                    u = w
            if u is not None and not edge_drawn:    # u is singleton
                done[u.key] = True
            prefix += 1
            if exists:
                prefix_string = f"{mv}"[0:level] + "*" * (max_level - level)
                g_aux.add_node(prefix_string)
                pos[prefix_string] = (x_coords["mv"], y)
                y += 1
    # print(f"g={g.edges}")
    # print(f"g_aux={g_aux.nodes}")
    # print(f"labels={labels}")
    # print(f"pos={pos}")
    return g, g_aux, labels, pos


def render_topology_base(ax: axes.Axes, nodes: list[SGNode], max_level: int):
    g, g_aux, labels, pos = ingredients(nodes, max_level)
    nx.draw_networkx(g, pos=pos, node_color="c", labels=labels, ax=ax)
    nx.draw_networkx(g_aux, pos=pos, node_shape="", ax=ax)
    return labels, pos


def output_topology(nodes: list[SGNode], max_level: int, filename: str) -> None:
    _, ax = plt.subplots(figsize=FIG_SIZE)
    render_topology_base(ax, nodes, max_level)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if filename is None:
        print("Showing the topology.  Close the window to proceed.", file=sys.stderr)
        plt.show()
    else:
        plt.savefig(filename)
    plt.close('all')


def render_hop_graph(root_msg: UnicastBase, nodes: list[SGNode], filename: str, diagonal=False) -> None:
    def recurse(parent: UnicastBase):
        nonlocal highest_level, edge_labels
        highest_level = max(highest_level, parent.render_level)
        for msg in parent.children:
            if diagonal:
                # draw diagonal lines
                u = node_with_level(parent.receiver, parent.render_level)
                v = node_with_level(msg.receiver, msg.render_level)
                unicast_graph.add_edge(u, v)
                edge_labels[(u, v)] = msg.hop
            else:
                # draw vertical and horizontal lines
                u = node_with_level(parent.receiver, parent.render_level)
                v = node_with_level(parent.receiver, msg.render_level)
                w = node_with_level(msg.receiver, msg.render_level)
                unicast_graph.add_edge(u, v)
                unicast_graph.add_edge(v, w)
                edge_labels[(v, w)] = msg.hop
            recurse(msg)
    unicast_graph = nx.DiGraph()
    highest_level = 0
    edge_labels = {}
    recurse(root_msg)
    print(f"unicast edges={list(unicast_graph.edges())}")

    _, ax = plt.subplots(figsize=FIG_SIZE)
    labels, pos = render_topology_base(ax, nodes, highest_level)
    # draw unicast hop lines
    nx.draw_networkx(unicast_graph, pos=pos, labels=labels, edge_color="orange", ax=ax, width=2)
    # attach edge labels to hop lines
    # bbox is for adding alpha in edge labels
    bbox = {
        "boxstyle": "round",
        "ec": (1.0, 1.0, 1.0),
        "fc": (1.0, 1.0, 1.0),
        "alpha": 0.3
    }
    texts = nx.draw_networkx_edge_labels(unicast_graph, pos=pos, edge_labels=edge_labels, bbox=bbox, ax=ax)
    # make labels foreground
    for t in texts.values():
        t.zorder = 3
    # add "src->target"
    legend_y = max(y for (x, y) in pos.values())
    plt.text(0, legend_y, f"{root_msg.receiver.key}->{root_msg.target}", color="magenta")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if filename is None:
        print("Showing the topology.  Close the window to proceed.", file=sys.stderr)
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
