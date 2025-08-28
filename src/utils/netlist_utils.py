import os
import numpy as np
import subprocess


def netlist_to_xdata(netlist_file, gate_to_index):
    """
    Parse a gate-level netlist file (Verilog-like) into x_data, edge_index
    x_data: list of [node_id, gate_type_id]
    edge_index: list of [src, dst]
    """
    nodes = []
    edges = []
    node_map = {}
    idx = 0

    with open(netlist_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # 格式: GATETYPE out in1 in2 ...
            toks = line.split()
            gate_type = toks[0].upper()
            out_net = toks[1]
            in_nets = toks[2:]

            if out_net not in node_map:
                node_map[out_net] = idx
                idx += 1
            out_idx = node_map[out_net]
            nodes.append([out_idx, gate_to_index[gate_type]])

            for net in in_nets:
                if net not in node_map:
                    node_map[net] = idx
                    idx += 1
                in_idx = node_map[net]
                edges.append([in_idx, out_idx])

    return nodes, edges
