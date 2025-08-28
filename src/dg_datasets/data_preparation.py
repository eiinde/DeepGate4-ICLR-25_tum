import os 
import glob
import deepgate as dg
from torch_geometric.data import Data
import torch
import pickle
import numpy as np 
import random
import copy
import time
import argparse
import torch.nn.functional as F
import sys
sys.path.append('./src')
# from utils import aiger_utils
from utils import circuit_utils
from utils import netlist_utils

from collections import defaultdict

gate_to_index = {
    "PI": 0, "AND": 1, "OR": 2, "NAND": 3, "NOR": 4,
    "XOR": 5, "XNOR": 6, "INV": 7, "BUF": 8, "DFF": 9
}
NODE_CONNECT_SAMPLE_RATIO = 0.1
NO_NODE_PATH = 10
NO_NODE_HOP = 10
K_HOP = 4

NO_NODES = [30, 500000]

import sys
sys.setrecursionlimit(1000000)

def get_parse_args():
    parser = argparse.ArgumentParser()

    # Range
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100000, type=int)
    
    # Input
    parser.add_argument('--netlist_dir', default='./raw_sample_data', type=str)
    
    # Output
    parser.add_argument('--save_path', default='./sample_data', type=str)
    
    args = parser.parse_args()
    
    return args

class AreaData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ninp_node_index' or key == 'ninh_node_index':
            return self.num_nodes
        elif key=="global_virtual_edge" or key == 'local_virtual_edge':
            return self.num_nodes
        elif key == 'ninp_path_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'ninh_hop_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_nodes': 
            return self.num_nodes
        elif key == 'area_po' or key == 'area_nodes':
            return self.num_nodes
        elif key == 'hop_pair_index' or key == 'hop_forward_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'path_forward_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'paths' or key == 'hop_nodes':
            return self.num_nodes
        elif 'index' in key or 'face' in key:
            return self.num_nodes

        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index' or key=="global_virtual_edge" or key == 'local_virtual_edge':
            return 1
        elif key == "connect_pair_index" or key == 'hop_pair_index':
            return 1
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_pi_stats' or key == 'hop_tt' or key == 'no_hops':
            return 0
        elif key == 'area_po' or key == 'area_nodes' or key == 'area_nodes_stats' or key == 'area_lev':
            return 0
        elif key == 'hop_nodes' or key == 'hop_nodes_stats':
            return 0
        elif key == 'paths':
            return 0
        else:
            return 0

def get_area(g, k_hop=8):
    graph = {}
    max_level = g.forward_level.max()
    forward_level = g.forward_level.numpy()
    backward_level = g.backward_level.numpy()
    edge_index = g.edge_index
    gate = g.gate

    level_list = [[] for _ in range(max_level+1)]
    for idx in range(len(x_data)):
        level_list[forward_level[idx]].append(idx)
    po_list = g.forward_index[backward_level == 0]
    
    all_hop_po = torch.zeros((0, 1), dtype=torch.long)
    all_hop_winlev = torch.zeros((0, 1), dtype=torch.long)
    max_hop_nodes_cnt = 1
    for k in range(k_hop+1):
        max_hop_nodes_cnt += 2**k
    all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
    all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
    
    
    # edge_index = torch.tensor(edge_index, dtype=torch.long)
    # gate = torch.tensor(g['gate'], dtype=torch.long)
    has_hop = [0] * len(x_data)
    hop_level = k_hop
    hop_winlev = 0
    while hop_level < max_level:
        for idx in level_list[hop_level]:
            hop_nodes, hop_gates, hop_pis, hop_pos = circuit_utils.get_hops(idx, edge_index, x_data, gate, k_hop=k_hop)
            if len(hop_gates) < 2:
                continue
            has_hop[idx] = 1
            
            # Record hop
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_hop_winlev = torch.cat([all_hop_winlev, torch.tensor([hop_winlev]).view(1, -1)], dim=0)
            
        hop_level += (k_hop - 2)
        hop_winlev += 1
    
    # Add PO 
    for po_idx in po_list:
        if has_hop[po_idx] == 0:
            hop_nodes, hop_gates, hop_pis, hop_pos = circuit_utils.get_hops(po_idx, edge_index, x_data, gate, k_hop=k_hop)
            if len(hop_gates) < 2:
                continue
            has_hop[po_idx] = 1
            
            # Record hop
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_hop_winlev = torch.cat([all_hop_winlev, torch.tensor([hop_winlev]).view(1, -1)], dim=0)
    graph = {
        'area_po': all_hop_po.numpy(),
        'area_nodes': all_hop_nodes.numpy(),
        'area_nodes_stats': all_hop_nodes_stats.numpy(), 
        'area_lev': all_hop_winlev.numpy()
    }
    g.update(graph)
    return g

def get_fanin_fanout_cone(forward_index,forward_level,backward_level,edge_index, max_no_nodes=512): 
    # Parse graph 
    no_nodes = len(forward_index)
    forward_level_list = [[] for I in range(forward_level.max()+1)]
    backward_level_list = [[] for I in range(backward_level.max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in edge_index.t():
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    for k, idx in enumerate(forward_index):
        forward_level_list[forward_level[k].item()].append(k)
        backward_level_list[backward_level[k].item()].append(k)
    
    # PI Cover 
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(forward_level_list)):
        for idx in forward_level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    
    # PO Cover
    po_cover = [[] for _ in range(no_nodes)]
    for level in range(len(backward_level_list)):
        for idx in backward_level_list[level]:
            if level == 0:
                po_cover[idx].append(idx)
            tmp_po_cover = []
            for post_k in fanout_list[idx]:
                tmp_po_cover += po_cover[post_k]
            tmp_po_cover = list(set(tmp_po_cover))
            po_cover[idx] += tmp_po_cover
    
    # fanin and fanout cone 
    fanin_fanout_cones = [[-1]*max_no_nodes for _ in range(max_no_nodes)]
    fanin_fanout_cones = torch.tensor(fanin_fanout_cones, dtype=torch.long)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i == j:
                fanin_fanout_cones[i][j] = 0
                continue
            if len(pi_cover[j]) <= len(pi_cover[i]) and forward_level[j] < forward_level[i]: # 这里len(pi_cover[j]) <= len(pi_cover[i])是什么意思
                j_in_i_fanin = True
                for pi in pi_cover[j]:
                    if pi not in pi_cover[i]:
                        j_in_i_fanin = False
                        break
                if j_in_i_fanin:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 1
                else:
                    fanin_fanout_cones[i][j] = 0
            elif len(po_cover[j]) <= len(po_cover[i]) and forward_level[j] > forward_level[i]:
                j_in_i_fanout = True
                for po in po_cover[j]:
                    if po not in po_cover[i]:
                        j_in_i_fanout = False
                        break
                if j_in_i_fanout:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 2
                else:
                    fanin_fanout_cones[i][j] = 0
            else:
                fanin_fanout_cones[i][j] = 0
    
    assert -1 not in fanin_fanout_cones[:no_nodes, :no_nodes]
    
    return fanin_fanout_cones

def build_graph(area_forward_index, area_forward_level, area_backward_level, edge_index, global_virtual_edge, local_virtual_edge, prob, area_nodes_stats, gate):
    area_g = AreaData()
    nodes = area_forward_index
    
    area_g.nodes = nodes
    area_g.gate = gate[nodes]
    # pi_mask = (area_nodes_stats == 1)[:len(nodes)]
    # area_g.gate[pi_mask] = 0
    area_g.prob = prob[nodes]
    area_g.forward_level = area_forward_level
    area_g.backward_level = area_backward_level
    area_g.forward_index = torch.tensor(range(len(nodes)))
    area_g.backward_index = torch.tensor(range(len(nodes)))
    
    # Edge_index
    glo_to_area = {}
    for i, node in enumerate(nodes):
        glo_to_area[node.item()] = i
    area_edge_index = []
    for edge in edge_index.t():
        if edge[0].item() in glo_to_area and edge[1].item() in glo_to_area:
            area_edge_index.append([glo_to_area[edge[0].item()], glo_to_area[edge[1].item()]])
    area_edge_index = torch.tensor(area_edge_index).t()
    area_g.edge_index = area_edge_index
    
    area_g.global_virtual_edge = global_virtual_edge
    area_g.local_virtual_edge = local_virtual_edge
    area_g.batch = torch.zeros(len(nodes), dtype=torch.long)
    return area_g

def build_areas_with_level(g):

    max_lv = g.area_lev.max()+1
    area_g_list = [[] for i in range(max_lv)]
    curr_bs = 0
    all_area_nodes = g.area_nodes
    all_area_nodes_stats = g.area_nodes_stats
    all_area_lev = g.area_lev
    all_area_faninout_cone = g.area_fanin_fanout_cone
    prob = g.prob.clone()
    for area_idx, area_nodes in enumerate(all_area_nodes):

        area_nodes_stats = all_area_nodes_stats[area_idx]
        area_faninout_cone = all_area_faninout_cone[area_idx]
        area_g = build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob)
        
        area_g_list[all_area_lev[area_idx]].append(area_g)

    return area_g_list

def find_predecessors_at_depth(edge_index, node_index, depth):

    predecessors_at_depth = {node.item(): set() for node in node_index}

    node_index_set = set(node_index.tolist())

    def dfs(node, current_depth):
        if current_depth == 0:
            return set()
        
        preds = set()
        for i in range(edge_index.size(1)):
            if edge_index[1, i] == node:
                pred_node = edge_index[0, i].item()
                preds.add(pred_node)
                if current_depth > 1:
                    preds.update(dfs(pred_node, current_depth - 1))
        return preds

    for node in node_index:
        predecessors_at_depth[node.item()] = list(dfs(node.item(), depth) & node_index_set)
    return predecessors_at_depth

def find_all_paths_to_sources(edge_index, node_index):

    adj_list = defaultdict(list)
    for i in range(edge_index.size(1)):
        adj_list[edge_index[1, i].item()].append(edge_index[0, i].item())
    
    paths_to_sources = {}

    def dfs(node):
        if node in paths_to_sources:
            return paths_to_sources[node]
        
        if node not in adj_list:
            return [[node]]

        all_paths = []
        for pred in adj_list[node]:
            pred_paths = dfs(pred)
            for path in pred_paths:
                all_paths.append([node] + path)
        
        paths_to_sources[node] = all_paths
        return all_paths

    for node in node_index:
        dfs(node.item())

    return paths_to_sources


def generate_orthogonal_vectors(n, dim):
    if n < dim * 8:
        # Choice 1: Generate n random orthogonal vectors in R^dim
        # Generate an initial random vector
        v0 = np.random.randn(dim)
        v0 /= np.linalg.norm(v0)
        # Generate n-1 additional vectors
        vectors = [v0]
        for i in range(n-1):
            while True:
                # Generate a random vector
                v = np.random.randn(dim)

                # Project the vector onto the subspace spanned by the previous vectors
                for j in range(i+1):
                    v -= np.dot(v, vectors[j]) * vectors[j]

                if np.linalg.norm(v) > 0:
                    # Normalize the vector
                    v /= np.linalg.norm(v)
                    break

            # Append the vector to the list
            vectors.append(v)
    else: 
        # Choice 2: Generate n random vectors:
        vectors = np.random.rand(n, dim) - 0.5
        for i in range(n):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    return vectors


if __name__ == '__main__':     
    args = get_parse_args()
    
    netlist_namelist_path = os.path.join(args.netlist_dir, 'netlist_namelist.txt')
    if not os.path.exists(netlist_namelist_path):
        netlist_files = glob.glob('{}/*.v'.format(args.netlist_dir))
        netlist_namelist = []
        for netlist_file in netlist_files:
            netlist_name = os.path.basename(netlist_file).replace('.v', '')
            netlist_namelist.append(netlist_name)
        with open(netlist_namelist_path, 'w') as f:
            for netlist_name in netlist_namelist:
                f.write(netlist_name + '\n')
    else:
        with open(netlist_namelist_path, 'r') as f:
            netlist_namelist = f.readlines()
            netlist_namelist = [x.strip() for x in netlist_namelist]
    
    netlist_namelist = netlist_namelist[args.start: args.end]
    no_circuits = len(netlist_namelist)
    tot_time = 0
    graphs = {}

    for netlist_idx, cir_name in enumerate(netlist_namelist):
        netlist_file = os.path.join(args.netlist_dir, cir_name + '.netlist')
        start_time = time.time()

        x_data, edge_index = netlist_utils.netlist_to_xdata(netlist_file,gate_to_index)
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            cir_name, netlist_idx, no_circuits, len(x_data),
            tot_time, tot_time / ((netlist_idx + 1) / no_circuits) - tot_time,
            len(graphs)
        ))
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        
        # Replace DFF as PPI and PPO
        no_ff = 0
        for idx in range(len(x_data)):
            if x_data[idx][1] == gate_to_index['DFF']:
                no_ff += 1
                x_data[idx][1] = gate_to_index['PI']
                for fanin_idx in fanin_list[idx]:
                    fanout_list[fanin_idx].remove(idx)
                fanin_list[idx] = []
        
        # Get x_data and edge_index
        edge_index = []
        for idx in range(len(x_data)):
            for fanin_idx in fanin_list[idx]:
                edge_index.append([fanin_idx, idx])
        x_data, edge_index = circuit_utils.remove_unconnected(x_data, edge_index)
        if len(edge_index) == 0 or len(x_data) < NO_NODES[0] or len(x_data) > NO_NODES[1]:
            continue
        x_one_hot = dg.construct_node_feature(x_data, 9)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
        
        print(f'node: {x_data.shape},max level: {forward_level.max()}')
        # break

        graph = AreaData()
        graph.x = x_one_hot
        graph.edge_index = edge_index
        graph.name = cir_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
                
        ################################################
        # DeepGate2 (node-level) labels
        ################################################
        t0 = time.time()
        prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.prepare_dg2_labels_cpp(graph, 15000, fast=False,simulator='./simulator/simulator')
        t1 = time.time()
        print(f'simulation time for {cir_name}: {t1-t0:.2f}s')
        assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
        if len(tt_pair_index) == 0:
            tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            tt_pair_index = tt_pair_index.t().contiguous()
        graph.prob = prob
        graph.tt_pair_index = tt_pair_index
        graph.tt_sim = tt_sim

        out_and = []
        out_not = []
        for i in range(graph.gate.shape[0]):

            out_gate = graph.edge_index[1,graph.edge_index[0,:]==i]
            out_gate = graph.gate[out_gate]
            out_and.append((out_gate==1).sum())
            out_not.append((out_gate==2).sum())
        
        graph.out_and = torch.tensor(out_and)
        graph.out_not = torch.tensor(out_not)

        num1 = (con_label==1).sum()
        num2 = (con_label==2).sum()
        num0 = (num1+num2) // 2 

        selected_indices = torch.randperm((con_label==0).sum())[:num0]

        graph.connect_pair_index = torch.cat([con_index[con_label==0][selected_indices],con_index[con_label==1],con_index[con_label==2]],dim=0)
        graph.connect_label = torch.cat([con_label[con_label==0][selected_indices],con_label[con_label==1],con_label[con_label==2]],dim=0)

        # prepare hs init for DG2
        t0 = time.time()
        hs_init = generate_orthogonal_vectors((graph.forward_level==0).sum(),128)
        graph.hs_init = torch.tensor(np.array(hs_init))
        t1 = time.time()
        print(f'time for generating orthogonal vectors:{t1-t0:.2f}s')

        # Statistics
        graph.no_nodes = len(x_data)
        graph.no_edges = len(edge_index[0])


        ################################################
        # Graph partition
        ################################################
        t0 = time.time()
        graph = get_area(graph, k_hop=8)
        t1 = time.time()
        print(f'partition time for {cir_name}: {t1-t0:.2f}s')
        n_area = graph.area_nodes.shape[0]
        area_list = [[] for _ in range(graph.area_lev.max()+1)]
        for i in range(n_area):
            if i%50 == 0:
                t = time.time()
                print(f'build virtual edge for {i}/{n_area} area, tot time:{t-start_time:.2f}s')
            # if i > 5: 
            #     break
            area_forward_index = graph.area_nodes[i][graph.area_nodes[i]!=-1]
            area_forward_level = graph.forward_level[area_forward_index]
            area_backward_level = graph.backward_level[area_forward_index]
            area_edge_index = graph.edge_index

            area = build_graph(area_forward_index, area_forward_level, area_backward_level, area_edge_index, None, None, graph.prob,graph.area_nodes_stats[i], graph.gate)
            area.po = torch.tensor(graph.area_po[i])

            ff_cone = get_fanin_fanout_cone(area.forward_index, area.forward_level, area.backward_level, area.edge_index, max_no_nodes=area.forward_index.shape[0])
            ff_cone = ff_cone + torch.eye(area.forward_index.shape[0]).int()
            global_virtual_edge = torch.argwhere(ff_cone.T==1)
            area.global_virtual_edge = global_virtual_edge.T


            area_list[graph.area_lev[i][0]].append(area)

        graph.area_list = area_list

        ################################################
        # Hop-level labels    
        ################################################  

        t0 = time.time()

        rand_idx_list = graph.area_po.squeeze(-1)

        all_hop_pi = torch.zeros((0, 2**(K_HOP-1)), dtype=torch.long)
        all_hop_pi_stats = torch.zeros((0, 2**(K_HOP-1)), dtype=torch.long)
        all_hop_po = torch.zeros((0, 1), dtype=torch.long)
        max_hop_nodes_cnt = 0
        for k in range(K_HOP+1):
            max_hop_nodes_cnt += 2**k
        all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
        all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
        all_tt = []
        all_hop_nodes_cnt = []
        all_hop_level_cnt = []
        for idx in rand_idx_list:
            last_target_idx = copy.deepcopy([idx])
            curr_target_idx = []
            hop_nodes = []
            hop_edges = torch.zeros((2, 0), dtype=torch.long)
            for k_hops in range(K_HOP):
                if len(last_target_idx) == 0:
                    break
                for n in last_target_idx:
                    ne_mask = edge_index[1] == n
                    curr_target_idx += edge_index[0, ne_mask].tolist()
                    hop_edges = torch.cat([hop_edges, edge_index[:, ne_mask]], dim=-1)
                    hop_nodes += edge_index[0, ne_mask].unique().tolist()
                last_target_idx = list(set(curr_target_idx))
                curr_target_idx = []

            if len(hop_nodes) < 2:
                continue
            hop_nodes = torch.tensor(hop_nodes).unique().long()
            hop_nodes = torch.cat([hop_nodes, torch.tensor([idx])])
            no_hops = k_hops + 1
            hop_forward_level, hop_forward_index, hop_backward_level, _ = dg.return_order_info(hop_edges, len(x_data))
            hop_forward_level = hop_forward_level[hop_nodes]
            hop_backward_level = hop_backward_level[hop_nodes]
            
            hop_gates = graph.gate[hop_nodes]
            hop_pis = hop_nodes[(hop_forward_level==0) & (hop_backward_level!=0)]
            hop_pos = hop_nodes[(hop_forward_level!=0) & (hop_backward_level==0)]
            if len(hop_pis) > 2**(K_HOP-1):
                continue
            
            hop_pi_stats = [2] * len(hop_pis)  # -1 Padding, 0 Logic-0, 1 Logic-1, 2 variable
            for assigned_pi_k in range(6, len(hop_pi_stats), 1):
                hop_pi_stats[assigned_pi_k] = random.randint(0, 1)
            hop_tt, _ = circuit_utils.complete_simulation(hop_pis, hop_pos, hop_forward_level, hop_nodes, hop_edges, hop_gates, pi_stats=hop_pi_stats)
            while len(hop_tt) < 2**6:
                hop_tt += hop_tt
                hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                hop_pi_stats.insert(0, -1)
            while len(hop_pi_stats) < 2**(K_HOP-1):
                hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                hop_pi_stats.insert(0, -1)
            
            # Record the hop 
            all_hop_pi = torch.cat([all_hop_pi, hop_pis.view(1, -1)], dim=0)
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            all_hop_pi_stats = torch.cat([all_hop_pi_stats, torch.tensor(hop_pi_stats).view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            all_hop_nodes_cnt.append(len(hop_nodes))
            all_hop_level_cnt.append(no_hops)
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_tt.append(hop_tt)

        graph.hop_pi = all_hop_pi
        graph.hop_po = all_hop_po
        graph.hop_pi_stats = all_hop_pi_stats
        graph.hop_nodes = all_hop_nodes
        graph.hop_nodes_stats = all_hop_nodes_stats
        graph.hop_tt = torch.tensor(all_tt, dtype=torch.long)
        graph.hop_nds = torch.tensor(all_hop_nodes_cnt, dtype=torch.long)
        graph.hop_levs = torch.tensor(all_hop_level_cnt, dtype=torch.long)
        graph.hop_forward_index = torch.tensor(range(len(all_hop_nodes)), dtype=torch.long)

        hop_pair_index, hop_pair_ged, hop_pair_tt_sim = circuit_utils.get_hop_pair_labels(
            all_hop_nodes, graph.hop_tt, edge_index, 
            no_pairs=min(int(len(all_hop_nodes) * len(all_hop_nodes) * 0.1), 2000)
        )

        no_pairs = len(hop_pair_index)
        if no_pairs == 0:
            continue
        graph.hop_pair_index = hop_pair_index.T.reshape(2, no_pairs)
        graph.hop_ged = hop_pair_ged
        graph.hop_tt_sim = torch.tensor(hop_pair_tt_sim, dtype=torch.float)
        
        # Sample node in hop 
        node_hop_pair_index = []
        node_hop_labels = []
        for hop_idx, sample_hop in enumerate(all_hop_nodes):
            hop = sample_hop[sample_hop != -1].tolist()
            node_in_hop = np.random.choice(hop, NO_NODE_HOP)
            node_in_hop = [[x, hop_idx] for x in node_in_hop]
            node_out_hop = [x for x in range(len(x_data)) if x not in hop]
            node_out_hop = np.random.choice(node_out_hop, NO_NODE_HOP)
            node_out_hop = [[x, hop_idx] for x in node_out_hop]
            node_hop_pair_index += node_in_hop + node_out_hop
            node_hop_labels += [1] * NO_NODE_HOP + [0] * NO_NODE_HOP
        node_hop_pair_index = torch.tensor(node_hop_pair_index, dtype=torch.long)
        node_hop_labels = torch.tensor(node_hop_labels, dtype=torch.long)
        ninh_node_index = node_hop_pair_index[:, 0]
        ninh_hop_index = node_hop_pair_index[:, 1]
        graph.ninh_node_index = ninh_node_index
        graph.ninh_hop_index = ninh_hop_index
        graph.ninh_labels = node_hop_labels
        graph.no_hops = len(all_hop_nodes)

        t1 = time.time()
        print(f'prepare hop label for {cir_name}: {t1-t0:.2f}s')
        

        end_time = time.time()
        tot_time += end_time - start_time

        with open(f'{args.save_path}/{cir_name}.pkl', 'wb') as f:
            pickle.dump(graph, f)

        print(f'{cir_name} finished, save to {args.save_path}/{cir_name}.pkl')
        print(graph.keys())
