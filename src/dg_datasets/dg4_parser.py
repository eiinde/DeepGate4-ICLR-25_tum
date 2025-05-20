import deepgate as dg 
import torch
import os
import time
import pickle
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F

from typing import  List

NODE_CONNECT_SAMPLE_RATIO = 0.1
NO_NODE_PATH = 10
NO_NODE_HOP = 10
        
def build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob):
    area_g = AreaData()
    nodes = area_nodes[area_nodes != -1]
    pi_mask = (area_nodes_stats == 1)[:len(nodes)]
    area_g.nodes = nodes
    area_g.gate = g.gate[nodes]
    area_g.gate[pi_mask] = 0
    area_g.prob = prob[nodes]
    area_g.forward_level = g.forward_level[nodes]
    area_g.backward_level = g.backward_level[nodes]
    area_g.forward_index = torch.tensor(range(len(nodes)))
    area_g.backward_index = torch.tensor(range(len(nodes)))
    
    # Edge_index
    glo_to_area = {}
    for i, node in enumerate(nodes):
        glo_to_area[node.item()] = i
    area_edge_index = []
    for edge in g.edge_index.t():
        if edge[0].item() in glo_to_area and edge[1].item() in glo_to_area:
            area_edge_index.append([glo_to_area[edge[0].item()], glo_to_area[edge[1].item()]])
    area_edge_index = torch.tensor(area_edge_index).t()
    area_g.edge_index = area_edge_index
    
    area_g.fanin_fanout_cones = area_faninout_cone
    area_g.batch = torch.zeros(len(nodes), dtype=torch.long)
    return area_g

def merge_area_g(batch_g, g):
    no_nodes = batch_g.nodes.shape[0]
    batch_g.nodes = torch.cat([batch_g.nodes, g.nodes])
    batch_g.gate = torch.cat([batch_g.gate, g.gate])
    batch_g.prob = torch.cat([batch_g.prob, g.prob])
    batch_g.forward_level = torch.cat([batch_g.forward_level, g.forward_level])
    batch_g.backward_level = torch.cat([batch_g.backward_level, g.backward_level])
    batch_g.edge_index = torch.cat([batch_g.edge_index, g.edge_index + no_nodes], dim=1)
    batch_g.fanin_fanout_cones = torch.cat([batch_g.fanin_fanout_cones, g.fanin_fanout_cones], dim=0)
    batch_g.batch = torch.cat([batch_g.batch, torch.tensor([batch_g.batch.max() + 1] * len(g.nodes)).to(batch_g.batch.device)])
    
    batch_g.forward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    batch_g.backward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    
    return batch_g
  

def get_areas(g):
    area_g_list = []
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
        
        area_g_list.append(area_g)

    return area_g_list

def get_areas_with_level(g):

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


class AreaData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ninp_node_index' or key == 'ninh_node_index':
            return self.num_nodes
        elif key=="global_virtual_edge" or key == 'local_virtual_edge':
            return 0
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
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index' or key=="global_virtual_edge" or key == 'local_virtual_edge' or key == 'hs_init':
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


class LargeNpzParser():
    def __init__(self, data_dir, pkl_dir, args, random_shuffle=True, trainval_split=0.9, val_name=None):
        self.data_dir = data_dir
        self.train_dataset = []
        self.pkl_dir = pkl_dir
        
        dataset = self.inmemory_dataset(data_dir, self.pkl_dir, args, debug=args.debug)
        data_len = len(dataset)

        # test dataset split
        if val_name is None:
            if 'ITC99_hybrid' in data_dir:
                val_name = ['b12_opt_C','b14_opt_C']
            elif 'ITC' in data_dir:
                val_name = ['b12_opt_C','b14_opt_C']
            elif 'EPFL_random_control' in data_dir:
                val_name = ['arbiter', 'cavlc']
            elif 'EPFL_pkl' in data_dir:
                val_name = ['cavlc', 'arbiter','mem_ctrl']
            else:
                val_name = []
        val_idx = []
        train_idx = []
        for i,n in enumerate(dataset.aig_namelist):
            if n in val_name:
                val_idx.append(i)
            else:
                train_idx.append(i)

        self.train_dataset = dataset[train_idx]
        self.val_dataset = dataset[val_idx]
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, pkl_dir, args, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.args = args
            self.pkl_dir = pkl_dir
            self.debug = debug
            
            aig_namelist_path = os.path.join(pkl_dir, "aig_namelist.txt")
            if not os.path.exists(aig_namelist_path): 
                # then create the aig_namelist file
                aig_namelist = []
                for i in os.listdir(pkl_dir):
                    if i.endswith('.pkl'):
                        aig_namelist.append(i[:-4])
            else:   
                with open(aig_namelist_path, 'r') as f:
                    aig_namelist = f.readlines()
                    aig_namelist = [x.strip() for x in aig_namelist]
            
            self.aig_namelist = aig_namelist

            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'immemory'
            inmemory_path = os.path.join(self.root, name)
            if os.path.exists(inmemory_path):
                print('Inmemory Dataset Path: {}, Existed'.format(inmemory_path))
            else:
                print('Inmemory Dataset Path: {}, New Created'.format(inmemory_path))
            
            return inmemory_path

        @property
        def raw_file_names(self) -> List[str]:
            return [os.path.join(self.pkl_dir,i+'.pkl') for i in self.aig_namelist]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process(self):
            data_list = []
            tot_pairs = 0
            print('Parse NPZ Datset ...')
            tot_time = 0

            for cir_idx, pkl_path in enumerate(self.raw_file_names):

                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    circuits = data

                cir_name = self.aig_namelist[cir_idx]

                start_time = time.time()
                print('Parse: {}, {:} / {:} = {:.2f}%, Time: {:.2f}s, ETA: {:.2f}s, Curr Size: {:}'.format(
                    cir_name, cir_idx, len(self.raw_file_names), cir_idx / len(self.raw_file_names) * 100, 
                    tot_time, tot_time * (len(self.raw_file_names) - cir_idx), 
                    len(data_list)
                ))
                
                succ = True

                for key in circuits.keys():
                    if key == 'connect_pair_index' and len(circuits[key]) == 0:
                        succ = False
                        print(f'{cir_name} dont have connect pair, fail')
                        break
                    if key == 'connect_pair_index':
                        circuits[key] = circuits[key].T
                        
                    if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key:
                        circuits[key] = torch.tensor(circuits[key], dtype=torch.float)
                    elif key=='hs_init':
                         circuits[key] = torch.tensor(circuits[key], dtype=torch.float)
                    elif key != 'name' and key !='area_list':
                        circuits[key] = torch.tensor(circuits[key], dtype=torch.long)

                
                if 'merge_small' not in cir_name:
                    circuits['nodes'] = circuits['forward_index']
                    circuits['po'] = torch.ones(0)
                    circuits['batch'] = torch.ones(0)
                    circuits['mini_batch'] = torch.ones(0)
                    circuits['global_virtual_edge'] = torch.ones(2,0)
                if circuits['connect_pair_index'].shape[1] == 2:
                    circuits['connect_pair_index'] = circuits['connect_pair_index'].T
                if 'merge_small' in cir_name:
                    circuits.area_list = []
                    circuits.mini_batch = circuits.batch

                if not succ:
                    continue
                circuits.name = cir_name

                data_list.append(circuits)
                tot_time = time.time() - start_time
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
            

