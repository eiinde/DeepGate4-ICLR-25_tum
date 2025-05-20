import os 
import numpy as np 
from config import get_parse_args
from models.dg4 import DeepGate4, Baseline_Model
from dg_datasets.dg4_parser import LargeNpzParser
from trainer.dg4_trainer import Trainer
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


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

def get_param(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue 

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def save_weights_to_file(model, filepath):
    with open(filepath, 'w') as f:
        for name, param in model.named_parameters():
            if "weight" in name:
                f.write(f"# {name}\n")
                weights = param.detach().cpu().numpy()
                for row in weights:
                    f.write(" ".join(map(str, row)) + "\n")
            elif "bias" in name:
                f.write(f"# {name}\n")
                biases = param.detach().cpu().numpy()
                f.write(" ".join(map(str, biases)) + "\n")

if __name__ == '__main__':
    args = get_parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    

    parser = LargeNpzParser(args.data_dir, args.pkl_path, args, random_shuffle=False)

    train_dataset, val_dataset = parser.get_dataset()

    # Compute the maximum in-degree in the training data.
    max_degree = 2

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    if args.encoder == 'PNA':
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

    # Create Model 
    if args.encoder == 'DeepGate4':
        model = DeepGate4(args)
    else:
        model = Baseline_Model(args, deg=deg)
    print(model)

    if args.dg4_path is not None:
        print(f'load model from {args.dg4_path}')
        model.load(args.dg4_path)
        
    
    # Train 
    trainer = Trainer(
        args=args, 
        model=model, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=args.device, 
        loss=args.loss, 
        num_workers=0
    )

    # Training
    if args.is_test==False:
        # trainer.train(args.epoch, train_dataset, val_dataset)
        trainer.train(args.epoch, train_dataset, train_dataset)
    else:
    # Validation
        trainer.inference(val_dataset)

    
    
    
