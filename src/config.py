import argparse
import os 
import torch

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gpus', default='-1', type=str)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--enable_cut', action='store_true', default=False)
    parser.add_argument('--is_test', action='store_true', default=False)
    
    # Dataset
    parser.add_argument('--data_dir', default='./inmemory/toy_aig')
    parser.add_argument('--dg4_path', default=None)
    parser.add_argument('--npz_dir', default=None)
    parser.add_argument('--pkl_path', default='./i99_large/graphs.pkl')
    parser.add_argument('--circuit_path', default='./data/dg3_80k/wl_4_hop.npz')
    parser.add_argument('--test_npz_path', default='./dg3_dataset/test/00.npz')
    parser.add_argument('--default_dataset', action='store_true')
    parser.add_argument('--hop_ratio', default=0.15, type=float)
    parser.add_argument('--k_hop', default=4, type=int)
    parser.add_argument('--max_hop_pi', default=6, type=int)
    parser.add_argument('--load_npz', default='', type=str)
    
    # Model 

    parser.add_argument('--cfg_file', default=None, type=str)
    parser.add_argument('--encoder', default='DeepGate4', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--pretrained_model_path', default='./DeepGate3-Transformer/trained/model_last.pth')
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--gnn_type', default='gcn', type=str)
    parser.add_argument('--workload', action='store_true', default=False)
    parser.add_argument('--skip_path', action='store_true', default=False)
    parser.add_argument('--skip_hop', action='store_true', default=False)
    
    # Transformer 
    parser.add_argument('--tf_arch', default='plain', type=str)
    parser.add_argument('--TF_depth', default=12, type=int)
    parser.add_argument('--token_emb', default=128, type=int)
    parser.add_argument('--tf_emb_size', default=128, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--MLP_expansion', default=4, type=int)
    
    # Mask Prediction 
    parser.add_argument('--mlp_hidden', default=128, type=int)
    parser.add_argument('--mlp_layer', default=3, type=int)
    parser.add_argument('--norm_layer', default='batchnorm', type=str)
    parser.add_argument('--act_layer', default='relu', type=str)
    
    # Train
    parser.add_argument('--en_distrubuted', default=False,action='store_true')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--mini_batch_size', default=256, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--loss', default='l2', type=str)
    parser.add_argument('--fast', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # device
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    print('Using GPU:', args.gpus)
    # args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    if len(args.gpus) > 1 and torch.cuda.is_available():
        args.en_distrubuted = True
    else:
        args.en_distrubuted = False
    args.device = torch.device(f'cuda:{args.gpus[0]}' if args.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu')

    return args