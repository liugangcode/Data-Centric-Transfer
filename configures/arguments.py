import yaml
import argparse

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    parser = argparse.ArgumentParser(description='Data-Centric Learning from Unlabeled Graphs with Diffusion Model')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--no-print', action='store_true', default=False,
                        help="don't use progress bar")

    parser.add_argument('--dataset', default="ogbg-molsider", type=str,
                        choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass', 'plym-thermal',
                                'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol', 
                                'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox','ogbg-molsider','ogbg-moltox21','ogbg-moltoxcast'],
                        help='dataset name (plym-, ogbg-)')
    
    # model
    parser.add_argument('--model', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='graph readout (default: sum)')
    parser.add_argument('--norm-layer', type=str, default='batch_norm', 
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop-ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num-layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb-dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    # training
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--trails', type=int, default=5,
                        help='nubmer of experiments (default: 5)')   
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--initw-name', type=str, default='default',
                        help="method to initialize the model paramter")
    # augmentation
    parser.add_argument('--start', type=int, default=20,
                        help="start epoch for augmentation")
    parser.add_argument('--iteration', type=int, default=20,
                        help='epoch to do augmentation')
    parser.add_argument('--strategy', default="replace_accumulate", type=str,
                        choices=['replace_once', 'add_once', 'replace_accumulate', 'add_accumulate'],
                        help='strategy about how to use the augmented examples. \
                                Replace or add to the original examples; \
                                Accumulate the augmented examples or not')

    parser.add_argument('--n-jobs', type=int, default=22,
                        help='# process to convert the dense adj input to pyg input form')
    parser.add_argument('--n-negative', type=int, default=5,
                        help='# negative samples to optimize the augmented example')
    parser.add_argument('--out-steps', type=int, default=5,
                        help='outer sampling steps for guided reverse diffusion')
    parser.add_argument('--topk', type=int, default=100,
                        help='top k in an augmentation batch ')
    parser.add_argument('--aug-batch', type=int, default=2000,
                        help='the  augmentation batch compared to training batch')
    parser.add_argument('--snr', type=float, default=0.2,
                        help='snr')
    parser.add_argument('--scale-eps', type=float, default=0,
                        help='scale eps')
    parser.add_argument('--perturb-ratio', type=float, default=None,
                        help='level of noise for perturbation')
    args = parser.parse_args()
    print('no print',args.no_print)

    ## n_steps for solver
    args.n_steps = 1
    return args