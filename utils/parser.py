import argparse
from xmlrpc.client import boolean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'MaFengWo', type = str,
                        help = 'Dataset to use. (MaFengWo or CAMRa2011)')
    parser.add_argument('--embed_size', default = 64, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.01, type = float,
                        help = 'learning rate') 
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'epoch number')
    parser.add_argument('--num_negatives', default = 1, type = int,
                        help = 'number of negative')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--pre', default = 1, type = int,
                        help = '1:saved 0:not save')
    # parser.add_argument('--reg', default = 0.001, type = float,
    #                     help = 'reg')
    parser.add_argument('--layer_num', default = 3, type = int,
                        help = 'number of layers')
    parser.add_argument('--num_aspects', default = 3, type = int,
                        help = 'number of aspects')
    parser.add_argument('--isReg', action='store_false', default=True, help="Aspect regularization")
    parser.add_argument('--reg_coef', default = 0.5, type = float,
                        help = 'Regularization coef')      
    parser.add_argument('--threshold', type=float, default=0.1, help="\epsilon in Eq.13")              
    parser.add_argument('--cuda', default = 0, type = int,
                        help = 'cuda')
    parser.add_argument('--weight_decay', default = 1e-8, type = float,
                        help = 'weight_decay')
    parser.add_argument('--tau_gumbel', type=float, default=0.5, help="temperature in Eq.7")
    parser.add_argument('--isHard', action='store_true', default=False)
    # parser.add_argument('--cold', default = 0, type = int,
    #                     help = '0: normal; 1: cold-start')
    # parser.add_argument('--num_cd', default = 500, type = int,
    #                     help = 'number of cold start users')
    args = parser.parse_args()
    return args