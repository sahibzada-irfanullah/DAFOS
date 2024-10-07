import argparse
from dafos import main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training DOFAS-GCN on arxiv/reddit Datasets')
    parser.add_argument('--dataset', type=str, default= 'arxiv',
                        help='Dataset name: arxiv/reddit')
    parser.add_argument('--n_epochs', type=int, default= 300,
                        help='Number of Epoch')
    parser.add_argument('--path', type=str, default='/media/irfanserver/Dev/datasets',
                        help='path to input dataset')
    parser.add_argument('--n_stops', type=int, default=200,
                        help='Stop after number of batches that f1 dont increase')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='size of output node in a batch')
    parser.add_argument('--nhid', type=int, default=256,
                        help='Hidden state dimension')
    parser.add_argument('--fanouts', type=list, default=[10, 15],
                        help='Hidden state dimension')
    parser.add_argument('--n_trial', type=int, default=5,
                        help='Number of times to repeat experiments')
    args = parser.parse_args()
    main(args)