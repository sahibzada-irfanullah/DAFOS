import sklearn.metrics
import tqdm
import numpy as np
from dgl.data import RedditDataset
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

def load_reddit(root):
    data = RedditDataset(raw_dir=root, self_loop=True)
    g = data[0]
    return g



def load_data(args):
    if args.dataset == "reddit":
        path = args.path + "/" + args.dataset
        graph = load_reddit(path)
        node_labels = graph.ndata['label']
        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()

    elif args.dataset == "arxiv":
        dataset = DglNodePropPredDataset('ogbn-arxiv', root=args.path)
        graph, node_labels = dataset[0]
        graph = dgl.add_reverse_edges(graph)
        graph.ndata['label'] = node_labels[:, 0]
        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']
    else:
        print(10*"WRONG DATA SET SELECTED: Select from arxiv/reddit")
    return graph, node_labels, train_nids, valid_nids, test_nids

def node_scoring(graph):
    in_degrees = graph.in_degrees()
    scores, indices = torch.sort(in_degrees, descending=True)
    return indices  # Return nodes sorted by degree (high to low)

def evaluate(dataloader, model, args, loss_fn, best_val, cnt, stop = False, valid = True):
    labels = []
    predictions = []
    with tqdm.tqdm(dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        labels = torch.from_numpy(labels).float()
        predictions = torch.from_numpy(predictions).float()
        if valid:
            loss_valid = loss_fn(predictions, labels)
            valid_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
            if valid_f1 > best_val + 1e-2:
                best_val = valid_f1
                cnt = 0
            else:
                cnt += 1
            if cnt == args.n_stops:
                stop = True
            print(f'Valid f1: {valid_f1} Best Valid f1: {best_val}')
            return stop, cnt, best_val
        else:
            test_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
            if test_f1 > best_val:
                best_val = test_f1
            print(f'Test f1: {test_f1}, Best Test f1: {best_val}')
            return None, None, best_val
