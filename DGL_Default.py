import os
import sys
import torch.nn.functional as F
from model import GCN
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Training DGL-GCN on arxiv/reddit Datasets')
parser.add_argument('--dataset', type=str, default= 'arxiv',
                    help='Dataset name: arxiv/reddit')
parser.add_argument('--n_epochs', type=int, default= 300,
                    help='Number of Epoch')
parser.add_argument('--path', type=str, default= '/media/irfanserver/Dev/datasets',
                    help='path to input dataset')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='size of output node in a batch')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--fanouts', type=list, default=[10, 15],
                    help='Hidden state dimension')
parser.add_argument('--n_trial', type=int, default=5,
                    help='Number of times to repeat experiments')
args = parser.parse_args()

graph, node_labels, train_nids, valid_nids, test_nids = load_data(args)


# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)


node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sampler = dgl.dataloading.NeighborSampler(args.fanouts)
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=args.batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)

model = GCN(num_features, args.nhid, num_classes).to(device)

opt = torch.optim.Adam(model.parameters())

valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)
test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)

model_name = "GCN"
system_name = "DAFOS"


# create main directory: "Results/args.dataset"
dir_name = '{}/{}'.format('Results', args.dataset)
# create directory for saving a best model, i.e., "model": "Results/args.dataset/model"
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

best_accuracy = 0
cnt = 0
original_stdout = sys.stdout

for oiter in range(args.n_trial):
    for epoch in range(args.n_epochs):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                accuracy = sklearn.metrics.f1_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='micro')
                print('Epoch {} Loss {}'.format(epoch, loss.item()))

        model.eval()

        predictions = []
        labels = []
        with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            labels = torch.from_numpy(labels).float()
            predictions = torch.from_numpy(predictions).float()
            loss_valid = F.cross_entropy(predictions, labels)
            accuracy = sklearn.metrics.f1_score(labels, predictions, average='micro')
            print('Epoch {} Validation F1 {}'.format(epoch, accuracy))

    predictions = []
    labels = []
    test_f1s = []
    with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        labels = torch.from_numpy(labels).float()
        predictions = torch.from_numpy(predictions).float()
        accuracy = sklearn.metrics.f1_score(labels, predictions, average='micro')
        print('Test F1 {}'.format(accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy

