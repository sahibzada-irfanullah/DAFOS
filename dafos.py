import torch.optim as optim
from dgl.dataloading import DataLoader
from adaptivesampler import *
from model import *
from utils import *


# Training function with adaptive sampling
def run(args, model, graph, labels, train_idx, valid_idx, test_idx, epochs, adaptive_sampler, batch_size, device):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Sort nodes by degree (node scoring)
    prioritized_nodes = node_scoring(graph)
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)
    train_idx = prioritized_nodes[train_idx]  # Prioritize training nodes

    current_fanout = args.fanouts
    previous_loss = float('inf')
    for oiter in range(args.n_trial):
        best_test = -1
        best_val = -1
        cnt = 0

        stop = False
        train_loss_fanout = []

        for epoch in range(args.n_epochs):
            model.train()

            sampler = adaptive_sampler.get_sampler(current_fanout)  # For two GCN layers

            train_dataloader = DataLoader(graph, train_idx, sampler, batch_size=batch_size, device=device, shuffle=True, drop_last=False)
            valid_dataloader = DataLoader(graph, valid_idx, sampler, batch_size=batch_size, device=device, shuffle=False, drop_last=False)
            test_dataloader = DataLoader(graph, test_idx, sampler, batch_size=batch_size, device=device, shuffle=False, drop_last=False)

            with tqdm.tqdm(train_dataloader) as tq:
                for step, (input_nodes, output_nodes, subgraphs) in enumerate(tq):
                    subgraph = subgraphs[0]  # Extract subgraph from the list
                    x = subgraph.srcdata['feat'].to(device)
                    labels = subgraphs[-1].dstdata['label']
                    logits = model(subgraphs, x)
                    loss = loss_fn(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss_fanout += [loss.item()]
                    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

            current_loss = np.average(train_loss_fanout)
            loss_diff = previous_loss - current_loss

            current_fanout = adaptive_sampler.update_fanout(current_fanout, loss_diff)


            model.eval()
            stop, cnt, best_val = evaluate(valid_dataloader, model, args,loss_fn, best_val, cnt, stop = stop, valid = True)
            if stop:
                break
        _, _, best_test = evaluate(test_dataloader, model, args, loss_fn, best_test, cnt, stop = False, valid = False)

# Main function to execute training
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph, node_labels, train_idx, valid_idx, test_idx = load_data(args)
    n_classes = (node_labels.max() + 1).item()
    # Load graph data
    graph = graph.to(device)
    graph.ndata['feat'] = graph.ndata['feat'].to(device)



    # Define GCN model
    in_feats = graph.ndata['feat'].shape[1]
    model = GCN(in_feats, args.nhid, n_classes).to(device)


    # Create adaptive sampler
    adaptive_sampler = AdaptiveNeighborSampler(threshold=0.01)

    # Train the model
    run(args, model, graph, node_labels, train_idx, valid_idx, valid_idx, args.n_epochs, adaptive_sampler, args.batch_size, device)






