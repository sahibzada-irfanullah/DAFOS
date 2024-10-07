
## DOFAS

This project provides a script for training a GNN model using Dynamic Adaptive Fanout Optimization Sampler (DAFOS) on the arxiv and reddit datasets.

### Requirements
#### OS: 
- Ubuntu 20 or Higher
- Windows 10 or Higher

#### Python: 
- 3.11 or Higher

#### Pytorch: 
- torchaudio=2.4.1 
- torchtriton=3.0.0 
- torchvision=0.19.1
- pytorch=2.4.1


#### DGL:
- dgl=2.4.0


## Usage

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone git@github.com:sahibzada-irfanullah/DAFOS.git
cd DAFOS
```
### Run the Training Script
To train the DOFAS-GCN model, you can use the command line interface. The script accepts several arguments to customize the training process. Here’s how to use it:
```
python main_args.py --dataset <dataset_name> --n_epochs <number_of_epochs> --path <dataset_path> --n_stops <number_of_stops> --batch_size <batch_size> --nhid <hidden_state_dimension> --fanouts <fanout_list> --n_trial <number_of_trials>
```


## Arguments and Default Values
- `--dataset`: (str) The name of the dataset to use for training. Options: `arxiv` (default) or `reddit`.

- `--n_epochs`: (int) The number of epochs to train the model. Default is `300`.

- `--path`: (str) Path to the input dataset. Default is `datasets`.

- `--n_stops`: (int) The number of batches to stop training after the F1 score does not improve. Default is `200`.

- `--batch_size`: (int) The size of output nodes in a batch. Default is `1024`.

- `--nhid`: (int) The dimension of the hidden state. Default is `256`.

- `--fanouts`: (list) A list defining the fanout for each layer. Default is `[10, 15]`.

- `--n_trial`: (int) The number of times to repeat the experiments. Default is `5`.


### Example Command
To train a DOFAS-GCN model on the arxiv dataset for 300 epochs with a batch size of 1024, you would run:
```bash
python main_args.py --dataset arxiv --n_epochs 300 --batch_size 1024 --n_trial 5
```



