import argparse
import numpy as np
import torch
import torch.optim as optim
import os
from torch_geometric.data import Data
from utils import *
from model import GCN, RGCN, Projection, PhenoGnet, GAT
from trainer import Trainer
from log_results import log_run_results
from hyperparameter_tuning import run_hyperparameter_tuning

embeddings_path = "/data2/masino_lab/abamini/population-phenotyping/data/external/hpo/hpo_class_node_desc_embeddings_model_mpnet.npy"
validation_dataset_path = "/home/abamini/PhenoGnet/data/processed/full_dataset_test.txt"
train_dataset_path = "/home/abamini/PhenoGnet/data/processed/full_dataset_train.txt"
plot_save_path = "/home/abamini/PhenoGnet/plots"

parser = argparse.ArgumentParser(description="PyTorch JCLModel")
parser.add_argument("--data", default="../data/processed", help="path to dataset")
parser.add_argument("--h_dim", default=32, type=int, help="dimension of layer h")
parser.add_argument("--z_dim", default=32, type=int, help="dimension of layer z")
parser.add_argument("--tau", default=0.3, type=float, help="softmax temperature")
parser.add_argument("--lr", default=0.0008, type=float, help="learning rate") #original 0.003
parser.add_argument("--epochs", default=101, type=int, help="train epochs")
parser.add_argument("--disable-cuda", default=False, action="store_true", help="disable CUDA")
parser.add_argument("--log-every-n-steps", default=1, type=int, help="log every n steps")
parser.add_argument("--use_hpo_embeddings", default=1, type=int, help="use hpo sentence embeddings for nodes of hpo2hpo graph")
parser.add_argument("--concat_hpo_embeddings", default=0, type=int, help="concatenate HPO embeddings after training")
parser.add_argument("--hpo_embeddings_path", default=embeddings_path, help="path to HPO embeddings file")
parser.add_argument("--wandb_label", default="run", help="Name the wandb run label")
parser.add_argument("--encoder_mode", default="hnet", choices=["hpo", "hnet", "combined"], 
                    help="Encoder mode for disease AUC calculation (hpo, hnet, or combined)")
parser.add_argument("--full_dataset", default=validation_dataset_path, help="Path to full dataset for inference validation")
parser.add_argument("--beta", default=0, type=float, help="Beta coefficient for conmtasive loss")
parser.add_argument("--gamma", default=0, type=float, help="gamma coefficient for validation, only valid for combined mode")
parser.add_argument("--hyperparameter_tuning", default=False, help="Perform hyperparameter tuning")
parser.add_argument("--cv_folds", default=5, type=int, help="Number of cross-validation folds for hyperparameter tuning")
parser.add_argument("--tuning_dataset", default=train_dataset_path, help="Path to dataset for hyperparameter tuning")
parser.add_argument("--n_trials", default=30, type=int, help="Number of trials for Bayesian optimization")
parser.add_argument("--output_dir", default="./hyperparameter_tuning", help="Directory to save hyperparameter tuning results")
args = parser.parse_args()

device = torch.device("cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu")

# Create output directory for hyperparameter tuning if it doesn't exist
if args.hyperparameter_tuning:
    os.makedirs(args.output_dir, exist_ok=True)

# Load human net for GCN Model
hnadj = load_sparse(args.data+"/hnet.npz")
src = hnadj.row
dst = hnadj.col
hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min())
hn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)

# Load hpo adj matrix for GCN Model
hpo2hpo = load_sparse(args.data+"/hpo2hpo_rec.npz")
src = hpo2hpo.row
dst = hpo2hpo.col
hpo_edge_weight = torch.tensor(np.hstack((hpo2hpo.data, hpo2hpo.data)), dtype=torch.float)
hpo_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)
hpo2hpo = mx_to_torch_sparse_tesnsor(hpo2hpo).to_dense()

# Load gene2HPO align or g2hpo_all
g2hpo = load_sparse(args.data+"/g2hpo_all_ancestors.npz")
g2hpo = mx_to_torch_sparse_tesnsor(g2hpo).to_dense()

x = generate_sparse_one_hot(g2hpo.shape[0])

# Load HPO embeddings if provided, otherwise use one-hot encoding
if (args.hpo_embeddings_path) and (args.use_hpo_embeddings):
    print(f"Loading HPO embeddings from {args.hpo_embeddings_path}")
    try:
        hpo_embeddings = np.load(args.hpo_embeddings_path)
        y = torch.tensor(hpo_embeddings, dtype=torch.float)
        print("loaded")
        
        # Verify the dimensions match
        if y.shape[0] != g2hpo.shape[1]:
            print(f"Warning: Embedding count ({y.shape[0]}) doesn't match HPO count ({g2hpo.shape[1]})")
            print("Falling back to one-hot encoding")
            y = generate_sparse_one_hot(g2hpo.shape[1])
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Falling back to one-hot encoding")
        y = generate_sparse_one_hot(g2hpo.shape[1])
else:
    print("No HPO embeddings provided, using one-hot encoding")
    y = generate_sparse_one_hot(g2hpo.shape[1])

#Load HPO embeddings for concatenation if specified
hpo_embeddings_concat = None                         
if args.concat_hpo_embeddings:
    print(f"Loading HPO embeddings for concatenation from {args.hpo_embeddings_path}")
    try:
        hpo_embeddings_concat = torch.tensor(
            np.load(args.hpo_embeddings_path), dtype=torch.float
        )
        if hpo_embeddings_concat.shape[0] != g2hpo.shape[1]:
            print("Mismatch between concat-embeddings and #HPO terms – skipping.")
            hpo_embeddings_concat = None
    except Exception as e:
        print(f"Could not load concat embeddings: {e}")
        hpo_embeddings_concat = None

# # Create graph for GCN/GAT model
# g_data = Data(x=y, edge_index=hpo_edge_index, edge_weight=hpo_edge_weight) # (HPO Graph)

# Load HPO for RGCN Model
train_triples = load_triples(args.data)
edge_index, edge_type = get_kg_data(train_triples, num_rels=1)
g_data = Data(x=y,edge_index=edge_index, edge_type=edge_type, num_nodes=hpo2hpo.shape[1])

kg_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight) # Human net graph

# Load disease mappings for inference
dis2hpo = load_sparse(args.data+"/dis2hpo.npz") # disease to hpo needed for HPO encoder
dis2hpo = mx_to_torch_sparse_tesnsor(dis2hpo)

dis2g = load_sparse(args.data+"/dis2g.npz") # disease to gene needed for HNET encoder
dis2g = mx_to_torch_sparse_tesnsor(dis2g)

if args.hyperparameter_tuning:
    print("Starting hyperparameter tuning...")
    best_params = run_hyperparameter_tuning(
        args=args,
        g_data=g_data,
        kg_data=kg_data,
        g2hpo=g2hpo,
        dis2hpo=dis2hpo,
        dis2g=dis2g,
        device=device
    )
    
    # Save best parameters to a file
    with open(os.path.join(args.output_dir, "best_params.txt"), "w") as f:
        f.write("Best hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
            # Update args with best parameters for possible model training
            if hasattr(args, param):
                setattr(args, param, value)
    
    print(f"Best parameters saved to {os.path.join(args.output_dir, 'best_params.txt')}")
    
    # Skip regular training and validation if only hyperparameter tuning is required
    print("Hyperparameter tuning completed. Exiting...")
    exit()

# Initialize models
# g_encoder = GCN(nfeat=g_data.x.shape[1], nhid=args.h_dim) #GCN for HPO
# g_encoder = GAT(nfeat=g_data.x.shape[1], nhid=args.h_dim) #GAT for HPO
g_encoder = RGCN(num_nodes=g_data.num_nodes, nhid=args.h_dim, num_rels=2)
kg_encoder = GCN(nfeat=kg_data.x.shape[1], nhid=args.h_dim) #GCN for gene network

projection = Projection(args.h_dim, args.z_dim)
model = PhenoGnet(g_encoder, kg_encoder, projection)

# Initialize optimizer
opt = optim.RMSprop(model.parameters(), args.lr)

# Initialize trainer
trainer = Trainer(model, tau=args.tau, optimizer=opt, log_every_n_steps=args.log_every_n_steps, 
                 device=device, wandb_label=args.wandb_label)

# Load data into trainer
trainer.load_data(g_data, kg_data, g2hpo, dis2hpo, dis2g, args.data, args.beta, args.gamma,concat=bool(args.concat_hpo_embeddings),hpo_embeddings_concat=hpo_embeddings_concat)

print("Finish initializing...")
print(f"Using encoder mode: {args.encoder_mode}")
print("---------------------------------------")

# Train the model
trainer.train(args.epochs, encoder_mode=args.encoder_mode)

# After training, validate with the full dataset if provided
if args.full_dataset:
    print(f"\nRunning validation on full dataset: {args.full_dataset}")
    auroc, auprc = trainer.validate_full_dataset(args.full_dataset, encoder_mode=args.encoder_mode, save_path=plot_save_path, label=args.wandb_label)
    print(f"Validation results - AUROC: {auroc*100:.2f}% | AUPRC: {auprc*100:.2f}%")

# Log the run results
log_run_results(args, device, auroc, auprc)