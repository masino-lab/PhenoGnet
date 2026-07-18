import argparse
import numpy as np
import torch
import torch.optim as optim
import os
from pathlib import Path
from torch_geometric.data import Data
from utils import *
from model import GCN, RGCN, Projection, PhenoGnet, GAT
from trainer import Trainer
from log_results import log_run_results
from hyperparameter_tuning import run_hyperparameter_tuning

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

processed_data_path = str(REPO_ROOT / "data" / "processed")
embeddings_path = str(REPO_ROOT / "data" / "hpo_embeddings" / "hpo_class_node_desc_embeddings_model_mpnet.npy")
validation_dataset_path = str(REPO_ROOT / "data" / "processed" / "full_dataset_test.txt")
train_dataset_path = str(REPO_ROOT / "data" / "processed" / "full_dataset_train.txt")
plot_save_path = str(REPO_ROOT / "plots") # Path where different plots with evaluation metrics will go (ROC curves, PR curves, similarity histograms, csv files with plot data)

# Detailed argument guidance lives in README.md; argparse provides quick CLI help.
parser = argparse.ArgumentParser(
    description="Train and evaluate the PhenoGnet model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--data", default=processed_data_path, help="path to dataset")
parser.add_argument("--h_dim", default=32, type=int, help="dimension of layer h")
parser.add_argument("--z_dim", default=32, type=int, help="dimension of layer z")
parser.add_argument("--tau", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--lr", default=0.0008129, type=float, help="learning rate") #original 0.003
parser.add_argument("--epochs", default=150, type=int, help="train epochs")
parser.add_argument("--disable-cuda", default=False, action="store_true", help="disable CUDA")
parser.add_argument("--log-every-n-steps", default=1, type=int, help="log every n steps")
parser.add_argument("--use_hpo_embeddings", default=1, type=int, help="use hpo sentence embeddings for nodes of hpo2hpo graph")
parser.add_argument("--concat_hpo_embeddings", default=0, type=int, help="concatenate HPO embeddings after training")
parser.add_argument("--hpo_embeddings_path", default=embeddings_path, help="path to HPO embeddings file")
parser.add_argument("--hpo_graph_source", default="triples", choices=["triples", "direct", "closure"],
                    help="HPO graph for the GAT: train2id triples, direct hpo2hpo edges, or true-path closure")
parser.add_argument("--use_hpo_ic_edge_weights", default=False, action=argparse.BooleanOptionalAction,
                    help="Use HPO information content as scalar edge attributes in the GAT")
parser.add_argument("--hpo_ic_weight_mode", default="min", choices=["source", "target", "mean", "delta", "min", "max"],
                    help="How to convert node IC values into edge weights")
parser.add_argument("--hpo_ic_min_edge_weight", default=0.00, type=float,
                    help="Minimum normalized IC edge weight; use 0.0 to allow hard-zero edges")
parser.add_argument("--use_hpo_ic_loss_weights", default=False, action=argparse.BooleanOptionalAction,
                    help="Weight positive gene-HPO contrastive pairs by HPO information content")
parser.add_argument("--hpo_ic_loss_min_weight", default=0.00, type=float,
                    help="Minimum normalized IC weight for positive contrastive pairs")
parser.add_argument("--use_hpo_ic_pooling", default=False, action=argparse.BooleanOptionalAction,
                    help="Use IC-weighted HPO pooling for disease embeddings")
parser.add_argument("--hpo_ic_pooling_min_weight", default=0.0, type=float,
                    help="Minimum normalized IC weight for HPO disease pooling")
parser.add_argument("--wandb_label", default="run", help="Name the wandb run label")
parser.add_argument("--encoder_mode", default="hpo", choices=["hpo", "hnet", "combined"], 
                    help="Encoder mode for disease AUC calculation (hpo, hnet, or combined)")
parser.add_argument("--full_dataset", default=validation_dataset_path, help="Path to full dataset for inference validation")
parser.add_argument("--beta", default=0.9, type=float, help="Beta coefficient for contrastive loss")
parser.add_argument("--gamma", default=0, type=float, help="gamma coefficient for validation, only valid for combined mode")
parser.add_argument("--hyperparameter_tuning", default=False, action="store_true", help="Perform hyperparameter tuning")
parser.add_argument("--cv_folds", default=5, type=int, help="Number of cross-validation folds for hyperparameter tuning")
parser.add_argument("--tuning_dataset", default=train_dataset_path, help="Path to dataset for hyperparameter tuning")
parser.add_argument("--n_trials", default=30, type=int, help="Number of trials for Bayesian optimization")
parser.add_argument("--output_dir", default=str(REPO_ROOT / "hyperparameter_tuning"), help="Directory to save hyperparameter tuning results")
parser.add_argument("--calculate_hits_at_k", default="no", choices=["yes", "no"], help="calculate the hits@k metric")
parser.add_argument("--holdout_target", default="gene", choices=["gene", "hpo"], 
                    help="The entity type to hold out for prediction (the 'missing' link)")
parser.add_argument("--disease_rep_mode", default="hpo", choices=["hpo", "gene", "combined"], 
                    help="How to represent the disease embedding during Hits@K calculation")
parser.add_argument("--save_hold_out_gene", type=str, default="/home/abamini/PhenoGnet/Code/held_out_genes_new.txt", 
                    help="Path to save the disease and gene indices of held-out genes")
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

# Load HPO graph. The closure graph contains true-path ancestor connections.
hpo_graph_file = "hpo2hpo_rec.npz" if args.hpo_graph_source == "closure" else "hpo2hpo.npz"
hpo2hpo = load_sparse(args.data + "/" + hpo_graph_file)
hpo_edge_index = torch.tensor(np.vstack((hpo2hpo.row, hpo2hpo.col)), dtype=torch.long)
hpo_num_nodes = hpo2hpo.shape[1]

# Load gene2HPO align with ancestors; this is also the annotation corpus for IC.
g2hpo_sparse = load_sparse(args.data + "/g2hpo_all_ancestors.npz")
uses_hpo_ic = (
    args.use_hpo_ic_edge_weights
    or args.use_hpo_ic_loss_weights
    or args.use_hpo_ic_pooling
)
hpo_node_ic = compute_information_content(g2hpo_sparse) if uses_hpo_ic else None
g2hpo = mx_to_torch_sparse_tesnsor(g2hpo_sparse).to_dense()

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

# Create graph for the HPO GAT.
if args.hpo_graph_source == "triples":
    train_triples = load_triples(args.data)
    edge_index, edge_type = get_kg_data(train_triples, num_rels=1)
    g_data = Data(x=y, edge_index=edge_index, edge_type=edge_type, num_nodes=hpo_num_nodes)
else:
    g_data = Data(x=y, edge_index=hpo_edge_index, num_nodes=hpo_num_nodes)

if hpo_node_ic is not None:
    g_data.node_ic = hpo_node_ic

if args.use_hpo_ic_edge_weights:
    hpo_edge_weight = build_ic_edge_weight(
        g_data.edge_index,
        hpo_node_ic,
        mode=args.hpo_ic_weight_mode,
        min_weight=args.hpo_ic_min_edge_weight,
    )
    g_data.edge_weight = hpo_edge_weight
    g_data.edge_attr = hpo_edge_weight.view(-1, 1)
    print(
        f"Using HPO IC edge attributes from {args.hpo_graph_source} graph "
        f"with mode={args.hpo_ic_weight_mode}, floor={args.hpo_ic_min_edge_weight}."
    )

if args.use_hpo_ic_loss_weights:
    print(f"Using HPO IC-weighted contrastive positives with floor={args.hpo_ic_loss_min_weight}.")
if args.use_hpo_ic_pooling:
    print(f"Using HPO IC-weighted disease pooling with floor={args.hpo_ic_pooling_min_weight}.")

kg_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight) # Human net graph

# Load disease mappings for inference
dis2hpo = load_sparse(args.data+"/dis2hpo.npz") # disease to hpo needed for HPO encoder
dis2hpo = mx_to_torch_sparse_tesnsor(dis2hpo)

dis2g = load_sparse(args.data+"/dis2g.npz") # disease to gene needed for HNET encoder
dis2g = mx_to_torch_sparse_tesnsor(dis2g)

#################################
held_out_indices = None
if args.calculate_hits_at_k == "yes":
    print(f"Performing leave-one-out holdout for target: {args.holdout_target}...")
    
    if args.holdout_target == "gene":
        # Target is Gene: Mask one gene in dis2g
        matrix = dis2g.to_dense().clone()
        indices = []
        for i in range(matrix.shape[0]):
            pos = torch.where(matrix[i] > 0)[0]
            if len(pos) > 0:
                idx = pos[torch.randint(0, len(pos), (1,))].item()
                matrix[i, idx] = 0
                indices.append(idx)
            else:
                indices.append(-1) #when a disease has no gene association, we append -1 to maintain the index alignment
        dis2g = matrix.to_sparse()
        held_out_indices = torch.tensor(indices)
    
    elif args.holdout_target == "hpo":
        # Target is HPO: Mask one HPO in dis2hpo
        matrix = dis2hpo.to_dense().clone()
        indices = []
        for i in range(matrix.shape[0]):
            pos = torch.where(matrix[i] > 0)[0]
            if len(pos) > 0:
                idx = pos[torch.randint(0, len(pos), (1,))].item()
                matrix[i, idx] = 0
                indices.append(idx)
            else:
                indices.append(-1)
        dis2hpo = matrix.to_sparse()
        held_out_indices = torch.tensor(indices)
#############################



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
hpo_edge_dim = g_data.edge_attr.size(-1) if getattr(g_data, "edge_attr", None) is not None else None
g_encoder = GAT(nfeat=g_data.x.shape[1], nhid=args.h_dim, edge_dim=hpo_edge_dim) #GAT for HPO
# g_encoder = RGCN(num_nodes=g_data.num_nodes, nhid=args.h_dim, num_rels=2)
kg_encoder = GCN(nfeat=kg_data.x.shape[1], nhid=args.h_dim) #GCN for gene network

projection = Projection(args.h_dim, args.z_dim)
model = PhenoGnet(g_encoder, kg_encoder, projection)

# Initialize optimizer
opt = optim.RMSprop(model.parameters(), args.lr)

# Initialize trainer
trainer = Trainer(model, tau=args.tau, optimizer=opt, log_every_n_steps=args.log_every_n_steps, 
                 device=device, wandb_label=args.wandb_label)

# Load data into trainer
trainer.load_data(
    g_data,
    kg_data,
    g2hpo,
    dis2hpo,
    dis2g,
    args.data,
    args.beta,
    args.gamma,
    concat=bool(args.concat_hpo_embeddings),
    hpo_embeddings_concat=hpo_embeddings_concat,
    hpo_node_ic=hpo_node_ic,
    use_hpo_ic_pooling=args.use_hpo_ic_pooling,
    use_hpo_ic_loss_weights=args.use_hpo_ic_loss_weights,
    hpo_ic_loss_min_weight=args.hpo_ic_loss_min_weight,
    hpo_ic_pooling_min_weight=args.hpo_ic_pooling_min_weight,
)

print("Finish initializing...")
print(f"Using encoder mode: {args.encoder_mode}")
print("---------------------------------------")

# Train the model
trainer.train(args.epochs, encoder_mode=args.encoder_mode)

if args.calculate_hits_at_k == "yes":
    trainer.calculate_hits_metrics(
        held_out_indices=held_out_indices, 
        target_type=args.holdout_target,
        rep_mode=args.disease_rep_mode,
        save_path=args.save_hold_out_gene  # Pass the new argument
    )

# After training, validate with the full dataset if provided
if args.full_dataset:
    print(f"\nRunning validation on full dataset: {args.full_dataset}")
    auroc, auprc = trainer.validate_full_dataset(args.full_dataset, encoder_mode=args.encoder_mode, save_path=plot_save_path, label=args.wandb_label)
    print(f"Validation results - AUROC: {auroc*100:.2f}% | AUPRC: {auprc*100:.2f}%")

# Log the run results
log_run_results(args, device, auroc, auprc)
