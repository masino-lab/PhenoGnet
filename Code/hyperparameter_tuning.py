import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from model import GCN, GAT, Projection, PhenoGnet
from trainer import Trainer
import optuna
from optuna.samplers import TPESampler
import json
import wandb
import matplotlib.pyplot as plt

def run_hyperparameter_tuning(args, g_data, kg_data, g2hpo, dis2hpo, dis2g, device):
    """
    Run hyperparameter tuning using Bayesian optimization with cross-validation
    
    Args:
        args: Command line arguments
        g_data: HPO graph data
        kg_data: Human net graph data
        g2hpo: Gene to HPO mapping
        dis2hpo: Disease to HPO mapping
        dis2g: Disease to gene mapping
        device: Device to run computations on
        
    Returns:
        dict: Best hyperparameters
    """
    
    def objective(trial):
        # Define the hyperparameters to tune
        beta = trial.suggest_float("beta", 0.1, 0.9, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 50, 300, step=50)
        tau = trial.suggest_float("tau", 0.1, 2.0, step=0.1)
        
        # Define cross-validation strategy
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Load disease pair data for CV from tuning dataset
        disease_pairs = []
        labels = []
        
        with open(args.tuning_dataset, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                dis1_idx = int(parts[0])
                dis2_idx = int(parts[1])
                label = int(parts[2])
                disease_pairs.append((dis1_idx, dis2_idx))
                labels.append(label)
        
        disease_pairs = np.array(disease_pairs)
        labels = np.array(labels)
        
        cv_scores = []
        fold_num = 1
        
        # Create a directory for this trial
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        for train_idx, val_idx in kf.split(disease_pairs):
            print(f"Starting fold {fold_num}/{args.cv_folds} for trial {trial.number}")
            
            # Get training and validation sets for this fold
            train_pairs = disease_pairs[train_idx]
            train_labels = labels[train_idx]
            val_pairs = disease_pairs[val_idx]
            val_labels = labels[val_idx]
            
            # Save validation data for this fold
            val_data_path = os.path.join(trial_dir, f"fold_{fold_num}_val.txt")
            with open(val_data_path, 'w') as f:
                for i in range(len(val_pairs)):
                    f.write(f"{val_pairs[i][0]}\t{val_pairs[i][1]}\t{val_labels[i]}\n")
            
            # Initialize models with current hyperparameters
            g_encoder = GAT(nfeat=g_data.x.shape[1], nhid=args.h_dim)
            kg_encoder = GCN(nfeat=kg_data.x.shape[1], nhid=args.h_dim)
            projection = Projection(args.h_dim, args.z_dim)
            model = PhenoGnet(g_encoder, kg_encoder, projection)
            
            # Initialize optimizer
            opt = optim.RMSprop(model.parameters(), lr)
            
            # Initialize trainer
            wandb_label = f"trial_{trial.number}_fold_{fold_num}"
            trainer = Trainer(model, tau=tau, optimizer=opt, log_every_n_steps=args.log_every_n_steps,
                             device=device, wandb_label=wandb_label)
            
            # Load data into trainer
            trainer.load_data(g_data, kg_data, g2hpo, dis2hpo, dis2g, args.data, beta)
            
            # Train the model
            trainer.train(epochs, encoder_mode=args.encoder_mode)
            
            # Validate the model using the validation set
            auroc, auprc = trainer.validate_full_dataset(val_data_path, encoder_mode=args.encoder_mode,
                                                        save_path=trial_dir, label=f"fold_{fold_num}")
            
            print(f"Fold {fold_num} - AUROC: {auroc*100:.2f}% | AUPRC: {auprc*100:.2f}%")
            
            # Add scores for this fold
            cv_scores.append((auroc, auprc))
            
            # Increment fold counter
            fold_num += 1
            
            # Clean up to free memory
            del trainer
            del model
            del opt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average scores across folds
        mean_auroc = np.mean([score[0] for score in cv_scores])
        mean_auprc = np.mean([score[1] for score in cv_scores])
        
        # Log results to a file
        with open(os.path.join(trial_dir, "results.txt"), "w") as f:
            f.write(f"Trial {trial.number}\n")
            f.write(f"Parameters: beta={beta}, lr={lr}, epochs={epochs}, tau={tau}\n")
            f.write(f"Mean AUROC: {mean_auroc*100:.2f}%\n")
            f.write(f"Mean AUPRC: {mean_auprc*100:.2f}%\n")
            f.write("\nFold results:\n")
            for i, (auroc, auprc) in enumerate(cv_scores):
                f.write(f"Fold {i+1}: AUROC={auroc*100:.2f}%, AUPRC={auprc*100:.2f}%\n")
        
        # Create combined metric (can be adjusted based on importance)
        # Using F1-like combination of AUROC and AUPRC
        combined_metric = 2 * (mean_auroc * mean_auprc) / (mean_auroc + mean_auprc) if (mean_auroc + mean_auprc) > 0 else 0
        
        return combined_metric
    
    # Set up the study
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Hyperparameter tuning completed.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(args.output_dir, "optimization_history.png"))
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(args.output_dir, "parameter_importances.png"))
    
    # Save all study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, "all_trials.csv"))
    
    # Save study as pickle file
    with open(os.path.join(args.output_dir, "study.json"), "w") as f:
        json_str = optuna.study.StudyManager._dump_study(study, include_best_trial=True)
        json.dump(json.loads(json_str), f, indent=4)
    
    return study.best_params