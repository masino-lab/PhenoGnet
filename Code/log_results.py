import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
import pandas as pd
from datetime import datetime
import os

# Import existing modules
from utils import *
from model import GCN, RGCN, Projection, PhenoGnet, GAT
from trainer import Trainer

def log_run_results(args, device, auroc, auprc, log_file_path='run_log.xlsx'):
    """
    Log the run parameters and results to an Excel file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        auroc (float): Area Under the Receiver Operating Characteristic curve
        auprc (float): Area Under the Precision-Recall Curve
        log_file_path (str): Path to the Excel log file
    """
    # Prepare the run data
    current_time = datetime.now()
    run_data = {
        'Timestamp': [current_time],
        'Encoder Mode': [args.encoder_mode],
        'Learning Rate': [args.lr],
        'Epochs': [args.epochs],
        'H Dimension': [args.h_dim],
        'Z Dimension': [args.z_dim],
        'Tau': [args.tau],
        'Beta': [args.beta],
        'Device': [str(device)],
        'AUROC': [auroc],
        'AUPRC': [auprc]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(run_data)
    
    # Check if file exists
    if os.path.exists(log_file_path):
        # Read existing file
        existing_df = pd.read_excel(log_file_path)
        
        # Append new run
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Create new DataFrame if file doesn't exist
        updated_df = df
    
    # Save to Excel
    updated_df.to_excel(log_file_path, index=False)
    
    print(f"Run results logged to {log_file_path}")