import time
import pickle
import numpy as np
import os
import logging
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb  # Import wandb
from plot_tsne import plot_tsne
from evaluation import get_disease_auc, validate_with_full_dataset

torch.manual_seed(0)

def pooling(x, y2x, weights=None):
    if weights is not None:
        weights = weights.to(device=y2x.device, dtype=y2x.dtype).view(1, -1)
        y2x = y2x * weights

    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    x = torch.div(x, row_sum)
    return x

    
class Trainer(object):
    def __init__(self, model, tau, optimizer, log_every_n_steps, device, wandb_label):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tau = tau
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        # Initialize Weights and Biases
        wandb.init(project="PhenoGnet", mode="offline", name=wandb_label, reinit=True)
        self.run_dir = wandb.run.dir  # Directory for saving files
        logging.basicConfig(filename=osp.join(self.run_dir, 'training.log'), level=logging.DEBUG)
        logging.info(f"Training device: {self.device}.")
        
    def load_data(self, g_data, kg_data, labels, dis2hpo, dis2g, dis_path, beta, gamma,
                  concat=False, hpo_embeddings_concat=None, hpo_node_ic=None,
                  use_hpo_ic_pooling=False, use_hpo_ic_loss_weights=False,
                  hpo_ic_loss_min_weight=0.05, hpo_ic_pooling_min_weight=0.0):
        self.g_data = g_data.to(self.device)
        self.kg_data = kg_data.to(self.device)
        self.labels = labels.to(self.device)
        self.dis2hpo = dis2hpo
        self.dis2g = dis2g
        self.dis_path = dis_path
        self.beta = beta
        self.gamma = gamma
        self.concat = concat

        if hpo_node_ic is None:
            self.hpo_node_ic = None
        elif torch.is_tensor(hpo_node_ic):
            self.hpo_node_ic = hpo_node_ic.detach().cpu().float()
        else:
            self.hpo_node_ic = torch.tensor(hpo_node_ic, dtype=torch.float)
        self.use_hpo_ic_pooling = use_hpo_ic_pooling and self.hpo_node_ic is not None
        self.use_hpo_ic_loss_weights = use_hpo_ic_loss_weights and self.hpo_node_ic is not None
        self.hpo_ic_loss_min_weight = hpo_ic_loss_min_weight
        self.hpo_ic_pooling_min_weight = hpo_ic_pooling_min_weight

         # keep a CPU copy to avoid CUDA ↔ CPU transfers when logging/validating
        self.hpo_embeddings_concat = (
            hpo_embeddings_concat if hpo_embeddings_concat is None
            else hpo_embeddings_concat.cpu()
        )

    def _hpo_weights(self, floor=0.0):
        if self.hpo_node_ic is None:
            return None
        weights = self.hpo_node_ic.clone()
        if floor and floor > 0:
            weights = floor + (1.0 - floor) * weights
        return weights

    def _hpo_pooling_weights(self):
        if not self.use_hpo_ic_pooling:
            return None
        return self._hpo_weights(self.hpo_ic_pooling_min_weight)
        
    def nce_loss(self, gz, kgz, labels):
        eps = 1e-8 # To eliminate risk of div/0 or log(0) in loss calculation
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = torch.exp((kgz @ gz.T) / self.tau)
        labels = labels.to(dtype=similarity_matrix.dtype)
        positive_mask = labels > 0

        positive_weights = labels
        if self.use_hpo_ic_loss_weights:
            hpo_weights = self._hpo_weights(self.hpo_ic_loss_min_weight).to(
                device=similarity_matrix.device,
                dtype=similarity_matrix.dtype,
            )
            positive_weights = labels * hpo_weights.view(1, -1)

        positive_scores = similarity_matrix * positive_weights

        hpo_has_pos = positive_mask.any(dim=0)
        gene_has_pos = positive_mask.any(dim=1)

        hpo_den = similarity_matrix.sum(dim=0)
        hpo_pos = positive_scores.sum(dim=0)
        gene_den = similarity_matrix.sum(dim=1)
        gene_pos = positive_scores.sum(dim=1)

        if torch.any(hpo_has_pos):
            hpo_ratio = (hpo_pos[hpo_has_pos] + eps) / (hpo_den[hpo_has_pos] + eps)
            hpo_loss = -torch.log(hpo_ratio.clamp_min(eps)).mean()
        else:
            hpo_loss = similarity_matrix.new_tensor(0.0)

        if torch.any(gene_has_pos):
            gene_ratio = (gene_pos[gene_has_pos] + eps) / (gene_den[gene_has_pos] + eps)
            gene_loss = -torch.log(gene_ratio.clamp_min(eps)).mean()
        else:
            gene_loss = similarity_matrix.new_tensor(0.0)

        beta = self.beta  # this is a hyperparameter
        loss = beta * hpo_loss + (1 - beta) * gene_loss
        return loss

    def train(self, epochs, encoder_mode='hpo'):
        t0 = time.time()
        print(f"Start JCLModel training for {epochs} epochs using {encoder_mode} encoder mode.")
        logging.info(f"Start JCLModel training for {epochs} epochs using {encoder_mode} encoder mode.")
        training_range = tqdm(range(epochs))

        for epoch in training_range:
            g_h, kg_h = self.model(self.g_data, self.kg_data)
            g_z = self.model.nonlinear_transformation(g_h)
            kg_z = self.model.nonlinear_transformation(kg_h)
            loss = self.nce_loss(g_z, kg_z, self.labels)
            training_range.set_description('Loss %.4f' % loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % self.log_every_n_steps == 0:
                with torch.no_grad():
                    g_h_cpu = g_h.detach().cpu()
                    kg_h_cpu = kg_h.detach().cpu()
                    # if epoch % 25 == 0:
                    #     plot_tsne(g_h_cpu, kg_h_cpu, self.dis2g.to_dense(), self.dis2hpo.to_dense(), epoch)
                    
                    # Compute AUC based on encoder mode
                    if encoder_mode == 'hpo':
                        # Use HPO network encoder
                        d_h = pooling(g_h_cpu, self.dis2hpo.to_dense(), self._hpo_pooling_weights())
                        auc, ap = get_disease_auc(d_h, self.dis_path)
                    elif encoder_mode == 'hnet':
                        # Use Human network encoder
                        d_h = pooling(kg_h_cpu, self.dis2g.to_dense())
                        auc, ap = get_disease_auc(d_h, self.dis_path)
                    elif encoder_mode == 'combined':
                        # Combine both encoders (average embeddings)
                        d_h_hpo = pooling(g_h_cpu, self.dis2hpo.to_dense(), self._hpo_pooling_weights())
                        d_h_hnet = pooling(kg_h_cpu, self.dis2g.to_dense())
                        
                        # Ensure dimensions match before combining
                        if d_h_hpo.shape == d_h_hnet.shape:
                            d_h = (d_h_hpo + d_h_hnet) / 2.0  # Simple average
                        else:
                            # If dimensions don't match, use a projection matrix or another method
                            logging.warning(f"Dimension mismatch: HPO dim {d_h_hpo.shape}, HNET dim {d_h_hnet.shape}")
                            # For now, just use HPO encoder if dimensions don't match
                            d_h = d_h_hpo
                            
                        auc, ap = get_disease_auc(d_h, self.dis_path)
                    else:
                        logging.error(f"Unknown encoder mode: {encoder_mode}")
                        auc, ap = 0.0, 0.0
                    
                    # Log metrics to wandb
                    wandb.log({
                        'Train_loss': loss,
                        'auc': auc,
                        'ap': ap,
                        'beta': self.beta,
                        'tau': self.tau,
                    }, step=epoch)
                    
                logging.debug(f"Epoch: {epoch}\tTrain_Loss: {loss}\tEncoder: {encoder_mode}\tAUROC: {auc}\tAP: {ap}")
                
        t1 = time.time()
        logging.info("Training has finished.")
        logging.info(f"Training takes {(t1-t0)/60} mins")
        
        # Save model checkpoint
        checkpoint_name = f"checkpoint_{encoder_mode}_{epochs:03d}.pth.tar"
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'encoder_mode': encoder_mode,
            'beta': self.beta,
            'tau': self.tau
        }, f=osp.join(self.run_dir, checkpoint_name))
        
        logging.info(f"Model checkpoint and metadata have been saved at {self.run_dir}.")
        
        # Save both embeddings
        pickle.dump(g_h.detach().cpu(),
                    open(osp.join(self.run_dir, "hpo_embedding.pkl"), "wb"))
        pickle.dump(kg_h.detach().cpu(),
                    open(osp.join(self.run_dir, "gene_embedding.pkl"), "wb"))
        logging.info(f"Gene and HPO embeddings have been saved at {self.run_dir}.")
        
        # Return final performance metrics
        return auc, ap

    def validate_full_dataset(self, dataset_path, encoder_mode='hpo', save_path='../plots', label=''):
        with torch.no_grad():
            gamma = self.gamma
            self.model.eval()
            g_h, kg_h = self.model(self.g_data, self.kg_data)
            g_h_cpu = g_h.detach().cpu()
            kg_h_cpu = kg_h.detach().cpu()
            if self.concat and (self.hpo_embeddings_concat is not None):
                g_h_for_pool = torch.cat((g_h_cpu, self.hpo_embeddings_concat), dim=1)
            else:
                g_h_for_pool = g_h_cpu

            # Compute disease embeddings
            if encoder_mode == 'hpo':
                d_h = pooling(g_h_for_pool, self.dis2hpo.to_dense(), self._hpo_pooling_weights())
                auroc, auprc, roc, precision, recall = validate_with_full_dataset(dataset_path, encoder_mode, gamma, d_h)
            elif encoder_mode == 'hnet':
                d_h = pooling(kg_h_cpu, self.dis2g.to_dense())
                auroc, auprc, roc, precision, recall = validate_with_full_dataset(dataset_path, encoder_mode, gamma, d_h)
            elif encoder_mode == 'combined':
                d_h_hpo = pooling(g_h_for_pool, self.dis2hpo.to_dense(), self._hpo_pooling_weights())
                d_h_hnet = pooling(kg_h_cpu, self.dis2g.to_dense())
                auroc, auprc, roc, precision, recall = validate_with_full_dataset(dataset_path, encoder_mode, gamma, None, d_h_hpo, d_h_hnet)
            else:
                logging.error(f"Unknown encoder mode: {encoder_mode}")
                return 0.0, 0.0

            fpr, tpr, thresholds_roc = roc

            # Load disease pair indices and labels
            all_pairs = np.loadtxt(dataset_path, dtype=int)
            d1_idx = all_pairs[:, 0]
            d2_idx = all_pairs[:, 1]
            lbl = all_pairs[:, 2] if all_pairs.shape[1] > 2 else None

            def pair_cosine(tensor_emb, idx1, idx2):
                return F.cosine_similarity(tensor_emb[idx1], tensor_emb[idx2], dim=1).cpu().numpy()

            if encoder_mode == "combined":
                sims_hpo = pair_cosine(d_h_hpo, d1_idx, d2_idx)
                sims_hnet = pair_cosine(d_h_hnet, d1_idx, d2_idx)
                similarity_scores = (sims_hpo + sims_hnet) / 2.0
            else:
                similarity_scores = pair_cosine(d_h, d1_idx, d2_idx)

            # Save similarity scores
            os.makedirs(save_path, exist_ok=True)
            if lbl is not None:
                sim_array = np.column_stack((d1_idx, d2_idx, similarity_scores, lbl))
                header = "disease1,disease2,similarity,label"
            else:
                sim_array = np.column_stack((d1_idx, d2_idx, similarity_scores))
                header = "disease1,disease2,similarity"

            sim_file = f"{save_path}/similarity_scores_{encoder_mode}_{label}.csv"
            np.savetxt(sim_file, sim_array, delimiter=",", header=header, comments="")
            logging.info(f"Saved similarity scores to {sim_file}")

            # Plot histogram
            plt.figure(figsize=(8, 6))
            plt.hist(similarity_scores, bins=50, edgecolor="black")
            plt.xlabel("Cosine similarity")
            plt.ylabel("Number of disease pairs")
            plt.title(f"Distribution of Disease Similarity – {encoder_mode}+{label}")
            plt.grid(True, alpha=0.3)
            hist_file = f"{save_path}/similarity_hist_{encoder_mode}_{label}.png"
            plt.savefig(hist_file)
            plt.close()
            logging.info(f"Saved similarity histogram to {hist_file}")

            # Save ROC and PR data
            roc_data = np.column_stack((fpr, tpr, thresholds_roc))
            np.savetxt(f'{save_path}/roc_data_{encoder_mode}_{label}.csv', roc_data, delimiter=',', header='fpr,tpr,threshold', comments='')
            pr_data = np.column_stack((recall, precision))
            np.savetxt(f'{save_path}/pr_data_{encoder_mode}_{label}.csv', pr_data, delimiter=',', header='recall,precision', comments='')
            logging.info(f"Saved ROC and PR data.")

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {encoder_mode}+{label}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(f'{save_path}/roc_curve_{encoder_mode}_{label}.png')
            plt.close()

            # Plot PR curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {auprc:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {encoder_mode}+{label}')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.savefig(f'{save_path}/pr_curve_{encoder_mode}_{label}.png')
            plt.close()

            # ----------------- Additional Metrics Calculation -----------------
            from sklearn.metrics import f1_score, confusion_matrix

            best_f1 = 0
            best_threshold = 0.5
            thresholds_to_try = np.linspace(0, 1, 101)

            for threshold in thresholds_to_try:
                preds = (similarity_scores >= threshold).astype(int)
                f1 = f1_score(lbl, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            final_preds = (similarity_scores >= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(lbl, final_preds).ravel()

            f1 = f1_score(lbl, final_preds)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            print(f"\n[Validation Metrics @ optimal threshold = {best_threshold:.2f}]")
            print(f"F1 Score       : {f1:.4f}")
            print(f"Sensitivity    : {sensitivity:.4f}")
            print(f"Specificity    : {specificity:.4f}")
            print(f"PPV (Precision): {ppv:.4f}")
            print(f"NPV            : {npv:.4f}")

            metrics_log_path = f"{save_path}/metrics_{encoder_mode}_{label}.txt"
            with open(metrics_log_path, "w") as f:
                f.write(f"[Validation Metrics @ optimal threshold = {best_threshold:.2f}]\n")
                f.write(f"validation_auroc       : {auroc:.4f}\n")
                f.write(f"validation_auprc       : {auprc:.4f}\n")
                f.write(f"F1 Score       : {f1:.4f}\n")
                f.write(f"Sensitivity    : {sensitivity:.4f}\n")
                f.write(f"Specificity    : {specificity:.4f}\n")
                f.write(f"PPV (Precision): {ppv:.4f}\n")
                f.write(f"NPV            : {npv:.4f}\n")

            logging.info(f"Saved detailed metrics to {metrics_log_path}")

            wandb.log({
                'validation_auroc': auroc,
                'validation_auprc': auprc,
                'f1_score': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'optimal_threshold': best_threshold,
                'validation_encoder': encoder_mode
            })

            logging.info(f"Validation AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}")
            return auroc, auprc
        
    # In trainer.py, update the calculate_hits_metrics method:

    def calculate_hits_metrics(self, held_out_indices, target_type, rep_mode, save_path=None):
        """
        Calculates Hits@K metrics for all diseases and optionally saves held-out pairs.
        """
        self.model.eval()
        with torch.no_grad():
            # Get raw embeddings from encoders
            g_h, kg_h = self.model(self.g_data, self.kg_data) # g_h: HPO, kg_h: Gene
            # --- Step 1: Represent the Disease ---
            if rep_mode == "hpo":
                # Disease as average of its HPO terms
                disease_rep = pooling(g_h, self.dis2hpo.to_dense().to(self.device), self._hpo_pooling_weights())
            elif rep_mode == "gene":
                # Disease as average of its Genes
                disease_rep = pooling(kg_h, self.dis2g.to_dense().to(self.device))
            elif rep_mode == "combined":
                # Disease as average of both HPO and Gene embeddings
                d_h_hpo = pooling(g_h, self.dis2hpo.to_dense().to(self.device), self._hpo_pooling_weights())
                d_h_gene = pooling(kg_h, self.dis2g.to_dense().to(self.device))
                disease_rep = (d_h_hpo + d_h_gene) / 2.0

            # Calculate similarity between all diseases and genes
            sim_matrix = torch.mm(disease_rep, kg_h.t())
            
            hits = {1: 0, 5: 0, 10: 0, 100: 0}
            num_diseases = disease_rep.size(0)
            
            # Prepare to save indices if save_path is provided
            save_file = open(save_path, "w") if save_path else None
            if save_file:
                save_file.write("Disease_Index\tGene_Index\n")

            print(f"\n--- Hits@K Metrics (All {num_diseases} Diseases) ---")
            
            # Iterate through ALL diseases instead of a random subset
            for i in range(num_diseases):
                target_idx = held_out_indices[i].item()
                scores = sim_matrix[i]
                
                # Save indices to file
                if save_file:
                    save_file.write(f"{i}\t{target_idx}\n")
                
                # # Calculate the exact rank
                # rank = (torch.where(torch.argsort(scores, descending=True) == target_idx)[0].item()) + 1
                
                # # Update the hits counter
                # for k in [1, 5, 10, 100]:
                #     if rank <= k:
                #         hits[k] += 1
            
            if save_file:
                save_file.close()
                print(f"Held-out indices saved to {save_path}")

            # Final Results Calculation
            for k in [1, 5, 10, 100]:
                score = hits[k] / num_diseases
                print(f"Hits@{k}: {score:.4f}")
