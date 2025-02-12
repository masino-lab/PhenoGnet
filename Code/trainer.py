import time
import pickle
import logging
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import wandb  # Import wandb

from evaluation import get_disease_auc#, topk


torch.manual_seed(0)

def pooling(x, y2x):
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    x = torch.div(x, row_sum)
    return x

    
class Trainer(object):
    def __init__(self, model, tau, optimizer, log_every_n_steps, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tau = tau
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        # Initialize Weights and Biases
        wandb.init(project="PhenoGnet",mode="offline")
        self.run_dir = wandb.run.dir  # Directory for saving files
        logging.basicConfig(filename=osp.join(self.run_dir, 'training.log'), level=logging.DEBUG)
        logging.info(f"Training device: {self.device}.")
        
    # def load_data(self, g_data, kg_data, labels, d2g, dis_path):
    #     self.g_data = g_data.to(self.device)
    #     self.kg_data = kg_data.to(self.device)
    #     self.labels = labels.to(self.device)
    #     self.d2g = d2g
    #     self.dis_path = dis_path

    def load_data(self, g_data, kg_data, labels,dis_path):
        self.g_data = g_data.to(self.device)
        self.kg_data = kg_data.to(self.device)
        self.labels = labels.to(self.device)
        self.dis_path = dis_path
        
    # def nce_loss(self, gz, kgz, labels):
    #     gz = F.normalize(gz, dim=1)
    #     kgz = F.normalize(kgz, dim=1)
    #     similarity_matrix = gz @ kgz.T
    #     similarity_matrix = torch.exp(similarity_matrix / self.tau)
    #     similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
    #     positives_sum = torch.sum(similarity_matrix * labels, 1)
    #     loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
    #     return loss
    
    def nce_loss(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        #print(gz)
        kgz = F.normalize(kgz, dim=1) #nan
        #print(kgz)
        similarity_matrix = gz @ kgz.T
        #print(similarity_matrix)
        #print(labels)
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, 1)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss
    

    def train(self, epochs):
        t0 = time.time()
        print(f"Start JCLModel training for {epochs} epochs.")
        logging.info(f"Start JCLModel training for {epochs} epochs.")
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
                    g_h = g_h.detach().cpu()
                    #d_h = pooling(g_h, self.d2g.to_dense())
                    #auc, ap = get_disease_auc(d_h, self.dis_path)
                    # Log metrics to wandb
                    wandb.log({
                        'loss': loss,
                        #'auc': auc,
                        #'ap': ap
                    }, step=epoch)
                #logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tAUROC: {auc}\tAP: {ap}")
                logging.debug(f"Epoch: {epoch}\tLoss: {loss}")
        t1 = time.time()
        logging.info("Training has finished.")
        logging.info(f"Training takes {(t1-t0)/60} mins")
        checkpoint_name = "checkpoint_{:03d}.pth.tar".format(epochs)
        torch.save({'epoch':epochs,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict()},
                   f=osp.join(self.run_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata have been saved at {self.run_dir}.")
        pickle.dump(g_h,
                    open(osp.join(self.run_dir, "gene_embedding.pkl"), "wb"))
        logging.info(f"Gene embedding has been saved at {self.run_dir}.")
    #     auc, ap, *_ = self.infer()
    #     logging.info(f"AUROC: {auc} \tAP: {ap}")

    # def infer(self):
    #     with torch.no_grad():
    #         self.model.eval()
    #         g_h = self.model.get_gene_embeddings(self.g_data)
    #         g_h = g_h.detach().cpu()
    #         d = pooling(g_h, self.d2g.to_dense())
    #         auroc, ap= get_disease_auc(d, self.dis_path)
    #         a, tk = topk(d, self.dis_path, 10)
    #         print(f"AUROC: {auroc*100} | AUPRC: {ap*100}")
    #     return auroc, ap, a, tk