import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

def mycs(x, y):
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return cosine_similarity(x, y)

def plot_tsne(hpo_embeddings, gene_embeddings, dis2g, dis2hpo, epoch, save_dir="Training_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy if torch tensor
    if torch.is_tensor(dis2g):
        dis2g = dis2g.cpu().numpy()
    if torch.is_tensor(dis2hpo):
        dis2hpo = dis2hpo.cpu().numpy()

    # Convert sparse to dense if needed
    if scipy.sparse.issparse(dis2g):
        dis2g = dis2g.toarray()
    if scipy.sparse.issparse(dis2hpo):
        dis2hpo = dis2hpo.toarray()

    # Determine common diseases
    common_diseases = np.intersect1d(np.nonzero(dis2g)[0], np.nonzero(dis2hpo)[0])

    # Initialize static variable to select 3 diseases only once
    if not hasattr(plot_tsne, "selected_diseases"):
        if len(common_diseases) < 3:
            raise ValueError("Not enough common diseases to select 3.")
        plot_tsne.selected_diseases = np.random.choice(common_diseases, 3, replace=False)

    selected_diseases = plot_tsne.selected_diseases

    # Get all indices (related to any disease)
    all_gene_indices = np.unique(np.nonzero(dis2g)[1])
    all_hpo_indices = np.unique(np.nonzero(dis2hpo)[1])

    base_gene_embeddings = gene_embeddings[all_gene_indices]
    base_hpo_embeddings = hpo_embeddings[all_hpo_indices]
    base_embeddings = torch.cat((base_hpo_embeddings, base_gene_embeddings), dim=0)
    base_colors = ['lightgray'] * base_embeddings.shape[0]

    # Highlight embeddings for selected diseases
    highlight_embeddings = []
    highlight_colors = []
    disease_colors = ['green', 'orange', 'purple']
    labels = ['Disease 1', 'Disease 2', 'Disease 3']

    for i, disease in enumerate(selected_diseases):
        gene_idx = np.nonzero(dis2g[disease])[0]
        hpo_idx = np.nonzero(dis2hpo[disease])[0]
        g_embeds = gene_embeddings[gene_idx]
        h_embeds = hpo_embeddings[hpo_idx]
        emb = torch.cat((h_embeds, g_embeds), dim=0)
        highlight_embeddings.append(emb)
        highlight_colors.extend([disease_colors[i]] * emb.shape[0])

    highlight_embeddings = torch.cat(highlight_embeddings, dim=0)

    # Combine all
    all_embeddings = torch.cat((base_embeddings, highlight_embeddings), dim=0)
    all_colors = base_colors + highlight_colors

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric=mycs)
    embeddings_2d = tsne.fit_transform(all_embeddings.detach().cpu().numpy())

    # Plot
    plt.figure(figsize=(8, 8))
    embeddings_2d = np.array(embeddings_2d)

    # Plot base
    base_len = len(base_colors)
    plt.scatter(embeddings_2d[:base_len, 0], embeddings_2d[:base_len, 1], c='lightgray', label='Other disease-related', alpha=0.4)

    # Plot highlights
    start = base_len
    for i, color in enumerate(disease_colors):
        count = highlight_colors.count(color)
        plt.scatter(embeddings_2d[start:start+count, 0],
                    embeddings_2d[start:start+count, 1],
                    c=color, label=labels[i], alpha=0.8)
        start += count

    plt.title(f't-SNE of All Disease-Related Embeddings + 3 Fixed Diseases (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()
