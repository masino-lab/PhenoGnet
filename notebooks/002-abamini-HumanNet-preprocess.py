# %% imports
import sys
import scipy.sparse
from ordered_set import OrderedSet
# File paths
file_humanNet = '../data/external/humannet/HumanNet-FN.tsv'
file_gene2id = '../data/processed/gene2id.txt'
file_hnet_npz = '../data/processed/hnet.npz'

# Read all unique genes from the first two columns
genes = OrderedSet()
with open(file_humanNet, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gene1, gene2 = parts[0], parts[1]
            genes.add(gene1)
            genes.add(gene2)
#print(genes)
# Sort the genes numerically
#sorted_genes = sorted(genes, key=lambda x: int(x))
sorted_genes = genes
# Write to the specified output file
with open(file_gene2id, 'w') as f:
    f.write(f"{len(sorted_genes)}\n")
    for idx, gene in enumerate(sorted_genes):
        f.write(f"{gene}\t{idx}\n")

# Create a gene-to-index mapping
gene2idx = {gene: idx for idx, gene in enumerate(sorted_genes)}

# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((len(sorted_genes), len(sorted_genes)), dtype=float)

# Populate the sparse matrix with weights
with open(file_humanNet, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            gene1, gene2, weight = parts[0], parts[1], float(parts[2])
            idx1, idx2 = gene2idx[gene1], gene2idx[gene2]
            sparse_matrix[idx1, idx2] = weight
            #sparse_matrix[idx2, idx1] = weight  # If the matrix is symmetric

# Save the sparse matrix in .npz format
scipy.sparse.save_npz(file_hnet_npz, sparse_matrix.tocoo())