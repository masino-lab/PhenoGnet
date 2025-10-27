# %% imports
import sys
import scipy.sparse
from ordered_set import OrderedSet
# File paths
file_dis_gene = '../data/external/DisGenet/all_gene_disease_associations.tsv'
file_gene2id = '../data/processed/gene2id.txt'
file_dis2id = '../data/processed/dis2id.txt'
output_file = '../data/processed/dis2g.txt'
file_di2gene_npz = '../data/processed/dis2g.npz'


# Load gene2id mapping
gene2id = {}
with open(file_gene2id, 'r') as f:
    num_genes = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        gene2id[parts[0]] = parts[1]
# Load dis2id mapping
dis2id = {}
with open(file_dis2id, 'r') as f:
    num_dis = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        dis2id[parts[0]] = parts[1]

# Process phenotype_to_genes file and replace IDs with indexes
with open(file_dis_gene, 'r') as f, open(output_file, 'w') as out:
    for line in f:
        if line.startswith('#'):
            continue  # Skip comments
        parts = line.strip().split('\t')
        
        # Ensure the line has enough columns
        if len(parts) <= 4:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        
        dis_id = parts[4]
        gene_id = parts[0]

        if dis_id not in dis2id:
            print(f"Warning: dis ID {dis_id} not found in dis2id mapping")

        if gene_id not in gene2id:
            print(f"Warning: Gene ID {gene_id} not found in gene2id mapping")

        dis_index = dis2id.get(dis_id, None)
        gene_index = gene2id.get(gene_id, None)
        
        if dis_index and gene_index:
            out.write(f"{dis_index} {gene_index}\n")

print(f"Processed file saved to {output_file}")


# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((num_dis, num_genes), dtype=float)
print(sparse_matrix)
# Populate the sparse matrix with weights
with open(output_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            gene, dis = int(parts[1]), int(parts[0])
            sparse_matrix[dis, gene] = 1.00000000000
            #sparse_matrix[gene, dis] = 1.00000000000


# Save the sparse matrix in .npz format
scipy.sparse.save_npz(file_di2gene_npz, sparse_matrix.tocoo())