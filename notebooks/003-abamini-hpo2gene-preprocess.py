# # %% imports
# import sys
# import scipy.sparse
# from ordered_set import OrderedSet
# # File paths
# file_hpo = '../data/external/hpo/phenotype_to_genes.txt'
# file_gene2id = '../data/processed/gene2id.txt'
# file_hpo2id = '../data/processed/hpo2id.txt'
# output_file = '../data/processed/g2hpo.txt'
# file_g2hpo_npz = '../data/processed/g2hpo.npz'


# # Load gene2id mapping
# gene2id = {}
# with open(file_gene2id, 'r') as f:
#     num_genes = int(f.readline().strip())
#     next(f)  # Skip header
#     for line in f:
#         parts = line.strip().split('\t')
#         gene2id[parts[0]] = parts[1]
# # Load hpo2id mapping
# hpo2id = {}
# with open(file_hpo2id, 'r') as f:
#     num_hpo = int(f.readline().strip())
#     next(f)  # Skip header
#     for line in f:
#         parts = line.strip().split('\t')
#         hpo2id[parts[0]] = parts[1]

# # Process phenotype_to_genes file and replace IDs with indexes
# with open(file_hpo, 'r') as f, open(output_file, 'w') as out:
#     for line in f:
#         if line.startswith('#'):
#             continue  # Skip comments
#         parts = line.strip().split('\t')
#         # if len(parts) < 2:
#         #     continue
#         hpo_id = parts[0].replace(":", "_")  # Convert HP:0000006 -> HP_0000006
#         gene_id = parts[2]
#         if hpo_id not in hpo2id:
#             print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")

#         if gene_id not in gene2id:
#             print(f"Warning: Gene ID {gene_id} not found in gene2id mapping")

#         hpo_index = hpo2id.get(hpo_id, None)
#         gene_index = gene2id.get(gene_id, None)
#         if hpo_index and gene_index:
#             out.write(f"{hpo_index} {gene_index}\n")

# print(f"Processed file saved to {output_file}")


# # Initialize a sparse matrix
# sparse_matrix = scipy.sparse.lil_matrix((num_genes, num_hpo), dtype=float)
# print(sparse_matrix)
# # Populate the sparse matrix with weights
# with open(output_file, 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) >= 2:
#             gene, hpo = int(parts[1]), int(parts[0])
#             #sparse_matrix[hpo, gene] = 1.00000000000
#             sparse_matrix[gene, hpo] = 1.00000000000


# # Save the sparse matrix in .npz format
# scipy.sparse.save_npz(file_g2hpo_npz, sparse_matrix.tocoo())





import sys
import scipy.sparse
from ordered_set import OrderedSet


# File paths
file_hpo2gene = '../data/external/hpo/phenotype_to_genes.txt'
file_gene2hpo = '../data/external/hpo/genes_to_phenotype.txt'
file_gene2id = '../data/processed/gene2id.txt'
file_hpo2id = '../data/processed/hpo2id.txt'
output_file = '../data/processed/g2hpo_all.txt'
file_g2hpo_npz = '../data/processed/g2hpo_all.npz'
file_node_to_all_parents = 'node_to_all_parents.txt'

# Load gene2id mapping
gene2id = {}
with open(file_gene2id, 'r') as f:
    num_gene = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        gene2id[parts[0]] = parts[1]

# Load hpo2id mapping
hpo2id = {}
with open(file_hpo2id, 'r') as f:
    num_hpo = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        hpo2id[parts[0]] = parts[1]

def process_hpo2gene(input_file, output_file, gene2id, hpo2id):
    """Process file where HPO IDs come first, followed by ncbi gene IDs"""
    with open(input_file, 'r') as f, open(output_file, 'a') as out:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            hpo_id = parts[0].replace(":", "_")  # Convert HP:0000006 -> HP_0000006
            gene_id = parts[2]

            if hpo_id not in hpo2id:
                print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")
                continue

            if gene_id not in gene2id:
                print(f"Warning gene ID {gene_id} not found in gene2id mapping")
                continue

            hpo_index = hpo2id[hpo_id]
            gene_index = gene2id[gene_id]
            out.write(f"{hpo_index} {gene_index}\n")

def process_gene2hpo(input_file, output_file, gene2id, hpo2id):
    """Process file where gene IDs come first, followed by HPO IDs"""
    with open(input_file, 'r') as f, open(output_file, 'a') as out:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            gene_id = parts[0]
            hpo_id = parts[2].replace(":", "_")  # Convert HP:0000006 -> HP_0000006

            if hpo_id not in hpo2id:
                print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")
                continue

            if gene_id not in gene2id:
                print(f"Warning: gene ID {gene_id} not found in gene2id mapping")
                continue

            hpo_index = hpo2id[hpo_id]
            gene_index = gene2id[gene_id]
            out.write(f"{hpo_index} {gene_index}\n")

# Clear the output file before appending
with open(output_file, 'w') as out:
    pass

# Process each file with the appropriate function
process_hpo2gene(file_hpo2gene, output_file, gene2id, hpo2id)
process_gene2hpo(file_gene2hpo, output_file, gene2id, hpo2id)

print(f"Processed combined file saved to {output_file}")

# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((num_gene, num_hpo), dtype=float)

# Populate the sparse matrix with weights
with open(output_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            gene, hpo = int(parts[1]), int(parts[0])
            sparse_matrix[gene, hpo] = 1.0

# Save the combined sparse matrix in .npz format
scipy.sparse.save_npz(file_g2hpo_npz, sparse_matrix.tocoo())
print(f"Sparse matrix saved to {file_g2hpo_npz}")