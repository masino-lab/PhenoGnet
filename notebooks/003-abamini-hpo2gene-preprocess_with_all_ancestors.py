import sys
import scipy.sparse
from ordered_set import OrderedSet

# File paths
file_hpo2gene = '../data/external/hpo/phenotype_to_genes.txt'
file_gene2hpo = '../data/external/hpo/genes_to_phenotype.txt'
file_gene2id = '../data/processed/gene2id.txt'
file_hpo2id = '../data/processed/hpo2id.txt'
output_file = '../data/processed/g2hpo_all_ancestors.txt'
file_g2hpo_npz = '../data/processed/g2hpo_all_ancestors.npz'
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

# Load node to all parents mapping
node_to_parents = {}
with open(file_node_to_all_parents, 'r') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 2:
            node = parts[0].strip()
            parents = parts[1].strip().split(', ')
            node_to_parents[node] = parents

def process_file(input_file, output_file, gene2id, hpo2id, node_to_parents, gene_first=False):
    with open(input_file, 'r') as f, open(output_file, 'a') as out:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            gene_id = parts[0] if gene_first else parts[2]
            hpo_id = parts[2].replace(":", "_") if gene_first else parts[0].replace(":", "_")

            if hpo_id not in hpo2id:
                print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")
                continue

            if gene_id not in gene2id:
                print(f"Warning: gene ID {gene_id} not found in gene2id mapping")
                continue

            hpo_index = hpo2id[hpo_id]
            gene_index = gene2id[gene_id]

            # Write the primary HPO-gene relationship
            out.write(f"{hpo_index} {gene_index}\n")

            # Write all ancestor nodes
            if hpo_index in node_to_parents:
                for ancestor in node_to_parents[hpo_index]:
                    out.write(f"{ancestor} {gene_index}\n")

# Clear the output file before appending
with open(output_file, 'w') as out:
    pass

# Process both HPO2Gene and Gene2HPO files
process_file(file_hpo2gene, output_file, gene2id, hpo2id, node_to_parents, gene_first=False)
process_file(file_gene2hpo, output_file, gene2id, hpo2id, node_to_parents, gene_first=True)

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
