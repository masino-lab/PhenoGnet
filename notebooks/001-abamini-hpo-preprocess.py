# %% imports
import json
import csv
import sys
import scipy.sparse
from ordered_set import OrderedSet
# %% file paths
file_nodes = '../data/external/hpo/hpo_class_nodes.csv'
file_edges = '../data/external/hpo/hpo_is_a_edges.csv'
file_hpo2id = '../data/processed/hpo2id.txt' 
file_hpo_npz = '../data/processed/hpo2hpo.npz'

# %% Process hpo_class_nodes.csv to create hpo2id.txt
with open(file_nodes, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    rows = list(reader)

# Determine the number of unique nodes
num_nodes = len(rows)

# Write to the output txt file
with open(file_hpo2id, 'w') as txtfile:
    # Write the number of nodes as the first line
    txtfile.write(f"{num_nodes}\n")
    # Write each hpo_id and node_idx pair
    for row in rows:
        hpo_id = row[1]
        node_idx = row[0]
        txtfile.write(f"{hpo_id}\t{node_idx}\n")

# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((num_nodes, num_nodes), dtype=float)

edges = OrderedSet()
with open(file_edges, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for parts in reader:
        if len(parts) >= 2:
            src, dst = int(parts[0]), int(parts[1])  # Convert to integer indices
            sparse_matrix[src, dst] = 1.00000000

# Save the sparse matrix in .npz format
scipy.sparse.save_npz(file_hpo_npz, sparse_matrix.tocoo())



