import json
import csv
import sys
import scipy.sparse
from ordered_set import OrderedSet

# File paths
file_nodes = '../data/external/hpo/hpo_class_nodes.csv'
file_edges = '../data/external/hpo/hpo_is_a_edges.csv'
file_hpo2id = '../data/processed/hpo2id.txt' 
file_hpo_npz = '../data/processed/hpo2hpo.npz'
file_train2id = '../data/processed/train2id.txt'

# Process hpo_class_nodes.csv to create hpo2id.txt
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

# Process edges and save to train2id.txt
with open(file_edges, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    
    # Open train2id.txt to write edges
    with open(file_train2id, 'w') as train2id_file:
        # Count the number of edges
        edges_list = list(reader)
        train2id_file.write(f"{len(edges_list)}\n")
        
        for parts in edges_list:
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                # Write to train2id with a 0 between the edges
                train2id_file.write(f"{src}\t0\t{dst}\n")
                
                # Add edge to sparse matrix (unidirectional as in original code)
                sparse_matrix[src, dst] = 1.00000000
                sparse_matrix[dst, src] = 1.00000000

# Save the sparse matrix in .npz format
scipy.sparse.save_npz(file_hpo_npz, sparse_matrix.tocoo())