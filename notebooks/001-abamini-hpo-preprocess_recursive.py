import json
import csv
import sys
import scipy.sparse
from ordered_set import OrderedSet

# File paths
file_nodes = '../data/external/hpo/hpo_class_nodes.csv'
file_edges = '../data/external/hpo/hpo_is_a_edges.csv'
file_hpo2id = '../data/processed/hpo2id.txt' 
file_hpo_npz = '../data/processed/hpo2hpo_rec.npz'
file_train2id = '../data/processed/train2id_rec.txt'

# Read all nodes from hpo_class_nodes.csv
with open(file_nodes, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    nodes = list(reader)

# Determine the number of unique nodes
num_nodes = len(nodes)

# Write to the hpo2id.txt file
with open(file_hpo2id, 'w') as txtfile:
    # Write the number of nodes as the first line
    txtfile.write(f"{num_nodes}\n")
    # Write each hpo_id and node_idx pair
    for node in nodes:
        hpo_id = node[1]
        node_idx = node[0]
        txtfile.write(f"{hpo_id}\t{node_idx}\n")

# Read all edges to build parent-child relationships
node_to_parents = {int(node[0]): set() for node in nodes}  # Map each node to its direct parents
node_to_all_parents = {int(node[0]): set() for node in nodes}  # Map each node to all its ancestors

# First, collect direct parent relationships
with open(file_edges, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    
    edges_list = list(reader)
    
    for parts in edges_list:
        if len(parts) >= 2:
            child, parent = int(parts[0]), int(parts[1])
            node_to_parents[child].add(parent)

# Function to find all ancestors of a node
def find_all_ancestors(node, ancestors=None):
    if ancestors is None:
        ancestors = set()
    
    # Get direct parents
    direct_parents = node_to_parents[node]
    
    # Add direct parents to ancestors
    for parent in direct_parents:
        ancestors.add(parent)
        
        # Check if we already computed this parent's ancestors
        if parent in node_to_all_parents and node_to_all_parents[parent]:
            ancestors.update(node_to_all_parents[parent])
        else:
            # Recursively find all ancestors for each parent
            find_all_ancestors(parent, ancestors)
    
    return ancestors

# Build complete ancestor relationships
for node in node_to_parents:
    if not node_to_all_parents[node]:  # Only compute if not already done
        node_to_all_parents[node] = find_all_ancestors(node)


# Save the dictionary to a text file
with open("node_to_all_parents.txt", "w") as f:
    for node, parents in node_to_all_parents.items():
        f.write(f"{node}: {', '.join(map(str, parents))}\n")

print("Dictionary saved to node_to_all_parents.txt")

# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((num_nodes, num_nodes), dtype=float)

# Process all relationships (direct edges + ancestor relationships)
all_relationships = set()

# First add direct edges
for parts in edges_list:
    if len(parts) >= 2:
        src, dst = int(parts[0]), int(parts[1])
        all_relationships.add((src, dst))

# Add all ancestor relationships
for node, ancestors in node_to_all_parents.items():
    for ancestor in ancestors:
        all_relationships.add((node, ancestor))

# Write to train2id.txt and update sparse matrix
with open(file_train2id, 'w') as train2id_file:
    # Write the number of relationships
    train2id_file.write(f"{len(all_relationships)}\n")
    
    for src, dst in all_relationships:
        # Write to train2id with a 0 between the edges
        train2id_file.write(f"{src}\t0\t{dst}\n")
        
        # Add relationship to sparse matrix (bidirectional)
        sparse_matrix[src, dst] = 1.00000000
        sparse_matrix[dst, src] = 1.00000000

# Save the sparse matrix in .npz format
scipy.sparse.save_npz(file_hpo_npz, sparse_matrix.tocoo())

print(f"Total nodes: {num_nodes}")
print(f"Total relationships (including ancestors): {len(all_relationships)}")