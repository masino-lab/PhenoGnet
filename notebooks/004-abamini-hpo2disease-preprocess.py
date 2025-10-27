import sys
import scipy.sparse
from ordered_set import OrderedSet

# File paths
file_hpo2dis = '../data/external/hpo/hpo_to_disease.txt'
file_dis2hpo = '../data/external/hpo/disease_to_hpo.txt'
file_dis2id = '../data/processed/dis2id.txt'
file_hpo2id = '../data/processed/hpo2id.txt'
output_file = '../data/processed/dis2hpo.txt'
file_dis2hpo_npz = '../data/processed/dis2hpo.npz'

# Load dis2id mapping
dis2id = {}
with open(file_dis2id, 'r') as f:
    num_dis = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        dis2id[parts[0]] = parts[1]

# Load hpo2id mapping
hpo2id = {}
with open(file_hpo2id, 'r') as f:
    num_hpo = int(f.readline().strip())
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        hpo2id[parts[0]] = parts[1]

def process_hpo2dis(input_file, output_file, dis2id, hpo2id):
    """Process file where HPO IDs come first, followed by disease IDs"""
    with open(input_file, 'r') as f, open(output_file, 'a') as out:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            hpo_id = parts[0].replace(":", "_")  # Convert HP:0000006 -> HP_0000006
            dis_id = parts[1]

            if hpo_id not in hpo2id:
                print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")
                continue

            if dis_id not in dis2id:
                print(f"Warning: dis ID {dis_id} not found in dis2id mapping")
                continue

            hpo_index = hpo2id[hpo_id]
            dis_index = dis2id[dis_id]
            out.write(f"{hpo_index} {dis_index}\n")

def process_dis2hpo(input_file, output_file, dis2id, hpo2id):
    """Process file where disease IDs come first, followed by HPO IDs"""
    with open(input_file, 'r') as f, open(output_file, 'a') as out:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            dis_id = parts[0]
            hpo_id = parts[1].replace(":", "_")  # Convert HP:0000006 -> HP_0000006

            if hpo_id not in hpo2id:
                print(f"Warning: HPO ID {hpo_id} not found in hpo2id mapping")
                continue

            if dis_id not in dis2id:
                print(f"Warning: dis ID {dis_id} not found in dis2id mapping")
                continue

            hpo_index = hpo2id[hpo_id]
            dis_index = dis2id[dis_id]
            out.write(f"{hpo_index} {dis_index}\n")

# Clear the output file before appending
with open(output_file, 'w') as out:
    pass

# Process each file with the appropriate function
process_hpo2dis(file_hpo2dis, output_file, dis2id, hpo2id)
process_dis2hpo(file_dis2hpo, output_file, dis2id, hpo2id)

print(f"Processed combined file saved to {output_file}")

# Initialize a sparse matrix
sparse_matrix = scipy.sparse.lil_matrix((num_dis, num_hpo), dtype=float)

# Populate the sparse matrix with weights
with open(output_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            dis, hpo = int(parts[1]), int(parts[0])
            sparse_matrix[dis, hpo] = 1.0

# Save the combined sparse matrix in .npz format
scipy.sparse.save_npz(file_dis2hpo_npz, sparse_matrix.tocoo())
print(f"Sparse matrix saved to {file_dis2hpo_npz}")