import torch
import numpy as np
import scipy.sparse as sp


########################################################################
# Sparse Matrix Utils
########################################################################
#Loads sparse matrices from files
def load_sparse(path):
    return sp.load_npz(path).tocoo()

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
#Converts scipy sparse matrices to PyTorch sparse tensors

def mx_to_torch_sparse_tesnsor(mx):
    """Convert scipy sparse coo matrix to a torch sparse tensor"""
    sparse_mx = mx.astype(np.float32)
    sparse_mx.eliminate_zeros()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    size = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, size)
#Creates sparse one-hot encoding tensors

def generate_sparse_one_hot(num_ents, dtype=torch.float32):
    """ Creates a two-dimensional sparse tensor with ones along the diagnoal as one-hot encoding. """
    diag_size = num_ents
    diag_range = list(range(num_ents))
    diag_range = torch.tensor(diag_range)

    return torch.sparse_coo_tensor(
        indices=torch.vstack([diag_range, diag_range]),
        values=torch.ones(diag_size, dtype=dtype),
        size=(diag_size, diag_size))


def compute_information_content(annotation_mx, smoothing=1.0, normalize=True):
    """Estimate HPO information content from true-path-propagated annotations."""
    if sp.issparse(annotation_mx):
        counts = np.asarray(annotation_mx.sum(axis=0)).ravel().astype(np.float32)
        total = annotation_mx.shape[0]
    elif torch.is_tensor(annotation_mx):
        counts = annotation_mx.to_dense().sum(dim=0).detach().cpu().numpy().astype(np.float32)
        total = annotation_mx.shape[0]
    else:
        counts = np.asarray(annotation_mx).sum(axis=0).astype(np.float32)
        total = annotation_mx.shape[0]

    freq = (counts + smoothing) / (total + smoothing)
    ic = -np.log(np.clip(freq, 1e-12, 1.0))

    if normalize:
        ic_min = ic.min()
        ic_max = ic.max()
        ic = (ic - ic_min) / (ic_max - ic_min + 1e-12)

    return torch.tensor(ic, dtype=torch.float)


def build_ic_edge_weight(edge_index, node_ic, mode="min", min_weight=0.05):
    """Map per-node IC values to scalar HPO edge weights."""
    if not torch.is_tensor(node_ic):
        node_ic = torch.tensor(node_ic, dtype=torch.float)

    src = edge_index[0]
    dst = edge_index[1]
    src_ic = node_ic[src]
    dst_ic = node_ic[dst]

    if mode == "source":
        edge_weight = src_ic
    elif mode == "target":
        edge_weight = dst_ic
    elif mode == "mean":
        edge_weight = (src_ic + dst_ic) / 2.0
    elif mode == "delta":
        edge_weight = torch.abs(src_ic - dst_ic)
    elif mode == "min":
        edge_weight = torch.minimum(src_ic, dst_ic)
    elif mode == "max":
        edge_weight = torch.maximum(src_ic, dst_ic)
    else:
        raise ValueError(f"Unknown IC edge-weight mode: {mode}")

    max_weight = edge_weight.max()
    if max_weight > 0:
        edge_weight = edge_weight / max_weight

    if min_weight and min_weight > 0:
        edge_weight = min_weight + (1.0 - min_weight) * edge_weight
    return edge_weight.float()


########################################################################
# Knowledge Graph Utils
########################################################################
#: Loads knowledge graph triples from files
def load_triples(path):
    """Load knowledge graphs for RGAE model
    
    :param path: dir path of data file
    :return: train/valid/test datasets
    """
    train_total = 0
    triples = []
    train_file = path + "/train2id.txt"

    def load(file):
        triples = []
        with open(file, "r") as f:
            total = (int)(f.readline())
            for line in f:
                line = line.strip().split()
                h, r, t = line
                triples.append(((int)(h), (int)(r), (int)(t)))
        return total, triples

    train_total, triples = load(train_file)

    print("GO(%d) datasets loaded." % (train_total))

    return triples
# Generates inverse relations for knowledge graphs
def generate_inverses(triples, num_rels):
    """ Generate inverse relations """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()
    return inverse_relations

def generate_self_loops(num_ents, num_rels, device='cpu'):
    """ Generates self-loop triples and then applies edge dropout """

    # Create a new relation id for self loop relation.
    all = torch.arange(num_ents, device=device)[:, None]
    id  = torch.empty(size=(num_ents, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_ents, 3)

    return self_loops

def add_inverse_and_self(triples, num_ents, num_rels, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation.
    all = torch.arange(num_ents, device=device)[:, None]
    id  = torch.empty(size=(num_ents, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_ents, 3)

    return torch.cat([triples, inverse_relations, self_loops], dim=0)
# Prepares edge indices and edge types for graph processing
def get_kg_data(triples, num_rels):
    triples = torch.tensor(triples, dtype=torch.long)
    inverse_triples = generate_inverses(triples, num_rels)
    triples = torch.cat([triples, inverse_triples], dim=0)
    edge_index = torch.cat([triples[:, 0, None], triples[:, 2, None]], dim=1).permute(1, 0)
    edge_type = triples[:, 1, None].view(-1)
    return edge_index, edge_type
