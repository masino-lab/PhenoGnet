## PhenoGnet

arXiv: 2509.14037 ([https://arxiv.org/abs/2509.14037](https://arxiv.org/abs/2509.14037))
License: MIT ([https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

This repository contains the official implementation for PhenoGnet, a novel graph-based contrastive learning framework designed to predict disease similarity.

PhenoGnet integrates gene functional interaction networks and the Human Phenotype Ontology (HPO) to learn powerful embeddings for genes and phenotypes. By aligning these two views, the model can compute disease similarity scores that capture complex biological relationships, outperforming existing state-of-the-art methods.

Model Architecture

PhenoGnet consists of two key components: an Intra-view Model to encode the gene and phenotype graphs separately, and a Cross-view Model to align them in a shared latent space.

1.  Intra-view Model:

      * Gene Network: A Graph Convolutional Network (GCN) is used to encode the gene functional interaction network (from HumanNet).
      * Phenotype Network: A Graph Attention Network (GAT) is used to encode the Human Phenotype Ontology (HPO) graph. The initial features for HPO terms are generated from their textual descriptions using Sentence-BERT (all-mpnet-base-v2).

2.  Cross-view Model:

      * A shared-weight multilayer perceptron (MLP) projects the embeddings from both the GCN and GAT into a common latent space.
      * Contrastive learning is applied to train the entire model. Known gene-phenotype associations are used as positive pairs, and randomly sampled unrelated pairs serve as negatives. This process "pulls" related gene and phenotype embeddings closer together and "pushes" unrelated ones apart.

3.  Disease Similarity Prediction:

      * Diseases are represented by the mean embedding (average-pooling) of their associated genes and/or phenotypes.
      * The similarity between any two diseases is calculated using the cosine similarity of their final embedding vectors.

Installation

1.  Clone the repository:
    ```
    git clone 
    cd PhenoGnet
    ```

2.  Install the required dependencies (e.g., using pip or conda):
    ```
    pip install -r requirements.txt
    ```
Citation

If you use this code or our work, please cite the paper:

```
@misc{baminiwatte2025phenognet,
title={PhenoGnet: A Graph-Based Contrastive Learning Framework for Disease Similarity Prediction},
author={Ranga Baminiwatte and Kazi Jewel Rana and Aaron J. Masino},
year={2025},
eprint={2509.14037},
archivePrefix={arXiv},
primaryClass={q-bio.GN}
}
```

Acknowledgments

[cite\_start]This work was supported by the NIH funded Center of Biomedical Research Excellence in Human Genetics at Clemson University[cite: 260].

License

This project is licensed under the MIT License - see the LICENSE file for details.
