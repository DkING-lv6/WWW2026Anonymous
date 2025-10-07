# PAGE: Progressive Anomaly Generation Network for Semi-Supervised Graph Anomaly Detection

## Overview

Semi-supervised graph anomaly detection aims to identify anomalous nodes that deviate from normal graph patterns in node attributes or structural connectivity, using only a small subset of labeled normal nodes. Existing methods typically generate anomalies by either perturbing node attributes in isolation or synthesizing embeddings without considering structural information, failing to capture the synergistic effects between attribute and structure perturbations. This limits their ability to detect complex anomalies.

To address this, we propose **PAGE (Progressive Anomaly Generation Network)**, which enhances anomaly detection by progressively injecting attribute noise and then applying structure perturbations. This process simulates the evolution from simple to complex anomalies. A novel anomaly progressive constraint loss enforces that hybrid anomalies exhibit a higher degree of abnormality than primary anomalies, improving anomaly quantification. PAGE also integrates reconstruction and classification losses to optimize node representations and boost detection accuracy.

Extensive experiments on benchmark datasets demonstrate that PAGE significantly outperforms state-of-the-art methods, validating its effectiveness for graph anomaly detection.

---

## Key Features

- Progressive attribute and structure perturbation to generate increasingly complex anomalies.
- Anomaly progressive constraint loss to enforce anomaly severity ordering.
- Joint optimization with reconstruction and classification losses.
- Semi-supervised learning framework requiring only partial normal node labels.
- Demonstrated superior performance on multiple benchmark datasets.

---
## Implementation Details

All experiments are conducted on a server equipped with an **NVIDIA RTX 4090 GPU**, utilizing **PyTorch 1.11.0** with **CUDA 11.5** for acceleration. The model is trained using the **Adam optimizer**.

To mimic real-world scenarios with scarce anomaly samples, training adopts a **semi-supervised learning** setting where only a subset of normal nodes is labeled. Specifically, a random subset of nodes is selected as supervision with ratios from the set \{0.2, 0.3, 0.5, 0.6, 0.8\}, while the remaining nodes remain unlabeled during training.

## Usage

1. **Environment Setup**  
   Install required Python packages (PyTorch, numpy, dgl, sklearn, etc.):

   ```bash
   pip install -r requirements.txt

## Training
We provide the test code. We will release the training code after the paper is accepted.
python run.py



## Datasets
For convenience, some datasets can be obtained from [google drive link](https://drive.google.com/drive/folders/1rEKW5JLdB1VGwyJefAD8ppXYDAXc5FFj?usp=sharing.). 
We sincerely thank the researchers for providing these datasets.
Due to the Copyright of DGraph-Fin, you need to download from [DGraph-Fin](https://dgraph.xinye.com/introduction).
We evaluate PAGE on six widely used benchmark datasets. The statistics are summarized as follows:  

| Dataset   | Nodes     | Edges      | Attributes | Anomalies (Rate) |
|-----------|-----------|------------|-------------|------------------|
| Amazon    | 11,944    | 4,398,392  | 25          | 821 (6.9%)       |
| T-Finance | 39,357    | 1,222,543  | 10          | 1,803 (4.6%)     |
| Reddit    | 10,984    | 168,016    | 64          | 366 (3.3%)       |
| Elliptic  | 46,564    | 73,248     | 93          | 4,545 (9.8%)     |
| Photo     | 7,535     | 119,043    | 745         | 698 (9.2%)       |
| DGraph    | 3,700,550 | 73,105,508 | 17          | 15,509 (1.3%)    |

---

