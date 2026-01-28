# 💊 Graph-Embedding-Toolkit 

> **Predicting Novel Therapeutic Indications for Established Drugs via Graph-based Embeddings and Machine Learning Models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Bioinformatics-green.svg)](#)

This repository provides a modular framework for drug repurposing by leveraging **Knowledge Graph Embeddings (KGEs)**. It transforms heterogeneous biomedical data into low-dimensional vector spaces to predict new drug-disease treatment links with high accuracy.

---

## 🧪 Achievements

My research benchmarks KGE-based methodologies against traditional state-of-the-art approaches.

* 🚀 **Performance:** better performance compared to the random walk-based method (e.g., Ruiz et al.'s Multiscale Interactome [1]).
* 📉 **Efficiency:** reduction of the dimensionality of massive biological networks while preserving critical semantic relationships.

---
## 🛠️ Methodology
This project is organized into four modular phases:
1. Knowledge Graph Construction
   - Transforms raw biomedical data into a Directed Heterogeneous Multigraph.
2. Embedding Generation
   - Maps entities and relations into a continuous vector space
   - Topology: Captures both local graph neighborhoods and global structures.
4. ML Model Training
   Uses the generated embeddings as features for downstream predictive tasks
   - Task: Link Prediction (Drug-Disease treatment pairs)
   - Models: Includes Classifiers (Random Forest, XGBoost) and Clustering algorithms.
6. Predictive Analysis
   - Direct comparison against the Ruiz et al. Random Walk approach.

## 🚀 Getting Started

Clone the Repo:

git clone [https://github.com/camiearth/Graph-Embedding-Toolkit.git](https://github.com/camiearth/Graph-Embedding-Toolkit.git)

## References
[1] Ruiz, C., Zitnik, M., & Leskovec, J. Identification of disease treatment mechanisms through the multiscale interactome. Nat Commun., Vol. 12, no. 1796. https://doi.org/10.1038/s41467-021-21770-8 (2021).

[2] Gualdi F, Oliva B, Piñero J. Predicting gene disease associations with knowledge graph embeddings for diseases with curtailed information. NAR Genom Bioinform. 2024 May 14;6(2):lqae049. doi: 10.1093/nargab/lqae049. PMID: 38745993; PMCID: PMC11091931.
