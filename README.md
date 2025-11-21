## BiMba: Predicting Protein Binding Sites using Vision Mamba

 
Accurately identifying protein binding sites is a central challenge in structural biology. Binding sites on protein surfaces, consisting of groups of residues, govern how proteins recognize and interact with their partners; therefore, identifying them is essential for understanding biological function and for guiding the design of effective drugs and biomolecules. Despite major progress in computational approaches, their performance remains limited because most models underrepresent the combined influence of surface properties and residue-level information, leaving room for improvement. Here, we introduce BiMba, a state-space–driven deep learning framework that leverages the efficient long-range modeling capability of the Vision Mamba architecture to learn from 3D protein surfaces represented as 2D physicochemical grids. BiMba integrates multiple complementary sources of information, capturing both geometric and physicochemical determinants of molecular recognition into surface patches, which are encoded into 2D images after adding additional residue-level descriptors such as polarity and secondary structure, yielding a unified representation that couples spatial topology with biochemical context. BiMba demonstrates superior performance across diverse and specialized benchmark datasets, surpassing existing state-of-the-art methods. In addition, BiMba incorporates perturbation-based and gradient-based interpretability analyses
by extracting hidden attentions from Mamba layers, enabling visualization of feature relevance and biologically meaningful residue clusters. Overall, our findings establish state-space models as efficient, interpretable, and scalable architectures for molecular surface learning, advancing the application of deep learning in structural bioinformatics.


# Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
   - [Create Conda Environment](#create-conda-environment)  
   - [Install Dependencies](#install-dependencies)  
3. [Repository Structure](#repository-structure)  
4. [Datasets & Reproducibility](#datasets--reproducibility)  
5. [Running BiMba](#running-bimba)  
   - [Testing](#testing)  
   - [Example Inference](#example-inference)  
6. [Preprocessed Features](#preprocessed-features)  
7. [Citation](#citation)  
8. [Contact](#contact)

---

# Features

- Vision-Mamba architecture for efficient long-range modeling  
- Unified 2D physicochemical surface-patch representation  
- Integration of geometric + residue-level descriptors  
- Interpretability via perturbation-based and gradient-based methods  
- Hidden attention extraction from Mamba layers  

---

# Installation

We strongly recommend using the provided `environment.yml` file.

---

## Create Conda Environment

```bash
conda env create -f environment.yml
conda activate bimba
