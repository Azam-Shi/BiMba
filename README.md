# BiMba: Predicting Protein Binding Sites using Vision Mamba

 
Accurately identifying protein binding sites is a central challenge in structural biology. Binding sites on protein surfaces—consisting of groups of interacting residues—govern how proteins recognize and interact with their partners; therefore, identifying them is essential for understanding biological function and for guiding the design of effective drugs and biomolecules. Despite major progress in computational approaches, their performance remains limited because most models underrepresent the combined influence of surface properties and residue-level physicochemical information. 

Here, we introduce **BiMba**, a state-space–driven deep learning framework that leverages the efficient long-range modeling capability of the **Vision Mamba** architecture to learn from 3D protein surfaces represented as **2D physicochemical grids**. **BiMba** integrates multiple complementary sources of information—capturing geometric and physicochemical determinants of molecular recognition—into 2D surface patches enriched with residue-level descriptors such as polarity and secondary structure. This unified representation couples spatial topology with biochemical context.

**BiMba** achieves superior performance across diverse and specialized benchmark datasets, surpassing existing state-of-the-art methods. In addition, **BiMba** incorporates perturbation-based and gradient-based interpretability analyses by extracting hidden attentions from Mamba layers, enabling visualization of feature relevance and biologically meaningful residue clusters. Overall, our findings establish state-space models as efficient, interpretable, and scalable architectures for molecular surface learning, advancing deep learning applications in structural bioinformatics.

---

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
   - [Create Conda Environment](#create-conda-environment)  
3. [Running BiMba](#running-bimba)  
   - [Training Example](#training-example)  
   - [Inference Example](#inference-example)  
4. [Preprocessing Scripts](#preprocessing-scripts)  
5. [Contact](#contact)

---

## Features

&nbsp;&nbsp;&nbsp;&nbsp;• Vision-Mamba architecture for efficient long-range modeling  
&nbsp;&nbsp;&nbsp;&nbsp;• Unified 2D physicochemical surface-patch representation  
&nbsp;&nbsp;&nbsp;&nbsp;• Integration of geometric and residue-level descriptors  
&nbsp;&nbsp;&nbsp;&nbsp;• Interpretability via perturbation-based and gradient-based analyses  
&nbsp;&nbsp;&nbsp;&nbsp;• Extraction of hidden Mamba attentions  
&nbsp;&nbsp;&nbsp;&nbsp;• Designed for protein surface learning and interface prediction  

---

## Installation

We strongly recommend using the provided `environment.yml` file.  
However, the model can also be installed manually.

### Create Conda Environment

You may create and activate the environment using:

```bash
conda create -n bimba python=3.10
conda activate bimba
pip install -r requirements.yml
```
---

## Running BiMba

### Training Example
- An example training script is available at:  
  👉 [bimba_train.py](https://github.com/Azam-Shi/BiMba/blob/main/bimba_train.py)
- Lists of training and testing sets used in this study are available in:  
  👉 [data](https://github.com/Azam-Shi/BiMba/tree/main/data)

### Inference Example
- Pretrained model weights are provided in:  
  👉 [model](https://github.com/Azam-Shi/BiMba/tree/main/model)
- A complete inference demonstration notebook is available at:  
  👉 [bimba_test.ipynb](https://github.com/Azam-Shi/BiMba/blob/main/bimba_test.ipynb)
  
&nbsp;&nbsp;&nbsp;&nbsp;This notebook can be used to generate residue-level prediction scores (one probability per residue indicating interface or non-interface) for input proteins, which will be saved in the Test_Results folder.

---

## Preprocessing Scripts  
Scripts for preprocessing input data can be found at [data_prepare](https://github.com/Azam-Shi/BiMba/tree/main/data_prepare) directory.  

---

## Contact

For questions, feedback, or collaborations:

&nbsp;&nbsp;&nbsp;&nbsp;**Azam Shirali**  
&nbsp;&nbsp;&nbsp;&nbsp;Bioinformatics Research Group (BioRG)  
&nbsp;&nbsp;&nbsp;&nbsp;Florida International University  
&nbsp;&nbsp;&nbsp;&nbsp;📧 **ashir018@fiu.edu**

