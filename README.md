# A Generative-Predictive AI Pipeline for Accelerating Discovery of High-Efficiency Additives for Aqueous Zinc-Ion Batteries

## Abstract
Improving the Coulombic Efficiency (CE) of aqueous zinc-ion batteries is crucial for minimizing side reactions like hydrogen evolution and dendrite formation. High CE in Zn|Cu asymmetric cells signifies less energy lost per cycle, reflecting the reversibility of Zn plating and stripping on Cu substrate, a key metric for interfacial stability and reaction efficiency. However, traditional discovery is limited by screening known compounds, a time-consuming process that overlooks vast chemical spaces.

This project introduces a pipeline leveraging generative Artificial Intelligence (AI) for the design of novel additives. The methodology first built a computational profile for 575 literature-reported additives by calculating 1517 distinct features, including fingerprints and descriptors from Density Functional Theory. After testing various machine learning models, a HistGB model, tuned via Bayesian optimization with the Optuna framework, was trained to predict CE, establishing a robust evaluation function with a cross-validated R2 of 0.82. Then, a generative MoLeR (Molecule Learning Representation) model was implemented, proposing 500 novel structures scored by the predictive model. MoLeR was iteratively fine-tuned using the 50 highest-scoring molecules from each round. After several cycles, the process converged, identifying a top candidate molecule (CC(N)C(=O)NC(C)C(=O)NO) with a predicted CE of 97.65%. This validates the closed-loop framework can autonomously navigate chemical space to discover novel, high-performing structures. Future work will focus on integrating structural constraints and multi-objective criteria to improve chemical interpretability and practical relevance. This work pioneers generative AI pipelines in the field of energy storage, establishing a new data-driven path for designing advanced battery materials.

---

## Repository Overview
This repository contains the code and data for the generative-predictive AI pipeline described above. The project is organized into the following directories:

- **`density_functional_theory_of_molecules/`**: Contains `.log` and `.xyz` files for molecular simulations and Density Functional Theory (DFT) calculations.
- **`feature_engineering_code/`**: Python scripts for feature extraction, including RDKit, Mordred, and xTB descriptors.
- **`generative_model_code/`**: Code for the MoLeR generative model, used to propose novel molecular structures.
- **`generative_model_output/`**: Outputs from the generative model, including proposed molecules and their predicted CE scores.
- **`predictive_model_code/`**: Machine learning models for CE prediction, including feature selection and model training pipelines.
- **`predictive_model_output/`**: Outputs from the predictive model, including performance metrics and predictions for novel molecules.

---

## Key Features

### 1. **Feature Engineering**
- **Tools Used**: RDKit, Mordred, xTB
- **Purpose**: Extract molecular descriptors and fingerprints for predictive modeling.
- **Example**: The `feature_engineering_code/feature_engineering.ipynb` notebook demonstrates the pipeline for generating 3D molecular structures, running xTB calculations, and parsing outputs.

### 2. **Predictive Modeling**
- **Model**: HistGB (Histogram-based Gradient Boosting)
- **Optimization**: Bayesian optimization with Optuna
- **Performance**: Cross-validated R2 of 0.82
- **Example**: The `predictive_model_code` directory contains scripts for feature selection, model training, and evaluation.

### 3. **Generative Modeling**
- **Model**: MoLeR (Molecule Learning Representation)
- **Process**: Iterative fine-tuning with top-scoring molecules
- **Output**: Novel molecular structures with high predicted CE


## How to Use This Repository

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `rdkit`, `mordred`, `shap`, `xgboost`, `optuna`, `scipy`, `matplotlib`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/NealKapadia/Generative-Predictive-AI-Model.git
   cd Generative-Predictive-AI-Model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
1. **Feature Engineering**:
   - Run the `feature_engineering_code/feature_engineering.ipynb` notebook to extract molecular descriptors.
2. **Predictive Modeling**:
   - Train the predictive model using scripts in `predictive_model_code`.
3. **Generative Modeling**:
   - Generate novel molecules using the `generative_model_code` scripts.

---

## Results
- **Top Candidate Molecule**: CC(N)C(=O)NC(C)C(=O)NO
- **Predicted CE**: 97.65%

---

## Future Work
- Integrate structural constraints for better chemical interpretability.
- Explore multi-objective optimization to balance CE with other properties.

---

## Authors
- Neal Kapadia
- Jiaqi Ke
- Laisuo Su*

---

## Acknowledgments
This work was supported by The University of Texas at Dallas and the Su Lab.
