# Loan Approval Prediction using Machine Learning, Deep Learning & Explainable AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-brightgreen) ![Explainability](https://img.shields.io/badge/XAI-SHAP-purple)

> A comparative research project evaluating neural networks and Random Forest for binary loan approval prediction, with an explainability workflow based on SHAP.

## Overview

This project investigates automated loan approval prediction using a dataset of **45,000 applications**, **13 input features**, and a binary target (`loan_status`: `1` = approved, `0` = rejected). It compares a multilayer perceptron (MLP), an artificial neural network (ANN) with early stopping, and a Random Forest classifier. The accompanying notebook also contains a **SHAP explainability analysis** for the Random Forest.

**Research manuscript:** [Machine Learning and Deep Learning Approaches for Loan Approval Prediction: A Comparative Study](paper/Loan_Approval_XAI_Enhanced_Manuscript.pdf)

**Authors:** Riyadh Hossain, Mahmudul Hasan Tanvir, Walid Ahmed  
**Affiliation:** Department of Computer Science & Engineering, East West University

## Research objectives

- Analyze applicant, financial, and credit-history attributes associated with loan approval outcomes.
- Compare tree-based machine learning with feedforward neural networks on structured tabular data.
- Evaluate predictive performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Examine global feature importance and generate local/global SHAP explanations for the Random Forest.
- Highlight interpretability and fairness considerations relevant to credit decision-support.

## Dataset

| Property | Description |
| --- | --- |
| File | [`data/loan_data.csv`](data/loan_data.csv) |
| Records | 45,000 |
| Input features | 13 |
| Target | `loan_status` (0 = rejected, 1 = approved) |
| Missing values | None reported in the manuscript |
| Class distribution | Approximately 77.8% rejected / 22.2% approved |
| Source | [Loan approval classification dataset — Kaggle](https://www.kaggle.com/datasets/riyadh262/loan-dataset) (as cited in the manuscript) |

Features include age, gender, education, income, employment experience, home ownership, requested loan amount, loan purpose, interest rate, loan-to-income ratio, credit history length, credit score, and previous defaults.

**Data provenance:** This repository contains the user-provided CSV. Verify the original dataset's license and redistribution permissions before making a public repository.

## Methods

1. **Preprocessing:** Encode categorical features; check missing values; separate input features and target.
2. **Train/test split:** Stratified 80/20 split with `random_state=42` (36,000 train / 9,000 test).
3. **Scaling:** Standardize features for neural-network training, fitting the scaler only on training data.
4. **Models:** MLP with dropout; ANN with early stopping; Random Forest with 1,000 trees as specified in the notebook.
5. **Evaluation:** Classification reports, confusion matrices, and comparative test accuracy.
6. **Explainability:** Random Forest impurity-based feature importance and SHAP global/local visualizations.

> The notebook calls the first dense network a neural-network model; despite some earlier CNN-style terminology, it uses fully connected Dense layers, **not convolutional layers**.

## Results reported in the manuscript

| Model | Accuracy | Precision (approved class) | Recall (approved class) | F1 (approved class) |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | **92.77%** | **0.89** | **0.77** | **0.83** |
| Multilayer Perceptron | 91.76% | 0.86 | 0.75 | 0.80 |
| Deep ANN (early stopping) | 91.44% | 0.86 | 0.73 | 0.79 |

These are **reported manuscript results**, not results independently reproduced during repository packaging. Accuracy alone is insufficient on an imbalanced dataset; review per-class metrics and confusion matrices.

### Explainable AI (XAI)

The notebook includes code for SHAP feature-attribution calculations, feature-importance tables, summary plots, and a local waterfall explanation. The manuscript describes SHAP as an extension and **does not report validated numerical SHAP results**. Run the notebook to generate and inspect the actual SHAP output. Feature attributions explain model behavior; they do not establish causality.

## Repository structure

```text
loan_approval_xai_github/
├── data/
│   └── loan_data.csv
├── notebooks/
│   └── loan_approval_prediction.ipynb
├── paper/
│   └── Loan_Approval_XAI_Enhanced_Manuscript.pdf
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and usage

**Requirements:** Python 3.10+ (a compatible TensorFlow-supported Python version is recommended), pip, and sufficient memory for a 1,000-tree Random Forest plus SHAP computations.

```bash
# Clone your repository (replace YOUR_USERNAME after publishing)
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction-xai.git
cd loan-approval-prediction-xai

# Create a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Open `notebooks/loan_approval_prediction.ipynb` in JupyterLab. **Run Jupyter from the repository root**; the notebook reads `data/loan_data.csv` relative to that directory. Run cells in order. Training and SHAP may take time. If your environment runs the notebook from the `notebooks/` folder instead, update the CSV path to `../data/loan_data.csv`.

The notebook has been adapted from a Kaggle-specific data path to the included local CSV and its saved outputs cleared for a clean rerun. It has **not** been fully executed or validated in this packaging environment.

## Limitations and responsible use

- This is an **academic research prototype**, not a production-ready lending decision system.
- Class imbalance can make accuracy misleading; compare precision, recall, F1, and false approval/rejection rates.
- The dataset includes sensitive demographic attributes. Fairness, bias, calibration, and independent validation are needed before real-world use.
- Label encoding of nominal categories may introduce arbitrary ordering; consider one-hot encoding or more suitable categorical handling in future work.
- SHAP explanations are model attributions, not causal justifications or guarantees of fairness.
- Reproducibility can depend on package versions, random seeds, and hardware.

## Future work

Explore calibrated probabilities, class imbalance handling, boosted trees, hyperparameter tuning, external validation, fairness auditing, and expanded SHAP analysis.

## Citation

If referencing this project, cite the accompanying manuscript:

> R. Hossain, M. H. Tanvir, and W. Ahmed, *Machine Learning and Deep Learning Approaches for Loan Approval Prediction: A Comparative Study*, East West University, research manuscript.

## License and attribution

No software license has been assigned in this package. Before adding an open-source license, confirm agreement among the project contributors and verify the dataset's reuse terms. The linked Kaggle dataset and any third-party materials retain their respective owners' rights.
