# Ultramarin Coding Challenge (Can Uzduran)

This repository contains a reproducible **machine learning pipeline** for a binary classification task on macro/market features.  
All experiments are done **via Jupyter notebooks** — the Python scripts in `src/` provide reusable utilities for imputation and model training.

---

## 📁 Project Structure

```
project_root/
├── data/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   ├── X_train_imp.csv        # created by preprocess notebook
│   ├── X_test_imp.csv         # created by preprocess notebook
│   └── y_test.csv             # created by model notebook
├── notebooks/
│   ├── eda.ipynb              # optional exploratory data analysis
│   ├── preprocess.ipynb       # imputation → *_imp.csv
│   └── xgb.ipynb              # model training, CV, tuning, predictions
├── src/
│   ├── util_io.py             # CSV loading utilities
│   ├── preprocess.py          # imputation logic
│   └── model.py               # XGBoost + Optuna functions
├── environment.yml            # main conda environment
├── requirements.txt           # optional pip fallback
```

---

## ⚙️ 1. Environment Setup

> ✅ Recommended: use **Conda** and the provided `environment.yml`.

```bash
# create environment
conda env create -f environment.yml
conda activate ultramarin-canuz

# make environment visible in Jupyter
python -m ipykernel install --user --name ultramarin-canuz --display-name "ultramarin-canuz"
```

Then start JupyterLab:

```bash
jupyter lab
```

If you update dependencies later:
```bash
conda env update -f environment.yml --prune
```

---

## 📊 2. Input Data

Before running the notebooks, ensure the following files exist in the `data/` directory:

| File | Description |
|------|--------------|
| `X_train.csv` | Training features |
| `y_train.csv` | Binary labels (0/1) |
| `X_test.csv` | Test features |

The helper function in `src/util_io.py` reads them using `index_col=0`, ensuring indices remain aligned.

---

## 📓 3. Notebook Workflow

Run notebooks **in this order**:

### 🧹 A) `notebooks/preprocess.ipynb`
- Loads raw CSVs (`X_train`, `y_train`, `X_test`)
- Runs imputation using functions from `src/preprocess.py`
- Writes clean datasets to `data/`:
  - `X_train_imp.csv`
  - `X_test_imp.csv`

After this step, the datasets contain **no NaN values**.

---

### 🚀 B) `notebooks/xgb.ipynb`
- Loads `X_train_imp`, `y_train`, and `X_test_imp`
- Trains an **XGBoost binary classifier** with:
  - Stratified K-Fold CV  
  - Inner early stopping (leakage-free)  
  - Per-fold threshold tuning (max accuracy)  
  - Optional **Optuna** Bayesian optimization
- Produces:
  - Out-of-fold (OOF) metrics + plots (ROC, PR)
  - Final tuned model
  - Predictions on test data:
    - `data/y_test.csv` (labels)
    - optionally `data/y_proba.csv` (probabilities)

Also includes:
- Confusion matrix visualization  
- Feature mean comparison between train/test  
- Feature importance plots

---

### 📈 (Optional) `notebooks/eda.ipynb`
Exploratory data analysis (missingness, distributions, correlations).  
No outputs are saved from this notebook.

---

## ⚙️ 4. Configurable Parameters (in `xgb.ipynb`)

| Parameter | Purpose | Example |
|------------|----------|----------|
| `n_splits` | Number of K-Folds | 5 |
| `inner_val_fraction` | Fraction used for early stopping | 0.1–0.15 |
| `early_stopping_rounds` | Early stopping patience | 200 |
| `use_scale_pos_weight` | Toggle class imbalance handling | True / False |
| `n_estimators`, `learning_rate`, `max_depth`, ... | XGBoost hyperparameters | – |
| `run_tuning`, `n_trials` | Optuna tuning control | – |

---

## 💾 5. Outputs

After training, results are written to:

```
data/
├── X_train_imp.csv
├── X_test_imp.csv
├── y_test.csv -- (main output of the model : labels from test data)
```

---

## 🔁 6. Reproducibility

- Random seeds are fixed for CV, splits, and model RNG.  
- Imputation and model training are deterministic for the same seed.  
- The environment.yml ensures consistent library versions.

To fully reproduce:
1. Use the provided environment.
2. Keep seeds unchanged.
3. Run notebooks in the same sequence.

---


## ✅ 7. Deliverables

- `data/y_test.csv` (final predictions)
- Optionally: plots and metrics from `notebooks/xgb.ipynb`

---