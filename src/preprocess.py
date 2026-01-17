import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def make_mice_bayes(max_iter=15, random_state=1):
    """
    simple bayesian MICE imputer
    """
    return IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="median",
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=False
    )

def fit_mice_bayes_imputer(X_train, *, max_iter=15, random_state=1):
    imputer = make_mice_bayes(max_iter=max_iter, random_state=random_state)
    imputer.fit(X_train)          # fit on TRAIN ONLY (no leakage)
    return imputer

def transform_with_imputer(imputer, X: pd.DataFrame) -> pd.DataFrame:
    arr = imputer.transform(X)
    out = pd.DataFrame(arr, index=X.index, columns=X.columns)
    if out.isna().any().any():
        raise RuntimeError("Imputation produced NaNs.")     # crucial for debugging
    return out

def summarize_differences(X_before, X_after, rtol=1e-10, atol=1e-12):
    """
    print every (row, column) where values differ between the two DataFrames
    less compact version of asset_no_overwrite, I implemented this func since I want to see the changes
    one by one with my own eyes :) -- since bad imputation can destroy every task I build on top of it
    """
    before_vals, after_vals = X_before.values, X_after.values
    nan_diff = np.isnan(before_vals) ^ np.isnan(after_vals)
    val_diff = ~np.isclose(before_vals, after_vals, rtol=rtol, atol=atol, equal_nan=True)
    diff_mask = nan_diff | val_diff

    total = int(diff_mask.sum())
    print(f"Total differing entries: {total}")

    rows, cols = np.where(diff_mask)
    for r, c in zip(rows, cols):
        print(f"({X_before.index[r]}, {X_before.columns[c]}) : {X_before.iat[r,c]} -> {X_after.iat[r,c]}")


def assert_no_overwrite(X_before: pd.DataFrame, X_after: pd.DataFrame, rtol=1e-10, atol=1e-12):
    """
    assert that any originally observed (non-NaN) value is unchanged after imputation.
    personally I find this func pretty crucial to ensure we are not overwriting existing signal/info
    """
    if X_before.shape != X_after.shape:
        raise AssertionError("Shapes differ between before/after DataFrames")

    observed = ~X_before.isna()
    # compare only where observed is True
    a = X_before.values[observed.values]
    b = X_after.values[observed.values]
    same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    if not np.all(same):
        # find and print first few offenders for debugging
        rows, cols = np.where((observed.values) & 
                              (~np.isclose(X_before.values, X_after.values, rtol=rtol, atol=atol, equal_nan=False)))
        print("Warning: some originally observed values changed. Examples:")
        for r, c in list(zip(rows, cols))[:10]:
            print(f"({X_before.index[r]}, {X_before.columns[c]}) : {X_before.iat[r,c]} -> {X_after.iat[r,c]}")
        raise AssertionError("Imputer modified existing non-NaN values.")
    else:
        print("passed -- no overwrite")

