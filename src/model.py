import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import optuna

STATE = 514
SEED = 654

# helper func -- get best prob threshold that maximizes accuracy in validation set
# we can benefit from tuning the threshold prob since our dataset is mildly imbalanced
def best_threshold(y_true, y_prob):
    """
    pick a probability threshold maximizing accuracy
    """
    thresholds = np.linspace(0.01, 0.99, 99)        # threshold grid
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        s = accuracy_score(y_true, pred)
        if s > best_s:
            best_s, best_t = s, t                   # choose the best
    return best_t


# given prob threshold, calculate the performance metrics
def evaluate(y_true, y_prob, thr):
    """
    return accuracy, F1, ROC-AUC, PR-AUC, confusion matrix counts
    """
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


# core training
def train_xgb(
    X_train,
    y_train,
    n_splits=5,
    inner_val_fraction=0.15,         # maybe 0.15 ? -- .1 since we don't have much data
    early_stopping_rounds=200,      # patience
    use_scale_pos_weight=True,      # I want to able to toggle this since imbalance is not that severe (maybe better performance when False?)
    random_state=STATE,             # TODO : make it a global and not hard-coded (used it so often)
    n_estimators=2000,              # # of boosting rounds
    learning_rate=0.02,             # TODO : maybe dynamic ?
    max_depth=4,                    # TODO : play around this
    subsample=0.9,                  # adjust data point bagging style TODO : can introduce over/under fitting?
    colsample_bytree=0.9,           # adjust feature bagging ratio, TODO : maybe 0.95? we have too little # of features
    reg_lambda=1.0,                 # ridge penalty coeff
    reg_alpha=0.0,                  # lasso penalty coeff TODO : data does not seem to be noisy but maybe reduce correlation? (try 0.05-0.1)
):
    """
    core training idea : stratified k-fold cross validation with inner early stopping and threshold tuning
    returns a dictionary with OOF predictions, metrics, and final model
    """
    X = X_train.copy()
    y = np.asarray(y_train).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prob = np.zeros(len(X))             # stores predicted probabilities for every training row, but only when that row is in a validation fold
    fold_rows = []                          # accumulates per-fold metrics for nice tables later on
    fold_thresholds = []                    # collects best decision threshold of each fold

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        # plain train-validation split
        X_tr_full, y_tr_full = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        # inner split for early stopping
        # use early stopping without touching the validation fold (avoids leakage / optimistic CV)
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_tr_full, y_tr_full,
            test_size=inner_val_fraction,
            stratify=y_tr_full,                 # keep class imbalance while extracting an inner split
            random_state=random_state + fold,   # random_state + fold makes the early stopping split change per fold but stay reproducible
        )

        # Handle imbalance
        spw = None
        if use_scale_pos_weight:                # calculate pos and neg per fold for more accurate weights (small wiggles across folds)
            pos = max(1, np.sum(y_tr == 1))
            neg = np.sum(y_tr == 0)
            alpha = .75
            spw = 1. + alpha * (neg / pos - 1.) if pos > 0 else 1.0

        # XGB
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            tree_method = "hist",                   # take a deeper look at this
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="binary:logistic",            # maybe look at alternatives
            eval_metric="logloss",                  # maybe look at alternatives
            random_state=random_state,
            scale_pos_weight=spw if spw is not None else 1.0,
        )

        # fit the fold model
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False            
        )

        va_prob = model.predict_proba(X_va)[:, 1]   # get probabilities for class 1 (posve threshold) on val set 
        oof_prob[va_idx] = va_prob                  # hold all prob preds from each fold in this array

        thr = best_threshold(y_va, va_prob)         # tune the decision threshold for this fold (max accuracy)
        fold_thresholds.append(thr)

        m = evaluate(y_va, va_prob, thr)            # compute metrics at the chosen threshold and get save the best_iteration
        # m.update({"fold": fold, "threshold": thr, "best_iteration": model.get_booster().best_iteration_})
        fold_rows.append(m)


    # post cv -- let's deploy final model
    avg_thr = np.mean(fold_thresholds)              # average the best thresholds to get the single operating point for the final model
    print(f"Prob thresholds of each fold {fold_thresholds}")
    oof_metrics = evaluate(y, oof_prob, avg_thr)

    # final model training on ALL data -- no val set atp in order to make model learn from larger data than each fold comapared to cv above
    # we still have es set for early stopping
    X_tr_all, X_es_all, y_tr_all, y_es_all = train_test_split(
        X, y,
        test_size=inner_val_fraction,
        stratify=y,
        random_state=random_state + SEED,            # still random but reproducable
    )

    spw_all = None                                  # TODO : repetitive I didnt like this
    if use_scale_pos_weight:
        pos = max(1, np.sum(y_tr_all == 1))
        neg = np.sum(y_tr_all == 0)
        spw_all = neg / pos if pos > 0 else 1.0


    # define final model on whole* data
    final_model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        tree_method = "hist",
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=spw_all if spw_all is not None else 1.0,
        early_stopping_rounds=early_stopping_rounds,
    )

    # fit model on all data
    final_model.fit(
        X_tr_all, y_tr_all,
        eval_set=[(X_es_all, y_es_all)],            # check for eval_set size -- 0.1 is too little
        verbose=False
    )

    # TODO : can add more to results for debugging -- instead of printing above
    results = {
        "oof_prob": oof_prob,
        "fold_metrics": pd.DataFrame(fold_rows),
        "avg_threshold": avg_thr,
        "oof_metrics": oof_metrics,
        "final_model": final_model,
    }
    
    return results



# bayesian optimization (tpe) -- optuna
def tune_optuna(
    X,
    y,
    n_trials=40,
    n_splits=5,
    inner_val_fraction=0.15,
    early_stopping_rounds=200,
    use_scale_pos_weight=True,
    random_state=STATE,
    sampler_seed=SEED,
):
    """
    bayesian hyperparameter search for xgb using optuna
    maximizes mean cv Accuracy with per-fold threshold tuning
    returns -- (best_params, study)
    """

    X = X.copy()
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        # define parameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        }

        scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr_full, y_tr_full = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            # inner es split
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_tr_full, y_tr_full,
                test_size=inner_val_fraction,
                stratify=y_tr_full,
                random_state=random_state + fold,
            )

            # class imbalance handling as always (check for avoiding code repetition)
            spw = 1.0
            if use_scale_pos_weight:
                pos = max(1, np.sum(y_tr == 1))
                neg = np.sum(y_tr == 0)
                alpha = .75
                spw = 1. + alpha * (neg / pos - 1.) if pos > 0 else 1.0

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                tree_method="hist",
                scale_pos_weight=spw,
                early_stopping_rounds=early_stopping_rounds,
                **params,
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                verbose=False
            )

            va_prob = model.predict_proba(X_va)[:, 1]
            thr = best_threshold(y_va, va_prob)  # maximize accuracy on val fold
            score = accuracy_score(y_va, (va_prob >= thr).astype(int))
            scores.append(float(score))

            # pruning for faster search
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        study_name="xgb_optuna_accuracy",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)   # TODO : maybe enable pbar in notebook?

    best = study.best_params
    return best, study
