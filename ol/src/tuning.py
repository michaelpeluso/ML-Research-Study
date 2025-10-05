import os
import numpy as np
import pandas as pd
from typing import Union
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, make_scorer, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split

def tune_parameter_grid(dataset, pipeline, X, y, param_grid, cv_splits=5, combination_cap=50, seed=1) -> tuple[Pipeline, Union[GridSearchCV, RandomizedSearchCV]]:
    print('Tuning with parameter grid.')
    method = "classification" if isinstance(pipeline.named_steps['estimator'], ClassifierMixin) else "regression"

    # multi class or binary
    roc_auc_scorer=None
    if method == "regression": 
        n_classes = len(np.unique(y))
        if n_classes == 2: roc_auc_scorer = make_scorer(lambda y_true, y_proba: roc_auc_score(y_true, y_proba[:, 1]), needs_proba=True)
        else: roc_auc_scorer = make_scorer(lambda y_true, y_proba: roc_auc_score(y_true, y_proba, multi_class='ovr'),needs_proba=True)
    else: roc_auc_scorer = "roc_auc"

    if method == "classification": 
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "average_precision": "average_precision",
            "roc_auc": roc_auc_scorer
        }
        refit_metric = "f1_weighted"
    else:
        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        }
        refit_metric = "r2"
    
    if method == "classification": 
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    else: 
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
   
    if total_combinations > combination_cap:
        print(f"Using RandomizedSearchCV: {combination_cap} iterations out of {total_combinations} combinations")
        search = RandomizedSearchCV(pipeline, param_grid, n_iter=combination_cap, cv=cv, scoring=scoring, refit=refit_metric, verbose=3, random_state=seed, n_jobs=-1)
    else:
        print(f"Using GridSearchCV: {total_combinations} combinations")
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, refit=refit_metric, verbose=3, n_jobs=-1)

    search.fit(X, y)

    return pipeline, search

def train_nn_with_curves(pipeline, X_train, y_train, method, seed=1):
    print("Processing neural network-specific curves.")
    
    preprocessor = pipeline.named_steps['preprocess']
    estimator = pipeline.named_steps['estimator']
    
    # fit if not
    if not hasattr(preprocessor, '_is_fitted') or not preprocessor._is_fitted(): 
        preprocessor.fit(X_train, y_train)
    
    # stratify
    stratify = y_train if method == "classification" else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=stratify
    )
    
    X_tr = preprocessor.transform(X_tr)
    X_val = preprocessor.transform(X_val)
    
    # epoch handling
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    patience_counter = 0
    n_iter_no_change = estimator.n_iter_no_change
    tol = estimator.tol
    max_iter = estimator.max_iter
    classes = np.unique(y_train) if method == "classification" else None
    
    for epoch in range(max_iter):
        
        if method == "classification":
            estimator.partial_fit(X_tr, y_tr, classes=classes)
            y_tr_pred = estimator.predict_proba(X_tr)
            train_loss = log_loss(y_tr, y_tr_pred)
            y_val_pred = estimator.predict_proba(X_val)
            val_loss = log_loss(y_val, y_val_pred)
        else:
            estimator.partial_fit(X_tr, y_tr)
            y_tr_pred = estimator.predict(X_tr)
            train_loss = mean_squared_error(y_tr, y_tr_pred)
            y_val_pred = estimator.predict(X_val)
            val_loss = mean_squared_error(y_val, y_val_pred)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss - tol:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= n_iter_no_change:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    X_train_pre = preprocessor.transform(X_train)
    estimator.fit(X_train_pre, y_train)
    
    return train_losses, val_losses

def save_results(search, dataset, save_path):
    if hasattr(search, 'cv_results_'):
        results_df = pd.DataFrame(search.cv_results_)
    else:
        results_df = pd.DataFrame(columns=['mean_test_score', 'params'])
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/tuning_results_{dataset}_{search.estimator.__class__.__name__.lower()}.csv"
    results_df.to_csv(file_path, index=False)
