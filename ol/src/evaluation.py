import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, validation_curve

def evaluate_test_set(pipeline, X_test, y_test):
    print(f"Evaluating '{type(pipeline.named_steps['estimator']).__name__}' on test and calculating metrics.")
    y_pred = pipeline.predict(X_test)
    metrics = {}
    estimator = pipeline.named_steps['estimator']
    if isinstance(estimator, ClassifierMixin):
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        if y_proba is not None:
            n_classes = y_proba.shape[1]
            average_type = 'binary' if n_classes == 2 else 'weighted' 
            metrics['average_precision'] = average_precision_score(y_test, y_proba[:, 1] if n_classes == 2 else y_proba, average=average_type if n_classes > 2 else None) # type: ignore
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1] if n_classes == 2 else y_proba, multi_class='ovr' if n_classes > 2 else 'raise', average=average_type if n_classes > 2 else None) # type: ignore
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average="weighted")
        metrics['ConfusionMatrix'] = confusion_matrix(y_test, y_pred)
    else:
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
    return metrics, y_pred

def compute_learning_curve(pipeline, X_train, y_train, scoring, seed):
    print("Computing learning curve.")
    train_sizes, train_scores, val_scores = learning_curve( # type: ignore
        pipeline, 
        X_train, y_train, 
        cv=3, 
        n_jobs=-1, 
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), 
        random_state=seed
    )
    return train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)

def compute_complexity_curve(pipeline, X_train, y_train, param_name, param_values=None, n_splits=5, seed=1):
    if param_values is None:
        param_defaults = {
            "max_depth": [3, 6, 10, 15, 20],  # For DT
            "ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],  # For DT pruning
            "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],  # For LSVM
            "C": [0.1, 0.5, 1, 2, 10],  # For SVM
            "n_neighbors": [1, 3, 5, 11, 21, 31],  # For kNN
            "learning_rate_init": [0.0001, 0.001, 0.01, 0.1]  # For NN
        }
        param_values = param_defaults.get(param_name, [0.1, 1, 10])
    
    estimator = pipeline.named_steps['estimator']
    if isinstance(estimator, ClassifierMixin):
        scoring = 'f1_weighted'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        scoring = 'r2'
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_scores, val_scores = validation_curve(
        pipeline, X_train, y_train,
        param_name=param_name,
        param_range=param_values,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    return {
        'param_name': param_name,
        'param_values': param_values,
        'train_scores_mean': train_scores.mean(axis=1),
        'val_scores_mean': val_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'val_scores_std': val_scores.std(axis=1)
    }