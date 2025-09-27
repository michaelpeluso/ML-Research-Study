from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CountEncoder, TargetEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.utils.tuning_config import DEFAULT_HYPERPARAMETERS


ESTIMATOR_MAP = {
    "dt": (DecisionTreeClassifier, DecisionTreeRegressor),
    "lsvm": (SGDClassifier, SGDRegressor),
    "svm": (SVC, SVR), 
    "knn": (KNeighborsClassifier, KNeighborsRegressor),
    "nn": (MLPClassifier, MLPRegressor)
}

def build_pipeline(transformer: ColumnTransformer, model: str, method: str, seed=1, **kwargs):
    print(f"Adding {method} {model.upper()} estimator to pipeline.")

    model = model.lower()
    method = method.lower()[0]
    if method not in ["c", "r"]: 
        raise ValueError(f"Unknown method '{method}'")
    
    hyperparams = DEFAULT_HYPERPARAMETERS.get((model, method), {})
    hyperparams.update(kwargs)
    estimator_cls = ESTIMATOR_MAP[model][0 if method == "c" else 1]
    
    # control random seed
    if estimator_cls == SVR or model == "knn": extra_params = {}
    else:extra_params = {"random_state": seed} if model != "knn" else {"random_state": seed}

    estimator = estimator_cls(**hyperparams, **extra_params)
    
    return Pipeline([
        ("preprocess", transformer), 
        ("estimator", estimator)
    ])


def build_column_transformer(numeric_cols: list[str], categorical_cols: list[str], high_card_cols: list[str] = None) -> ColumnTransformer:
    print("Building column transformer with numeric and low/high cardinality columns.")

    num_pipeline = Pipeline([
        # Univariate imputer for completing missing values with simple strategies.
        # Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
        ("imputer", SimpleImputer(strategy="median")), 
        # A feature scaling technique which follows Standard Normal Distribution (SND) and is used to standardize the values of numeric features. 
        # It transforms data so that the mean becomes 0 and the standard deviation becomes 1.
        # https://www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    cat_pipeline = Pipeline([ # low/medium cardinality
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Also known as mean encoding, involves replacing categorical values with the mean of the target variable for each category.
        # https://www.geeksforgeeks.org/machine-learning/target-encoding-using-nested-cv-in-sklearn-pipeline/
        ("encoder", TargetEncoder()),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    freq_pipeline = Pipeline([ # high cardinality
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        # Count encoding for categorical features. For a given categorical feature, replace the names of the groups with the group counts.
        # https://contrib.scikit-learn.org/category_encoders/count.html
        ("encoder", CountEncoder()),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])

    transformers = [("num", num_pipeline, numeric_cols)]
    if high_card_cols:
        low_med_card_cols = [c for c in categorical_cols if c not in high_card_cols]
        if low_med_card_cols:
            transformers.append(("cat", cat_pipeline, low_med_card_cols))
        transformers.append(("freq", freq_pipeline, high_card_cols))
    else:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")
