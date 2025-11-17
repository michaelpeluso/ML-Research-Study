import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import CountEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.logger import MLLogger, print_t as print

# load data from cache if available - returns FULL dataset for unsupervised learning
def load_or_process_data(dataset: str, target: str, method: str, subsample: float, seed: int, cache_dir="", ml_logger: MLLogger|None=None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Load and process the full dataset"""
    if not ml_logger: ml_logger = MLLogger()
    with ml_logger.log_step("Load Data") as step_info:
        if not cache_dir:
            cache_dir = os.path.join(os.environ['ROOT'], "cache")
        hotels_path = os.path.join(os.environ['ROOT'], f"data/hotel_bookings.csv")
        accidents_path = os.path.join(os.environ['ROOT'], f"data/US_Accidents_March23_2M_rows.csv")

        print(f"Loading {dataset} data.")
        data_cache_dir = os.path.join(cache_dir, 'data')
        os.makedirs(data_cache_dir, exist_ok=True)
        cache_file = os.path.join(data_cache_dir, f"{dataset}_subsample_{int(subsample*100)}.pkl")
        col_log_data = {}

        if os.path.exists(cache_file):
            # load data from cache if available
            X_full, y_full = joblib.load(cache_file)
            total_cleaned = total_rows = len(X_full)

            def get_mem_mb(arr):
                return arr.memory_usage(deep=True) if hasattr(arr, "memory_usage") else arr.nbytes
            mem_before = mem_after = (get_mem_mb(X_full) + get_mem_mb(y_full)) / (1024**2)
            print(f"Loaded cached data: {total_rows:,} rows, {X_full.shape[1]} features")

        else:
            # load raw data
            df = pd.DataFrame()
            df = pd.read_csv(hotels_path if dataset == "hotels" else accidents_path)
            if len(df) == 0:
                raise ValueError(f"Dataset {dataset} is empty or not found.")
            
            total_rows = len(df)
            print(f"Raw data loaded: {total_rows:,} rows")
            
            # subsample
            df = subsample_dataset(df, target, subsample, seed, method)
            after_subsample = len(df)
            print(f"After subsampling ({subsample*100:.1f}%): {after_subsample:,} rows")

            # clean
            print(f"Cleaning {dataset} data.")
            mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)
            cleaned_df = clean_hotels(df) if dataset == "hotels" else clean_accidents(df)
            cleaned_df = general_clean(cleaned_df, target)
            # Winsorize numeric columns to cap extreme outliers (default 1%/99% bounds)
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:cleaned_df = winsorize_df(cleaned_df, numeric_cols, lower=0.01, upper=0.99)
            total_cleaned = len(cleaned_df)
            if method == "regression":
                cleaned_df['Duration_Seconds'] = np.log1p(cleaned_df['Duration_Seconds']) # log(1 + x) to handle zeros/small values
            mem_after = cleaned_df.memory_usage(deep=True).sum() / (1024 ** 2)
            
            print(f"After cleaning: {total_cleaned:,} rows, {cleaned_df.shape[1]} columns")
            print(f"Memory usage: {mem_before:.1f} MB → {mem_after:.1f} MB ({(mem_before-mem_after)/mem_before*100:.1f}% reduction)")

            # Separate X and y
            X = cleaned_df.drop(columns=[target])
            y = cleaned_df[target]
            
            # Show target distribution
            if method == "classification":
                target_counts = y.value_counts().sort_index()
                print(f"Target distribution ({target}):")
                for val, count in target_counts.items():
                    pct = count / len(y) * 100
                    print(f"  {val}: {count:,} ({pct:.1f}%)")
            else:
                print(f"Target statistics ({target}):")
                print(f"  Mean: {y.mean():.3f}, Std: {y.std():.3f}, Min: {y.min():.3f}, Max: {y.max():.3f}")
            
            # Transform columns using the full dataset
            X_full, col_log_data = transform_full_dataset(dataset, X, y)
            y_full = y.values if hasattr(y, 'values') else y
            
            print(f"Final processed data: {X_full.shape[0]:,} samples × {X_full.shape[1]} features")
            
            # cache data
            joblib.dump((X_full, y_full), cache_file)
            print(f"Saved processed full dataset to {cache_file}")

        step_info = { # data for logging
            'used_cached_df' : os.path.exists(cache_file),
            'n_loaded_rows': total_rows, 
            'n_cleaned_rows': total_cleaned,
            'full_dataset_shape': f"{X_full.shape[0]} samples x {X_full.shape[1]} features",
            'target_distribution': pd.Series(y_full).value_counts().to_dict() if 'classification' in method.lower() else None,
            'memory_before_clean': "Used cached memory" if mem_before==mem_after else f"{mem_before:.2f} MB",
            'memory_after_clean': f"{mem_after:.2f} MB",
            'memory_reduction': f"{round((mem_before - mem_after) / mem_before * 100, 2)}%" if mem_before!=mem_after else "0%",
        }
        step_info.update(col_log_data)

    return X_full, y_full # type:ignore


def split_processed_data(X: np.ndarray, y: np.ndarray, method: str, test_size: float = 0.2, val_size: float = 0.2, 
                         seed: int = 42, ml_logger: MLLogger|None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split preprocessed data into train/val/test sets"""
    if not ml_logger: ml_logger = MLLogger()
    with ml_logger.log_step("Split Data") as step_info:
        print(f"Splitting data (test={test_size*100:.1f}%, val={val_size*100:.1f}%).")
        
        stratify = y if method == "classification" else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        
        # Second split: separate train and validation
        stratify_temp = y_temp if method == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=seed, stratify=stratify_temp
        )
        
        # Print detailed split information
        total_samples = len(X)
        train_pct = len(X_train) / total_samples * 100
        val_pct = len(X_val) / total_samples * 100
        test_pct = len(X_test) / total_samples * 100
        
        print(f"Data splits:")
        print(f"  Train: {len(X_train):,} samples ({train_pct:.1f}%) - {X_train.shape[1]} features")
        print(f"  Validation: {len(X_val):,} samples ({val_pct:.1f}%) - {X_val.shape[1]} features")
        print(f"  Test: {len(X_test):,} samples ({test_pct:.1f}%) - {X_test.shape[1]} features")
        
        if method == "classification":
            print(f"Stratified sampling: Yes")
            print(f"Train class distribution:")
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            for cls, count in zip(train_classes, train_counts):
                pct = count / len(y_train) * 100
                print(f"  Class {cls}: {count:,} ({pct:.1f}%)")
        else:
            print(f"Stratified sampling: No (regression)")
        
        step_info = {
            'train_shape': f"{X_train.shape[0]} samples x {X_train.shape[1]} features",
            'validation_shape': f"{X_val.shape[0]} samples x {X_val.shape[1]} features",
            'test_shape': f"{X_test.shape[0]} samples x {X_test.shape[1]} features",
            'split_pct': {
                'train': f"{len(X_train)/len(X)*100:.1f}%",
                'val': f"{len(X_val)/len(X)*100:.1f}%",
                'test': f"{len(X_test)/len(X)*100:.1f}%"
            },
            'stratified': method == "classification",
        }
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def wrap_into_loaders(method, X_train, X_val, X_test, y_train, y_val, y_test, batch_size) -> tuple[DataLoader, DataLoader, DataLoader]:
    # https://edstem.org/us/courses/81923/discussion/6999408
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_dtype = torch.float if 'regression' in method.lower() else torch.long
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=y_dtype, device=device)
    y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=y_dtype, device=device)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=y_dtype, device=device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def transform_full_dataset(dataset: str, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, dict]:
    """Fits and transforms on the entire dataset."""
    numeric_cols = X.select_dtypes(include=["float", "int"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    high_card_cols = []

    if dataset == "hotels":
        high_card_cols = [c for c in [
        "country", 
        "agent", 
        "company"
        ] if c in X.columns]
    
    elif dataset == "accidents":
        high_card_cols = [c for c in [
        "Street", 
        "County", 
        "City", 
        "Zipcode", 
        "Airport_Code",
        # "Wind_Direction", # not enough variation
        # "Weather_Condition", # not enough variation
        ] if c in X.columns]

    # Build column transformer
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", CountEncoder()),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    freq_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("encoder", CountEncoder()),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])

    transformers = [("num", num_pipeline, numeric_cols)]
    low_med_card_cols = [c for c in categorical_cols if c not in high_card_cols] if high_card_cols else categorical_cols
    if low_med_card_cols:
        transformers.append(("cat", cat_pipeline, low_med_card_cols))
    if high_card_cols:
        transformers.append(("freq", freq_pipeline, high_card_cols))

    # Fit and transform on full dataset
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    X_transformed = preprocessor.fit_transform(X)  # No y parameter - label-free

    log_data = {
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        'high_cardinality_features': len(high_card_cols) if high_card_cols else 0,
        'encoding_strategy': {
            'numeric': 'median imputation + standard scaling',
            'low_med_cardinality': 'frequency encoding (CountEncoder) - LABEL-FREE',
            'high_cardinality': 'frequency encoding (CountEncoder) - LABEL-FREE',
            'target_encoding_used': False, 
        },
        'ul_compliance': 'LABEL-FREE for Steps 1-3 (clustering, DR)',
    }
    
    return X_transformed, log_data # type:ignore


# clean either dataset
def general_clean(df, target):
    # missing target, duplicates, leakage, and missing values
    df = df.dropna(subset=[target])
    df = df.drop_duplicates()
    df = df.dropna(axis=1, thresh=(len(df) * 0.7))

    # downcast numeric types
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # convert low-cardinality objects to category
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    
    return df


def winsorize_df(df: pd.DataFrame, cols: list[str] | None = None, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        try:
            lo = df[c].quantile(lower)
            hi = df[c].quantile(upper)
            df[c] = df[c].clip(lower=lo, upper=hi)
        except Exception:
            continue
    return df


# user selected dataset with specific drops
def clean_hotels(df):
    df = df.drop(columns=[
        "reservation_status", # given
        "reservation_status_date" # given
        ], errors="ignore")

    return df

def clean_accidents(df) :
    df['Zipcode'] = df['Zipcode'].astype(str).str[:5] # truncate to 5 digits
    
    # derive times
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce') 
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df['Start_Hour'] = df['Start_Time'].dt.hour.astype('int32')  # 0-23
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek.astype('int32')  # 0-6
    df['Start_Month'] = df['Start_Time'].dt.month.astype('int32')  # 1-12
    df['Is_Weekend'] = (df['Start_DayOfWeek'] >= 5).astype('int32')  # binary flag
    df['Is_Rush_Hour'] = df['Start_Hour'].isin([7,8,9,16,17,18]).astype('int32')  # binary flag

    # derive duration
    df['Duration_Seconds'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()
    df['Duration_Seconds'] = df['Duration_Seconds'].astype('float32')
    df = df.drop(columns=['Start_Time', 'End_Time'])

    # categorize Weather_Condition
    def categorize_weather(cond):
        if pd.isna(cond): return 0
        cond = cond.lower()
        if 'clear' in cond or 'fair' in cond or 'cloud' in cond or 'overcast' in cond: return 1  # safe
        elif 'mist' in cond or 'haze' in cond or 'drizzle' in cond: return 2  # mild
        elif 'rain' in cond or 'shower' in cond: return 3  # moderate
        elif 'snow' in cond or 'sleet' in cond: return 4  # severe
        elif 'fog' in cond or 'thunder' in cond or 'storm' in cond: return 5  # very severe
        else: return 6
    df['Weather_Category'] = df['Weather_Condition'].apply(categorize_weather).astype('int32')

    # Wind_Direction to numeric
    wind_map = {'Calm': 0, 'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
                'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5, 'VAR': 0, 'Variable': 0}
    df['Wind_Angle'] = df['Wind_Direction'].map(wind_map).fillna(0).astype('float32')
    
    df = df.drop(columns=[
        "End_Time", # given
        "Weather_Timestamp", # given 
        "Weather_Condition", # converted to numeric range
        "Wind_Direction", # converted to numeric range
        "ID", # completely unique - no info
        "Source", # doesnt offer any info unless sources have a bias
        "Description",  # completely unique - no info
        "Country", # all USA
        "Turning_Loop", # all are False
        "Start_Time", # derived
        "Bumper", # 99% are False
        "Roundabout", # 99% are False
        "Traffic_Calming", # ~100% are False
        ], errors="ignore")
    
    return df

def split_dataset(df: pd.DataFrame, target: str, method:str, test_size, val_size, seed=1):
    print("Generating test/train/validation data splits.")
    # test/train/val split
    X = df.drop(columns=[target])
    y = df[target]

    # even class proportion between test/train sets
    stratify = y if method == "classification" else None
    
    # split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=seed
    )

    # split train_val into train and val
    stratify_train_val = y_train_val if method == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=stratify_train_val, random_state=seed
    )

    return  X_train, X_val, X_test, y_train, y_val, y_test
    

# cut the data into a smaller sample
def subsample_dataset(df: pd.DataFrame, target: str, subsample_frac: float, seed: int, method: str):
    if subsample_frac < 1.0:
        if 'regression' in method.lower():
            # random sample for regression
            df = df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True)
        else:
            # stratified sampling by target for classification
            # Using sample on each group separately to avoid FutureWarning
            sampled_groups = []
            for _, group in df.groupby(target, group_keys=False):
                sampled_groups.append(group.sample(frac=subsample_frac, random_state=seed))
            df = pd.concat(sampled_groups, ignore_index=True)
    return df


def sample_fit_labels(X: np.ndarray, labels, sample_size: int | None = None, seed: int = 42):
    """ Deterministically sample up to `sample_size` rows from a numpy array X and labels. """
    if sample_size is None:
        return X, np.array(labels), None
    n = X.shape[0]
    if n <= sample_size:
        return X, np.array(labels), None
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, sample_size, replace=False)
    return X[idx], np.array(labels)[idx], idx
    