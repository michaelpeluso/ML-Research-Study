import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# load data from cache if available
def load_or_process_data(dataset: str, target: str, method: str, subsample: float, seed: int, cache_dir="cache"):
    print(f"Loading {dataset} data.")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset}_subsample_{int(subsample*100)}.pkl")

    if os.path.exists(cache_file):
        X_train, X_test, y_train, y_test = joblib.load(cache_file)
        total_rows = "Used cached clean data"
        print(f"Loaded cached dataset from {cache_file}")
    else:
        X_train, X_test, y_train, y_test, total_rows = get_data(dataset, target, method, subsample, seed)
        joblib.dump((X_train, X_test, y_train, y_test), cache_file)
        print(f"Saved processed dataset to {cache_file}")

    info = { # data for logging
        'n_rows_initial': total_rows, 
        'n_rows_cleaned': len(X_train) + len(X_test),
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'target_distribution': y_train.value_counts().to_dict() if 'classification' in method.lower() else None,
        'memory_usage_mb': (X_train.memory_usage(deep=True).sum() + X_test.memory_usage(deep=True).sum() +
                            y_train.memory_usage(deep=True) + y_test.memory_usage(deep=True)) / (1024 ** 2)
    }

    return X_train, X_test, y_train, y_test, info


# main get data function
def get_data(dataset: str, target: str, method: str, subsample_frac: float, seed: int):
    df = load_raw_df(dataset)
    total_rows = len(df)
    
    # Subsample & clean
    subsampled_df = subsample_dataset(df, target, subsample_frac, seed, method)
    cleaned_df = clean_data(subsampled_df, target)

    # after cleaning
    cleaned_df['Duration_Seconds'] = np.log1p(cleaned_df['Duration_Seconds'])  # log(1 + x) to handle zeros/small values

    # Split
    X_train, X_test, y_train, y_test = split_dataset(cleaned_df, target, method, test_size=0.2)

    # confirm converted to seconds
    y_train = y_train.dt.total_seconds().astype("float32") if pd.api.types.is_timedelta64_dtype(y_train) else y_train
    y_test = y_test.dt.total_seconds().astype("float32") if pd.api.types.is_timedelta64_dtype(y_test) else y_test

    return X_train, X_test, y_train, y_test, total_rows


# user selected dataset with specific drops
def load_raw_df(dataset:str):
    df = None

    # dataset-specific drops
    if dataset == "hotels":
        df = pd.read_csv("data/hotel_bookings.csv")
        df = df.drop(columns=[
            "reservation_status", # given
            "reservation_status_date" # given
            ], errors="ignore")
    
    elif dataset == "accidents":
        df = pd.read_csv("data/US_Accidents_March23_1M_rows.csv")

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
        df = df[df['Duration_Seconds'] > 0]
        df = df[(df['Duration_Seconds'] > 0) & (df['Duration_Seconds'] <= df['Duration_Seconds'].quantile(0.95))]
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
    else :
        print("Missing dataset name: 'accidents' or 'hotels'")
        raise ValueError("Missing dataset name: 'accidents' or 'hotels'")
    
    return df


def clean_data(df: pd.DataFrame, target: str):
    print(f"Cleaning data.\nRaw rows: {len(df)}")
    mem_before = df.memory_usage(deep=True).sum() / (1024 ** 2)

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
    
    mem_after = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Cleaned Rows: {len(df)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB ({100 * (mem_before - mem_after) / mem_before:.1f}% reduction)")
    print(f"Dtypes: \n{df.dtypes}")

    return df


def split_dataset(df: pd.DataFrame, target: str, method:str, test_size=0.2, seed=1):
    print("Generating test/train/validation data splits.")
    # test/train/val split
    X = df.drop(columns=[target])
    y = df[target]

    # even class proportion between test/train sets
    stratify = y if method == "classification" else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=seed,
    )

    return X_train_val, X_test, y_train_val, y_test
    

# cut the data into a smaller sample
def subsample_dataset(df: pd.DataFrame, target: str, subsample_frac: float, seed: int, method: str):
    if subsample_frac < 1.0:
        if 'regression' in method.lower():
            # random sample for regression
            df = df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True)
        else:
            df = df.groupby(target, group_keys=False).apply(
                lambda x: x.sample(frac=subsample_frac, random_state=seed)
            ).reset_index(drop=True)
    return df

# convert value types
def get_col_types(df: pd.DataFrame, arg: str):
    
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    high_card_cols = None

    if arg == "hotels":
        high_card_cols = [c for c in [
        "country", 
        "agent", 
        "company"
        ] if c in df.columns]
    
    elif arg == "accidents":
        high_card_cols = [c for c in [
        "Street", 
        "County", 
        "City", 
        "Zipcode", 
        "Airport_Code",
        # "Wind_Direction", # not enough variation
        # "Weather_Condition", # not enough variation
        ] if c in df.columns]
    
    return numeric_cols, categorical_cols, high_card_cols

'''
def scale_target(y, scaler=None, apply_log=False):
    y_values = y.values
    if apply_log:
        y_values = np.log1p(y_values)
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(y_values.reshape(-1, 1)).ravel()
    else:
        scaled = scaler.transform(y_values.reshape(-1, 1)).ravel()
    return scaled, scaler

def unscale_target(y_pred_scaled, y_scaler, apply_log=False):
    unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    if apply_log:
        unscaled = np.expm1(unscaled)
    return unscaled
'''