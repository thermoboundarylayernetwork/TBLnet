from scaler_manager import ScalerManager
import pandas as pd
import torch
import time

def _ensure_time_column(df):
    """
    Ensure dataframe has a 'time' column as seconds since the first timestamp.
    Accepts original CSVs that have 'time' or 'date' column.
    """
    if 'time' in df.columns:
        # Parse to datetime if not already numeric
        try:
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()
        except Exception:
            # If already numeric, leave as-is
            pass
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
        df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()
    else:
        raise KeyError("Input CSV must contain either a 'time' or 'date' column. Found columns: "
                       + ", ".join(df.columns.tolist()))
    return df

def load_csv_data(file_path, device='cpu'):
    """
    Load data from CSV, initialize and fit ScalerManager, normalize features.
    Returns normalized features, targets, scaler manager, and dataframe.
    """
    df = pd.read_csv(file_path)

    # Ensure time column in seconds
    df = _ensure_time_column(df)

    scaler_mgr = ScalerManager()
    scaler_mgr.fit(df)

    features_norm = scaler_mgr.transform_all(df)
    inputs = torch.tensor(features_norm, dtype=torch.float32).to(device)

    # Extract target velocities (uo, vo)
    targets = df[['uo', 'vo']].values
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # Split normalized features
    t_norm = inputs[:, 0:1]
    x_norm = inputs[:, 1:2]
    y_norm = inputs[:, 2:3]
    z_norm = inputs[:, 3:4]
    u_true = targets[:, 0:1]
    v_true = targets[:, 1:2]

    return t_norm, x_norm, y_norm, z_norm, u_true, v_true, scaler_mgr, df

def split_dataset_random(csv_path, train_ratio=0.8, seed=None):
    """
    Randomly shuffle and split a CSV dataset into train/test by ratio.
    """
    if seed is None:
        seed = int(time.time() * 1000 % 2 ** 32)

    df = pd.read_csv(csv_path)

    # Ensure time column in seconds
    df = _ensure_time_column(df)

    # Shuffle dataset
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split by ratio
    split_index = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled.iloc[:split_index].copy()
    test_df = df_shuffled.iloc[split_index:].copy()

    print(f"[INFO] Random split: train samples = {len(train_df)}, test samples = {len(test_df)}")
    return train_df, test_df

def load_csv_data_from_df(df, device='cpu', scaler_mgr=None, fit_scaler=True):
    """
    Normalize a dataframe using ScalerManager.
    
    Args:
        df: DataFrame of training or test data
        device: torch device string or object
        scaler_mgr: Pass None and set fit_scaler=True to fit a new scaler,
                    or pass an existing scaler_mgr and set fit_scaler=False to reuse it.
        fit_scaler: If True, fit scaler_mgr on df; if False, use existing scaler_mgr.
    Returns:
        t_norm, x_norm, y_norm, z_norm, u_true, v_true, scaler_mgr, df
    """
    df = _ensure_time_column(df)

    if fit_scaler:
        scaler_mgr = ScalerManager()
        scaler_mgr.fit(df)
    else:
        if scaler_mgr is None:
            raise ValueError("scaler_mgr must be provided when fit_scaler=False (for test/validation data).")

    features_norm = scaler_mgr.transform_all(df)
    inputs = torch.tensor(features_norm, dtype=torch.float32).to(device)

    targets = df[['uo', 'vo']].values
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    t_norm = inputs[:, 0:1]
    x_norm = inputs[:, 1:2]
    y_norm = inputs[:, 2:3]
    z_norm = inputs[:, 3:4]
    u_true = targets[:, 0:1]
    v_true = targets[:, 1:2]

    return t_norm, x_norm, y_norm, z_norm, u_true, v_true, scaler_mgr, df