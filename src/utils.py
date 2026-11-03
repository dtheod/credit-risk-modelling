import numpy as np
import pandas as pd

def log_transform_series(feature_name: str, df: pd.DataFrame, cap: float) -> pd.Series:
    return np.log1p(df[feature_name].clip(upper=cap))
