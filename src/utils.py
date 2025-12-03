import numpy as np
import pandas as pd

def log_transform_series(feature_name: str, df: pd.DataFrame) -> pd.Series:
    cap = df[feature_name].quantile(0.995)
    return np.log1p(df[feature_name].clip(upper=cap))
    
