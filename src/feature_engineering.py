from prefect import task
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import numpy as np



@task
def out_of_time_split(df: pd.DataFrame, date_col: str, test_size: float = 0.10):
    """
    Out-of-time split: the most recent `test_size` share of rows (by `date_col`) becomes
    the test set, so the model is evaluated on loans from a later period than it was
    trained on.
    """
    cutoff = df[date_col].quantile(1 - test_size)
    train = df[df[date_col] < cutoff]
    test = df[df[date_col] >= cutoff]
    print(f"Out-of-time split on {date_col} at {cutoff.date()}: {len(train)} train / {len(test)} test rows")
    return train, test


@task
def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features between the following features:
    - LoanToIncomeRatio
    - PaymentToIncomeRatio
    - Dti
    """
    df = (
        df
        .assign(LoanToIncomeRatio = lambda df_: np.where(
            df_["IncomeTotal"] == 0, 
            0, 
            df_["Amount"] / df_["IncomeTotal"]
        ))
        .assign(PaymentToIncomeRatio = lambda df_: np.where(
            df_["IncomeTotal"] == 0, 
            0, 
            df_["MonthlyPayment"] / df_["IncomeTotal"]
        ))
        .assign(Dti = lambda df_: np.where(
            df_["IncomeTotal"] == 0, 
            0, 
            df_["LiabilitiesTotal"] / df_["IncomeTotal"]
        ))
    )

    return df


LGD_CATEGORICAL_FEATURES = ["Country", "Rating"]


@task
def feature_engineering_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the LGD model from information known at the default date.
    DefaultDate is kept for the out-of-time split and dropped before training.
    """
    df = (
        df
        .assign(MonthsOnBookAtDefault = lambda df_: (df_["DefaultDate"] - df_["LoanDate"]).dt.days / 30.44)
        .assign(ExposureToAmountRatio = lambda df_: df_["EAD1"] / df_["Amount"])
        .drop(columns=["LoanDate"])
    )

    return df
