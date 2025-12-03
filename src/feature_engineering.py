from prefect import task
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np



@task
def train_test_split_task(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """
    Initial split to create the test set to test the final model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target,
        test_size=0.10,
        stratify=target,
        random_state=42       
    )
    return X_train, X_test, y_train, y_test


@task
def one_hot_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the features
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = df.select_dtypes(include=["category", "object"])
    non_categorical_features = df.select_dtypes(exclude=["category", "object"])
    
    # Convert all categorical features to string to handle mixed types
    categorical_features = categorical_features.astype(str)
    
    encoder.fit(categorical_features)
    encoded_features = encoder.transform(categorical_features)
    encoded_features = pd.DataFrame(encoded_features.toarray(), 
                                    columns=encoder.get_feature_names_out()
                                    )
    joblib.dump(encoder, "./models/one_hot_encoder.joblib")
    
    # Reset index of non-categorical features to align with the new encoded dataframe
    non_categorical_features = non_categorical_features.reset_index(drop=True)
    
    return pd.concat([non_categorical_features, encoded_features], axis=1)



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


@task
def feature_engineering_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the LGD model 
    """
    df = (
        df
        .assign(HasPrimaryArrears = lambda df_: np.where(df_['CurrentDebtDaysPrimary'] > 0, 1, 0))
        .assign(HasSecondaryArrears = lambda df_: np.where(df_['CurrentDebtDaysSecondary'] > 0, 1, 0))
        .assign(StageActiveDays = lambda df_: ((df_['LastObservationDate'] - df_['StageActiveSince']).dt.days).clip(lower=0))
        .assign(DaysInPrincipalDebt = lambda df_: ((df_['LastObservationDate'] - df_['DebtOccuredOn']).dt.days).clip(lower=0))
        .assign(DaysInInterestDebt = lambda df_: ((df_['LastObservationDate'] - df_['DebtOccuredOnForSecondary']).dt.days).clip(lower=0))
        .assign(HasPrincipalDebt = lambda df_: np.where(
            df_["DebtOccuredOn"].notna(), 
            1, 
            0
        ))
        .assign(HasInterestDebt = lambda df_: np.where(
            df_["DebtOccuredOnForSecondary"].notna(), 
            1, 
            0
        ))
        .drop(columns=["StageActiveSince", "DebtOccuredOn", "DebtOccuredOnForSecondary", "LastObservationDate"])
        .fillna({"DaysInPrincipalDebt": 0,
                 "DaysInInterestDebt": 0,
                 "StageActiveDays": 0})
    )

    return df
