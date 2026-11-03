import pandas as pd
import pytest

from data_load import data_preprocessing_lgd
from feature_engineering import feature_engineering_lgd

LEAKY_COLUMNS = [
    "PrincipalBalance", "InterestAndPenaltyBalance", "RecoveryStage", "StageActiveSince",
    "PrincipalDebtServicingCost", "InterestAndPenaltyDebtServicingCost", "ActiveLateCategory",
    "WorseLateCategory", "CurrentDebtDaysPrimary", "PrincipalRecovery", "LossGivenDefault", "Gender",
]


def raw_defaults() -> pd.DataFrame:
    base = {
        "Amount": 2000.0, "LoanDuration": 36, "Interest": 30.0, "MonthlyPayment": 90.0,
        "IncomeTotal": 1500.0, "DebtToIncome": 20.0, "Age": 35, "NewCreditCustomer": True,
        "VerificationType": 3.0, "HomeOwnershipType": 1.0, "Country": "EE", "Rating": "D",
        "EAD2": 1500.0, "LoanDate": "2018-01-01", "ReportAsOfEOD": "2023-10-15", "Gender": 1.0,
        "PrincipalBalance": 300.0, "InterestAndPenaltyBalance": 50.0, "RecoveryStage": 2,
        "StageActiveSince": "2023-01-01", "PrincipalDebtServicingCost": 10.0,
        "InterestAndPenaltyDebtServicingCost": 5.0, "ActiveLateCategory": "180",
        "WorseLateCategory": "180", "CurrentDebtDaysPrimary": 900, "LossGivenDefault": 0.5,
    }
    rows = [
        # 40% of EAD recovered -> LGD 0.6
        {"DefaultDate": "2019-01-01", "EAD1": 1000.0, "PrincipalRecovery": 400.0},
        # Recovered more than EAD -> clipped to 0
        {"DefaultDate": "2019-01-01", "EAD1": 1000.0, "PrincipalRecovery": 1200.0},
        # Default only 10 months before snapshot -> recoveries incomplete, excluded
        {"DefaultDate": "2022-12-15", "EAD1": 1000.0, "PrincipalRecovery": 0.0},
        # Never defaulted -> excluded even though LossGivenDefault is filled
        {"DefaultDate": None, "EAD1": None, "PrincipalRecovery": None},
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_lgd_target_is_realised_and_mature_only():
    out = data_preprocessing_lgd(raw_defaults())
    assert out["LGD"].tolist() == pytest.approx([0.6, 0.0])


def test_lgd_features_exclude_post_default_information():
    out = feature_engineering_lgd(data_preprocessing_lgd(raw_defaults()))
    assert not set(LEAKY_COLUMNS) & set(out.columns)
    assert out["MonthsOnBookAtDefault"].iloc[0] == pytest.approx(365 / 30.44)
    assert out["ExposureToAmountRatio"].iloc[0] == pytest.approx(0.5)
