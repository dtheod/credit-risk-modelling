import numpy as np
import pandas as pd
import pytest

from app import LoanRequest
from data_processing import create_pd12_target, set_column_dtypes
from feature_engineering import engineered_features, out_of_time_split
from pipeline_builder import build_preprocessing_pipeline

SNAPSHOT = "2023-10-15"


def raw_loans(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw rows with the columns the PD pipeline reads."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(500, 5000, n)
    return pd.DataFrame({
        "LoanId": [f"id{i}" for i in range(n)],
        "NewCreditCustomer": rng.choice([True, False], n),
        "VerificationType": rng.choice([1.0, 2.0, 3.0, 4.0], n),
        "LoanDuration": rng.choice([12, 24, 36, 60], n),
        "IncomeFromPrincipalEmployer": income,
        "IncomeFromPension": 0.0,
        "IncomeFromFamilyAllowance": 0.0,
        "IncomeFromSocialWelfare": 0.0,
        "IncomeFromLeavePay": 0.0,
        "IncomeFromChildSupport": 0.0,
        "IncomeOther": 0.0,
        "EmploymentDurationCurrentEmployer": rng.choice(["UpTo1Year", "MoreThan5Years"], n),
        "IncomeTotal": income,
        "Education": rng.choice([1.0, 3.0, 4.0, 5.0], n),
        "EmploymentStatus": rng.choice([3.0, 4.0, 6.0], n),
        "ExistingLiabilities": rng.integers(0, 5, n),
        "LiabilitiesTotal": rng.uniform(0, 2000, n),
        "LanguageCode": rng.choice([1, 3, 4, 6], n),
        "Amount": rng.uniform(500, 10000, n),
        "Age": rng.integers(18, 70, n),
        "Interest": rng.uniform(5, 60, n),
        "UseOfLoan": rng.choice([0, 2, 6, 7], n),
        "CreditScoreEsMicroL": rng.choice(["M1", "M5", None], n),
        "AmountOfPreviousLoansBeforeLoan": rng.uniform(0, 5000, n),
        "PreviousEarlyRepaymentsCountBeforeLoan": rng.integers(0, 3, n).astype(float),
        "PreviousRepaymentsBeforeLoan": rng.uniform(0, 3000, n),
        "Country": rng.choice(["EE", "FI", "ES"], n),
        "Gender": rng.choice([0.0, 1.0], n),
        "MonthlyPayment": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(50, 400, n)),
        "HomeOwnershipType": rng.choice([1.0, 3.0, 8.0], n),
        "NoOfPreviousLoansBeforeLoan": rng.integers(0, 4, n).astype(float),
    })


def test_ratio_features_survive_dtype_cast():
    df = pd.DataFrame({"Amount": [1000.0], "IncomeTotal": [2000.0], "MonthlyPayment": [150.0],
                       "LiabilitiesTotal": [500.0], "Interest": [24.5], "Education": [4.0]})
    out = set_column_dtypes(engineered_features(df))
    assert out["PaymentToIncomeRatio"].iloc[0] == pytest.approx(0.075)
    assert out["LoanToIncomeRatio"].iloc[0] == pytest.approx(0.5)
    assert out["Dti"].iloc[0] == pytest.approx(0.25)
    assert out["Interest"].iloc[0] == pytest.approx(24.5)
    assert out["Education"].dtype == "int64"


def test_pd12_target_censors_on_snapshot_not_future_dates():
    df = pd.DataFrame({
        "LoanId": ["young", "old_good", "old_bad", "young_bad", "late_default"],
        "LoanDate": ["2023-08-01", "2020-01-01", "2020-01-01", "2023-08-01", "2020-01-01"],
        "DefaultDate": [None, None, "2020-06-01", "2023-09-01", "2022-01-01"],
        # Scheduled maturity years in the future must not count as observation time
        "MaturityDate_Last": ["2028-08-01"] * 5,
        "ReportAsOfEOD": [SNAPSHOT] * 5,
    })
    out = create_pd12_target(df).set_index("LoanId")["DefaultTarget_12m"]
    assert out.to_dict() == {"old_good": 0, "old_bad": 1, "late_default": 0}


def test_out_of_time_split_puts_latest_loans_in_test():
    df = pd.DataFrame({"LoanDate": pd.date_range("2015-01-01", periods=100, freq="W")})
    train, test = out_of_time_split(df, "LoanDate", test_size=0.1)
    assert train["LoanDate"].max() < test["LoanDate"].min()
    assert len(test) == 10


def test_preprocessing_state_is_learned_on_fit_data_only():
    train = raw_loans()
    pipe = build_preprocessing_pipeline().fit(train)

    # Log cap comes from the training data, so an extreme single row is clipped at serving time
    extreme = raw_loans(1, seed=1).assign(LiabilitiesTotal=1e9)
    cap = pipe.named_steps["continuous_transformer"].caps_["LiabilitiesTotal"]
    assert cap < 2000
    assert pipe.transform(extreme)["LiabilitiesTotal"].iloc[0] == pytest.approx(np.log1p(cap))

    # Median imputation uses the training median
    missing = raw_loans(1, seed=2).assign(MonthlyPayment=np.nan)
    imputer = pipe.named_steps["imputer"]
    median = imputer.imputer_.statistics_[imputer.numeric_cols_.index("MonthlyPayment")]
    assert median == pytest.approx(train["MonthlyPayment"].round().median())  # select_columns rounds it
    assert not pipe.transform(missing).isna().any().any()


def test_gender_is_not_a_feature():
    pipe = build_preprocessing_pipeline().fit(raw_loans())
    columns = pipe.transform(raw_loans(5, seed=3)).columns
    assert not any("Gender" in c for c in columns)


def test_api_row_matches_training_columns_and_new_customer_flag():
    train = raw_loans()
    pipe = build_preprocessing_pipeline().fit(train)
    expected = pipe.transform(train).columns

    # A sparse API request: missing fields arrive as None
    for flag, binary in [(True, 1), (False, 0)]:
        request = LoanRequest(NewCreditCustomer=flag, Amount=2000.0, IncomeTotal=3000.0, Country="EE")
        row = pipe.transform(pd.DataFrame([request.model_dump()]))
        assert list(row.columns) == list(expected)
        assert row["new_customer_binary"].iloc[0] == binary


def test_api_parses_boolean_new_customer_strings():
    assert LoanRequest(NewCreditCustomer="false").NewCreditCustomer is False
    assert LoanRequest(NewCreditCustomer="True").NewCreditCustomer is True
