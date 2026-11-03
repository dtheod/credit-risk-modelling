from prefect import task
import pandas as pd
import numpy as np

@task
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the columns of interest from the raw data
    - Make sure we do not select features that are not available at the time of the loan application (FirstPaymentDate)
    """
    missing_values = {"VerificationType": -1, # -1 Other
                      "EmploymentStatus": -1, # -1 Other
                      "Education": -1, # -1 Other
                      "HomeOwnershipType": -1, # -1 Other
                      "NoOfPreviousLoansBeforeLoan": 0,
                      "AmountOfPreviousLoansBeforeLoan": 0,
                      "PreviousEarlyRepaymentsCountBeforeLoan": 0, # 0 Majority category
                      "PreviousRepaymentsBeforeLoan": 0} # 0 Majority category
    data = (
        df
        .filter([
                 "LoanId",
                 "NewCreditCustomer",
                 "VerificationType",
                 "LoanDuration",
                 "IncomeFromPrincipalEmployer",
                 "IncomeFromPension",
                 "IncomeFromFamilyAllowance",
                 "IncomeFromSocialWelfare",
                 "IncomeFromLeavePay",
                 "IncomeFromChildSupport",
                 "IncomeOther",
                 "EmploymentDurationCurrentEmployer",
                 "IncomeTotal",
                 "Education",
                 "EmploymentStatus",
                 "ExistingLiabilities",
                 "LiabilitiesTotal",
                 "LanguageCode",
                 "Amount",
                 "Age",
                 "Interest",
                 "UseOfLoan",
                 "CreditScoreEsMicroL",
                 "AmountOfPreviousLoansBeforeLoan",
                 "PreviousEarlyRepaymentsCountBeforeLoan",
                 "PreviousRepaymentsBeforeLoan",
                 "Country",
                 "MonthlyPayment",
                 "HomeOwnershipType",
                 "NoOfPreviousLoansBeforeLoan"
                 ])
        .fillna(value = missing_values)
        .assign(
            LiabilitiesTotal = lambda df_: round(df_.LiabilitiesTotal, 0),
            AmountOfPreviousLoansBeforeLoan = lambda df_: round(df_.AmountOfPreviousLoansBeforeLoan, 0),
            MonthlyPayment = lambda df_: round(df_.MonthlyPayment, 0)
        )
    )
    return data


LGD_MIN_WORKOUT_DAYS = 730


@task
def data_preprocessing_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the LGD modelling table from defaulted loans.

    Target: realised LGD = 1 - PrincipalRecovery / EAD1, clipped to [0, 1]. Bondora's
    LossGivenDefault column is not used: it is filled for non-defaulted loans too, i.e. it
    is Bondora's own ex-ante estimate rather than an observed loss.

    Only defaults with at least LGD_MIN_WORKOUT_DAYS between DefaultDate and the data
    snapshot (ReportAsOfEOD) are kept, so recoveries are largely complete.

    Features are restricted to information known at the default date: application data
    and exposure at default. Balances, recovery stage, servicing costs and late categories
    are measured at the snapshot and would leak the outcome.
    """
    df = df.copy()
    for col in ["LoanDate", "DefaultDate", "ReportAsOfEOD"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    snapshot_date = df["ReportAsOfEOD"].max()

    data = (
        df
        .query("DefaultDate.notna() and EAD1 > 0")
        .loc[lambda df_: (snapshot_date - df_["DefaultDate"]).dt.days >= LGD_MIN_WORKOUT_DAYS]
        .assign(LGD = lambda df_: (1 - df_["PrincipalRecovery"].fillna(0) / df_["EAD1"]).clip(0, 1))
        .filter([
            # Known at application
            "Amount",
            "LoanDuration",
            "Interest",
            "MonthlyPayment",
            "IncomeTotal",
            "DebtToIncome",
            "Age",
            "NewCreditCustomer",
            "VerificationType",
            "HomeOwnershipType",
            "Country",
            "Rating",
            # Known at default
            "EAD1",
            "EAD2",
            "LoanDate",
            "DefaultDate",
            # Target
            "LGD",
        ])
        .fillna({"Rating": "Unknown", "VerificationType": -1, "HomeOwnershipType": -1})
        .astype({"NewCreditCustomer": "int64", "VerificationType": "int64", "HomeOwnershipType": "int64"})
        .reset_index(drop = True)
    )

    return data
