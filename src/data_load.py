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
                      "Gender": 2, # 2 Other
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
                 "Gender",
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


@task
def data_preprocessing_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choosing the features available at the time when the loan had defaulted.
    """
    # exposure_features = ["EAD2", "EAD1", "InterestAndPenaltyBalance", "PrincipalBalance", "Amount", "LoanDuration"]
    # severity_features = ["CurrentDebtDaysPrimary", "CurrentDebtDaysSecondary", "ActiveLateCategory", "WorseLateCategory"]
    # recovery_features = ["RecoveryStage", "DebtOccuredOn","DebtOccuredOnForSecondary"]
    # servicing_features = ["PrincipalDebtServicingCost", "InterestAndPenaltyDebtServicingCost"]
    # collection_features = ["StageActiveSince", "DebtOccuredOn", "DebtOccuredOnForSecondary"]
    
    missing_values = {
        "PrincipalDebtServicingCost": 0, # Monetary or count feature
        "RecoveryStage": 0, # If Null then 0
        "InterestAndPenaltyDebtServicingCost": 0, # Monetary or count feature
        "InterestAndPenaltyBalance": 0, # Monetary or count feature
        "CurrentDebtDaysPrimary": 0, # Count feature
        "CurrentDebtDaysSecondary": 0, # Count feature
        "ActiveLateCategory": "0", #never late,
        "WorseLateCategory": "0",#never late
    }

    data = (
        df
        .query("DefaultDate.notna()") # Get only Default Loans
        .filter(
            ["EAD1",
             "EAD2",
             "InterestAndPenaltyBalance",
             "PrincipalBalance",
             "Amount",
             "LoanDuration",
             "CurrentDebtDaysPrimary",
             "CurrentDebtDaysSecondary",
             "PrincipalDebtServicingCost",
             "InterestAndPenaltyDebtServicingCost",
             "ActiveLateCategory",
             "WorseLateCategory",
             "LossGivenDefault",
             "RecoveryStage",
             "StageActiveSince",
             "DebtOccuredOn",
             "DebtOccuredOnForSecondary",
             "Rescheduled",
             "DefaultDate",
             "LastPaymentOn",
             "MaturityDate_Last"
             ]
        )
        .fillna(value = missing_values)
        .assign(EAD1 = lambda df_: df_["EAD1"].fillna(df_["EAD1"].median()),
                EAD2 = lambda df_: df_["EAD2"].fillna(df_["EAD2"].median()))
        .assign(InterestAndPenaltyBalance = lambda df_: round(df_["InterestAndPenaltyBalance"], 0),
                PrincipalBalance = lambda df_: round(df_["PrincipalBalance"], 0),
                Amount = lambda df_: round(df_["Amount"], 0),
                LoanDuration = lambda df_: round(df_["LoanDuration"], 0),
                CurrentDebtDaysPrimary = lambda df_: round(df_["CurrentDebtDaysPrimary"], 0),
                CurrentDebtDaysSecondary = lambda df_: round(df_["CurrentDebtDaysSecondary"], 0),
                PrincipalDebtServicingCost = lambda df_: round(df_["PrincipalDebtServicingCost"], 0),
                InterestAndPenaltyDebtServicingCost = lambda df_: round(df_["InterestAndPenaltyDebtServicingCost"], 0))
        .astype({
            "EAD1": "int64",
            "EAD2": "int64",
            "InterestAndPenaltyBalance": "int64",
            "PrincipalBalance": "int64",
            "Amount": "int64",
            "LoanDuration": "int64",
            "CurrentDebtDaysPrimary": "int64",
            "CurrentDebtDaysSecondary": "int64",
            "PrincipalDebtServicingCost": "int64",
            "InterestAndPenaltyDebtServicingCost": "int64",
            "StageActiveSince": "datetime64[ns]",
            "DebtOccuredOn": "datetime64[ns]",
            "DebtOccuredOnForSecondary": "datetime64[ns]",
            "DefaultDate": "datetime64[ns]",
            "LastPaymentOn": "datetime64[ns]",
            "MaturityDate_Last": "datetime64[ns]"
        })
        .assign(LastObservationDate = lambda df_: df_.filter(["DefaultDate", "LastPaymentOn", "MaturityDate_Last"]).max(axis = 1))
        .dropna(subset=["LossGivenDefault"]) # Drop rows where LossGivenDefault is missing
        .reset_index(drop = True)
        .drop(["DefaultDate", "LastPaymentOn", "MaturityDate_Last"], axis = 1)
    )

    return data








