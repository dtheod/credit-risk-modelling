from prefect import flow, task
import pandas as pd
import numpy as np
from utils import log_transform_series



@task
def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with:
        - Interest > 100
        - LiabilitiesTotal > 250000
        - DebtToIncome > 100
        - Age < 18
    Will append this list once I find more criteria to remove.
    """
    df = (
        df
        .query("Interest < 100")
        .query("LiabilitiesTotal < 250000")
        .query("DebtToIncome < 100")
        .query("Age >= 18")
    ) 
    return df


@task
def create_pd12_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a PD-12 target:
      DefaultTarget_12m = 1  -> default within 12 months of LoanDate
      DefaultTarget_12m = 0  -> no default within 12 months of LoanDate
    Loans issued less than 12 months before the data snapshot (ReportAsOfEOD) are
    censored and dropped, whether or not they have defaulted yet. Scheduled dates such as
    MaturityDate_Last or ContractEndDate lie in the future and say nothing about how long
    a loan has actually been observed, so they are not used here.
    """

    df = df.copy()
    df["LoanDate"] = pd.to_datetime(df["LoanDate"], errors="coerce")
    df["DefaultDate"] = pd.to_datetime(df["DefaultDate"], errors="coerce")
    snapshot_date = pd.to_datetime(df["ReportAsOfEOD"], errors="coerce").max()

    # Keep only loans with a full 12-month observation window before the snapshot
    df = df[(snapshot_date - df["LoanDate"]).dt.days >= 365]

    days_to_default = (df["DefaultDate"] - df["LoanDate"]).dt.days
    df["DefaultTarget_12m"] = (df["DefaultDate"].notna() & (days_to_default <= 365)).astype("int64")

    df = df.drop(columns=["DefaultDate", "LoanApplicationStartedDate"], errors="ignore")

    return df


@task
def clean_remove_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes data bad collection data
    """
    return df

# Numeric columns that hold category codes or counts. Everything else (amounts, rates,
# engineered ratios) stays float.
CODE_COLUMNS = [
    "VerificationType",
    "LoanDuration",
    "Education",
    "EmploymentStatus",
    "ExistingLiabilities",
    "LanguageCode",
    "Age",
    "UseOfLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan",
    "HomeOwnershipType",
    "NoOfPreviousLoansBeforeLoan",
]


@task
def set_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rounds category-code and count columns (CODE_COLUMNS) to integers so they can be
    mapped to categories. Continuous columns and engineered ratios are left untouched.
    """
    cols = [c for c in CODE_COLUMNS if c in df.columns]
    df[cols] = df[cols].round().astype("int64")
    return df

@task
def categorical_variables_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    From numeric to categorical variables
    """

    df = (
        df
        .assign(principal_employer_binary = lambda df_: np.where(df_.IncomeFromPrincipalEmployer > 0, 1, 0))
        .assign(pension_binary = lambda df_: np.where(df_.IncomeFromPension > 0, 1, 0))
        .assign(family_allowance = lambda df_: np.where(df_.IncomeFromFamilyAllowance > 0, 1, 0))
        .assign(social_welfare_binary = lambda df_: np.where(df_.IncomeFromSocialWelfare > 0, 1, 0))
        .assign(leave_pay_binary = lambda df_: np.where(df_.IncomeFromLeavePay > 0, 1, 0))
        .assign(child_support_binary = lambda df_: np.where(df_.IncomeFromChildSupport > 0, 1, 0))
        .assign(other_income_binary = lambda df_: np.where(df_.IncomeOther > 0, 1, 0))
        .assign(number_income_streams = lambda df_: df_.principal_employer_binary + df_.pension_binary + df_.family_allowance + df_.social_welfare_binary + df_.leave_pay_binary + df_.child_support_binary + df_.other_income_binary)
        .drop(["principal_employer_binary", "pension_binary", "family_allowance", "social_welfare_binary", "leave_pay_binary", "child_support_binary", "other_income_binary"], axis=1)   
    )

    df = (
        df
        .assign(new_customer_binary = lambda df_: np.where(df_.NewCreditCustomer==True, 1, 0))
        .drop(["NewCreditCustomer"], axis=1)
    )

    df = (
        df
        .assign(VerificationCategorical = lambda df_: df_.VerificationType.case_when(
                [
                    (df_["VerificationType"] == 0, "NotSet"),
                    (df_["VerificationType"] == 1, "IncomeNotVerified"),
                    (df_["VerificationType"] == 2, "UnverifiedCrossReferencedPhone"),
                    (df_["VerificationType"] == 3, "IncomeVerified"),
                    (df_["VerificationType"] == 4, "IncomeExpensesVerified"),
                    (df_["VerificationType"] == -1, "Unknown")
                ]
            )
        )
        .drop(["VerificationType"], axis=1)
    )

    df = (
        df
        .assign(EducationCategorical = lambda df_: df_.Education.case_when(
                [
                    (df_["Education"] == 0, "NotSet"),
                    (df_["Education"] == 1, "Primary"),
                    (df_["Education"] == 2, "BasicEducation"),
                    (df_["Education"] == 3, "VocationalEducation"),
                    (df_["Education"] == 4, "SecondaryEducation"),
                    (df_["Education"] == 5, "HigherEducation"),
                    (df_["Education"] == -1, "NotSet")
                ]
            )
        )
        .drop(["Education"], axis=1)
    )

    df = (
        df
        .assign(EmploymentCategorical = lambda df_: df_.EmploymentStatus.case_when(
            [
                (df_["EmploymentStatus"] == 1, "Unemployed"),
                (df_["EmploymentStatus"] == 2, "Partial"),
                (df_["EmploymentStatus"] == 3, "FullyEmployed"),
                (df_["EmploymentStatus"] == 4, "SelfEmployed"),
                (df_["EmploymentStatus"] == 5, "Entrepreneur"),
                (df_["EmploymentStatus"] == 6, "Retiree"),
                (df_["EmploymentStatus"] == -1, "Unknown"),
                (df_["EmploymentStatus"] == 0, "Unknown")
            ]
            )
        )
        .drop(["EmploymentStatus"], axis=1)
        )
    
    df = (
        df
        .assign(HomeOwnershipCategorical = lambda df_: df_.HomeOwnershipType.case_when(
            [
                (df_["HomeOwnershipType"] == 0, "Homeless"),
                (df_["HomeOwnershipType"] == 1, "Owner"),
                (df_["HomeOwnershipType"] == 2, "LivingWithParents"),
                (df_["HomeOwnershipType"] == 3, "TenantPreFurnished"),
                (df_["HomeOwnershipType"] == 4, "TenantNotPreFurnished"),
                (df_["HomeOwnershipType"] == 5, "CouncilHouse"),
                (df_["HomeOwnershipType"] == 6, "JointTenant"),
                (df_["HomeOwnershipType"] == 7, "JointOwnership"),
                (df_["HomeOwnershipType"] == 8, "Mortgage"),
                (df_["HomeOwnershipType"] == 9, "OwnerWithEncumbrance"),
                (df_["HomeOwnershipType"] == 10, "Other"),
                (df_["HomeOwnershipType"] == -1, "Other"),

            ]
            )
        )
        .drop(["HomeOwnershipType"], axis=1)
    )

    df = (
        df
        .assign(UseOfLoanCategorical = lambda df_: df_.UseOfLoan.case_when(
            [
                (df_["UseOfLoan"] == 0, "LoanConsolidation "),
                (df_["UseOfLoan"] == 1, "RealEstate"),
                (df_["UseOfLoan"] == 2, "HomeImprovement"),
                (df_["UseOfLoan"] == 3, "Business"),
                (df_["UseOfLoan"] == 4, "Education"),
                (df_["UseOfLoan"] == 5, "Travel"),
                (df_["UseOfLoan"] == 6, "Vehicle"),
                (df_["UseOfLoan"] == 7, "Other"),
                (df_["UseOfLoan"] == 8, "Health"),
                (df_["UseOfLoan"] == 101, "WorkingCapitalFinancing"),
                (df_["UseOfLoan"] == 102, "PurchaseMachineryEquipment"),
                (df_["UseOfLoan"] == 103, "OtherBusiness"),
                (df_["UseOfLoan"] == 104, "OtherBusiness"),
                (df_["UseOfLoan"] == 105, "OtherBusiness"),
                (df_["UseOfLoan"] == 106, "OtherBusiness"),
                (df_["UseOfLoan"] == 107, "OtherBusiness"),
                (df_["UseOfLoan"] == 108, "OtherBusiness"),
                (df_["UseOfLoan"] == 109, "OtherBusiness"),
                (df_["UseOfLoan"] == 110, "OtherBusiness"),
                (df_["UseOfLoan"] == -1, "NotSet")
            ]
            )
        )
        .drop(["UseOfLoan"], axis=1)
    )

    df = (
        df
        .assign(Language = lambda df_: df_.LanguageCode.case_when(
            [
                (df_["LanguageCode"] == 1, "Estonian"),
                (df_["LanguageCode"] == 2, "English"),
                (df_["LanguageCode"] == 3, "Russian"),
                (df_["LanguageCode"] == 4, "Finnish"),
                (df_["LanguageCode"] == 5, "German"),
                (df_["LanguageCode"] == 6, "Spanish"),
                (df_["LanguageCode"] == 7, "Slovakian"),
                (df_["LanguageCode"] > 7, "Other"),
                (df_["LanguageCode"] == -1, "Other")
            ]
        ))
        .drop(["LanguageCode"], axis=1)
    )

    df = (
        df
        .astype(
            {
                "EducationCategorical": "category",
                "EmploymentCategorical": "category",
                "VerificationCategorical": "category",
                "HomeOwnershipCategorical": "category",
                "UseOfLoanCategorical": "category",
                "LoanId": "string"
            }
        )
    )

    return df



LOG_TRANSFORM_COLUMNS = ["AmountOfPreviousLoansBeforeLoan", "LiabilitiesTotal"]


@task
def continuous_variables_transformations(df: pd.DataFrame, caps: dict) -> pd.DataFrame:

    """
    Applying Log transformations to highly skewed variables (LOG_TRANSFORM_COLUMNS),
    after clipping at caps fitted on the training data (see ContinuousTransformer).
    """

    for col in LOG_TRANSFORM_COLUMNS:
        df[col] = log_transform_series(col, df, caps[col])
    return df
