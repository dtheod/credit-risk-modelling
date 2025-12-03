from prefect import flow, task
import pandas as pd
import numpy as np
from utils import log_transform_series
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import joblib



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
      DefaultTarget_12m = 0  -> observed >= 12 months with no default
      DefaultTarget_12m = NaN -> censored (not yet 12 months of observation), usually drop from training
    """

    df = df.copy()

    # Put all date columns in a list to find the max date
    date_cols = [
        "LoanDate",
        "DefaultDate",
        "LoanApplicationStartedDate",
        "LastPaymentOn",
        "MaturityDate_Last",
        "ContractEndDate",
        "SoldInResale_Date",
        "ListedOnUTC"
    ]

    # Keep only those that exist in the dataframe
    date_cols = [c for c in date_cols if c in df.columns]

    df[date_cols] = df[date_cols].apply(
        pd.to_datetime, errors="coerce"
    )

    # --- 2. Loan-level last observed date (max of all known dates for that loan) ---
    df["last_obs_date"] = df[date_cols].max(axis=1)

    # --- 3. Days to default & default within 12 months ---
    days_to_default = (df["DefaultDate"] - df["LoanDate"]).dt.days
    default_within_12m = df["DefaultDate"].notna() & (days_to_default <= 365)

    # --- 4. How long the loan is observed in total ---
    days_observed = (df["last_obs_date"] - df["LoanDate"]).dt.days

    # Loan has at least 12 months of observable life
    fully_observed_12m = days_observed >= 365

    # --- 5. Build target ---
    #  1 if defaulted within 12m
    #  0 if fully observed 12m and no default within 12m
    #  NaN otherwise (too young / censored)
    df["DefaultTarget_12m"] = np.where(
        default_within_12m,
        1,
        np.where(
            fully_observed_12m,
            0,
            np.nan
        )
    )

    # --- 6. Optionally drop helper columns (keep LoanDate if you want vintage features) ---
    df = df.drop(columns=["last_obs_date"], errors="ignore")

    # Removing the loans that are not fully observed
    df.dropna(subset=["DefaultTarget_12m"], inplace=True)

    # If you want to mimic your original behavior more closely, you can also drop
    # DefaultDate and LoanApplicationStartedDate here:
    df = df.drop(columns=["DefaultDate", "LoanApplicationStartedDate"], errors="ignore")

    return df


@task
def clean_remove_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes data bad collection data
    """
    return df

@task
def set_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all float columns to integers
    (Object/category columns are already handled by categorical_variables_task)
    """
    # Convert all float columns to integers
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        df[col] = df[col].astype('int64')
    
    print(f"Converted {len(float_cols)} float columns to int64")
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
        .assign(GenderCategorical = lambda df_: df_.Gender.case_when(
            [
                (df_["Gender"] == 0, "Male"),
                (df_["Gender"] == 1, "Female"),
                (df_["Gender"] == -1, "Undefined"),
                (df_["Gender"] == 2, "Undefined")
            ]
            )
        )
        .drop(["Gender"], axis=1)
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
                "GenderCategorical": "category",
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



@task
def continuous_variables_transformations(df: pd.DataFrame) -> pd.DataFrame:

    """
    Applying Log transformations to highly skewed variables like:
        - AmountOfPreviousLoansBeforeLoan
        - FreeCash
        - LiabilitiesTotal
    """

    df["AmountOfPreviousLoansBeforeLoan"] = log_transform_series("AmountOfPreviousLoansBeforeLoan", df)
    df["LiabilitiesTotal"] = log_transform_series("LiabilitiesTotal", df)
    return df

@task
def ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encoding of categorical variables:
    - CreditScoreEsMicroL: M1-M10 (subprime categories)
    """
    # Define categories for each ordinal feature
    subprime_categories = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
    
    # Create single encoder with both category lists
    ordinal_encoder = OrdinalEncoder(
        categories=[subprime_categories],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    
    # Encode both columns at once
    encoded_values = ordinal_encoder.fit_transform(df[["CreditScoreEsMicroL"]])
    
    # Assign encoded values to new columns
    df["CreditScoreEsMicroLEnc"] = encoded_values[:, 0]
    
    # Drop original columns
    df = df.drop(["CreditScoreEsMicroL"], axis=1)
    
    # Save the encoder
    joblib.dump(ordinal_encoder, "./models/ordinal_encoder.joblib")
    
    return df

@task
def ordinal_encoding_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encoding of late variables
    """
    late_or_worse_order = [
        "0",        # never late
        "1-7", "8-15",
        "16-30", "31-60",
        "61-90", "91-120",
        "121-150", "151-180", "180"
    ]
    late_map = {v: i for i, v in enumerate(late_or_worse_order)}
    worse_late_map = {v: i for i, v in enumerate(late_or_worse_order)}
    
    df = df.assign(
        ActiveLateCategoryOrd = lambda df_: df_["ActiveLateCategory"].map(late_map),
        WorseLateCategoryOrd = lambda df_: df_["WorseLateCategory"].map(worse_late_map)
    )
    
    df = df.drop(["ActiveLateCategory", "WorseLateCategory"], axis=1)
    
    return df


@task
def knn_impute_continuous_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in continuous variables using KNN Imputer
    """
    # Identify continuous (numeric) columns, excluding the target if present
    continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_neighbors = 5

    
    # Check if there are any missing values in continuous columns
    missing_counts = df[continuous_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    print(f"Imputing missing values in {len(cols_with_missing)} continuous columns using KNN (n_neighbors={n_neighbors})")
    print(f"Columns with missing values: {cols_with_missing}")
    
    # Apply KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[continuous_cols] = imputer.fit_transform(df[continuous_cols])
    
    joblib.dump(imputer, "./models/knn_imputer.joblib")


    return df
