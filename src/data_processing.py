from prefect import flow, task
import pandas as pd
from utils import impute_good_standing_missing_values


@task
def data_processing_task(df: pd.DataFrame) -> pd.DataFrame:
    #Impute missing values in past_bondora_good_standing column
    df = df.apply(impute_good_standing_missing_values, axis=1)

    

    df = df.dropna()
    df = df.drop_duplicates()
    
    return df

