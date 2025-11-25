from prefect import task
import pandas as pd

#@task
def load_data(file_name: str):
    data = (
        pd.read_csv(file_name)
        .drop("application_id", axis = 1)
    )
    return data








