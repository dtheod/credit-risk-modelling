import sys
from prefect import flow, task
from data_load import load_data
import os

#@flow(log_prints=True)
def main(file_name: str = "./data/raw_data/bondora_credit_risk_synthetic.csv") -> None:
    print(os.getcwd())
    ready_data = load_data(file_name)
    print(ready_data.head())   



if __name__ == "__main__":

    main()
