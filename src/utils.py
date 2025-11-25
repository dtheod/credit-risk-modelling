def impute_good_standing_missing_values(row):
    if pd.isnull(row['past_bondora_good_standing']):
        return 2
    else:
        return row['past_bondora_good_standing']
    