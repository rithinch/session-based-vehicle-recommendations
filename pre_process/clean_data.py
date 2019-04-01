import pandas as pd

clicks_df = pd.read_csv("clicks_raw.csv")

clicks_df.dropna(inplace=True)

grouped = clicks_df.groupby('session_id')['page'].agg(["count"])

filtered = grouped.query('count>1')

clean_df = pd.merge(clicks_df, filtered, on='session_id')

clean_df.drop('count', axis=1, inplace=True)

clean_df.to_csv("clicks_clean.csv", index=False) #Save the cleaned data