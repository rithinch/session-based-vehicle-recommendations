import pandas as pd

clicks_df = pd.read_csv("clicks_raw.csv")
vehicle_size = 500
session_size = -1

clicks_df.dropna(inplace=True)

grouped = clicks_df.groupby('session_id')['page'].agg(["count"])

filtered = grouped.query('count>1')

clean_df = pd.merge(clicks_df, filtered, on='session_id')

clean_df.drop('count', axis=1, inplace=True)

filename = "clicks_clean.csv"

if (session_size > 0):

    random_sample = filtered.sample(n=session_size)

    clean_df = pd.merge(clean_df, random_sample, on='session_id')

    clean_df.drop('count', axis=1, inplace=True)
    
    filename = f"clicks_clean_{session_size}_sessions.csv"
    
if (vehicle_size > 0):

    grouped = clean_df.groupby('reg_no')['session_id'].agg(["count"])
    
    random_sample = grouped.sample(n=vehicle_size)
    
    x = list(random_sample.index.get_level_values('reg_no'))
    
    sessions = clean_df[clean_df['reg_no'].isin(x)]
    
    clean_df = clean_df[clean_df['session_id'].isin(sessions['session_id'])]
    
    filename = f"clicks_clean_{vehicle_size}_vehicles.csv"

clean_df.to_csv(filename, index=False) #Save the cleaned data