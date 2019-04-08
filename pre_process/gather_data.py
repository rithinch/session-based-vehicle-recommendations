import glob
import pandas as pd
import time

extracted_values = {}
errors = []

def get_details(page_url):

    page_url = page_url.split("?")[0]

    if page_url in extracted_values:
        return extracted_values[page_url]
    
    details = {
        'make':'', 
        'model':'',
        'transmission':'',
        'fuel':'',
        'reg_no':'',
        'body':'',
        'colour':''
    }

    vals = page_url[1:].split("/")

    if len(vals) > 1:
        try:
            spec = vals[3].split("-")
            details['make'] = vals[1].lower()
            details['model'] = vals[2].lower()
            details['transmission'] = spec[0].lower()
            details['fuel'] = spec[1].lower()
            details['reg_no'] = spec[-1].lower()
            details['body'] = " ".join(spec[3:-1]).lower()
            details['colour'] = spec[2].lower()
        except IndexError:
            errors.append(page_url)

    extracted_values[page_url] = details

    return details

def get_timeframe(row):

    string = f"{row['date']}T{row['time']}"
    return int(time.mktime(time.strptime(string, '%Y-%m-%dT%H:%M')))

def rename_df_columns(df):
    names = {'Client ID': 'client_id', 
            'Page': 'page', 
            'Session ID': 'session_id',
            'Hour of Day': 'date',
            'Minute': 'time'}
    
    df.rename(columns=names, inplace=True)
    
def concat_csv_files(files_path, filename):

    filenames = glob.glob(files_path, recursive=True)

    combined_csv = pd.concat([pd.read_csv(f, skiprows=6) for f in filenames])
    
    rename_df_columns(combined_csv)
    combined_csv.drop('Users', axis=1, inplace=True)
    combined_csv['date'] = combined_csv['date'].astype(str)

    combined_csv['time'] = combined_csv.apply(lambda x: f"{x['date'][-2:]}:{x['time']}", axis=1)
    combined_csv['date'] = combined_csv['date'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
    combined_csv['timeframe'] = combined_csv.apply(get_timeframe, axis=1)
    
    combined_csv['reg_no'] = combined_csv['page'].apply(lambda x: get_details(x)['reg_no'])
    combined_csv['make'] = combined_csv['page'].apply(lambda x: get_details(x)['make'])
    combined_csv['model'] = combined_csv['page'].apply(lambda x: get_details(x)['model'])
    combined_csv['fuel'] = combined_csv['page'].apply(lambda x: get_details(x)['fuel'])
    combined_csv['colour'] = combined_csv['page'].apply(lambda x: get_details(x)['colour'])
    combined_csv['body'] = combined_csv['page'].apply(lambda x: get_details(x)['body'])
    combined_csv['trasmission'] = combined_csv['page'].apply(lambda x: get_details(x)['transmission'])

    combined_csv.to_csv(filename, index=False)

    print(combined_csv.head())

    error_data = pd.DataFrame(data={"Page URL": errors})
    print(f"Skipped URL's (Without Reg No): {len(errors)}")
    error_data.to_csv("invalid_page_urls.csv", index=False)

files_path = "real_data/batch_3/*.csv"

concat_csv_files(files_path, "clicks_latest_small_raw.csv")



    


