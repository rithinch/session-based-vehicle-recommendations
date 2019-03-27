import glob
import pandas as pd

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

def concat_csv_files(files_path):

    filenames = glob.glob(files_path, recursive=True)

    combined_csv = pd.concat([pd.read_csv(f, skiprows=6) for f in filenames])

    combined_csv['reg_no'] = combined_csv['Page'].apply(lambda x: get_details(x)['reg_no'])
    combined_csv['make'] = combined_csv['Page'].apply(lambda x: get_details(x)['make'])
    combined_csv['model'] = combined_csv['Page'].apply(lambda x: get_details(x)['model'])
    combined_csv['fuel'] = combined_csv['Page'].apply(lambda x: get_details(x)['fuel'])
    combined_csv['colour'] = combined_csv['Page'].apply(lambda x: get_details(x)['colour'])
    combined_csv['body'] = combined_csv['Page'].apply(lambda x: get_details(x)['body'])
    combined_csv['trasmission'] = combined_csv['Page'].apply(lambda x: get_details(x)['transmission'])

    combined_csv.to_csv("vehicle_page_views.csv", index=False)

    print(combined_csv.head())

    error_data = pd.DataFrame(data={"Page URL": errors})
    print(f"Skipped URL's (Without Reg No): {len(errors)}")
    error_data.to_csv("invalid_page_urls.csv", index=False)

files_path = "real_data/**/*.csv"

concat_csv_files(files_path)



    


