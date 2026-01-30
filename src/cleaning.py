import pandas as pd

def load_and_clean_data(file_path):

    df = pd.read_csv(file_path)

cols_to_use = [
    'Year', 'Disaster Type', 'Country', 'Region',
    'Start Month', 'Start Day',
    'Total Deaths', 'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value'
]
    df = df[cols_to_use]

    df.dropna(subset=['Disaster Type', 'Total Deaths', 'Total Affected'], inplace=True)


    df.fillna(0, inplace=True)

    return df

