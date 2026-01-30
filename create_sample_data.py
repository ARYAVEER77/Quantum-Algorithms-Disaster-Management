import pandas as pd
import os

# Create sample disaster data
data = {
    'Year': list(range(1970, 2022)),
    'Disaster Type': ['Flood']*30 + ['Earthquake']*30 + ['Cyclone']*30 + ['Drought']*22,
    'Country': ['Bangladesh', 'India', 'China', 'USA', 'Philippines']*26,
    'Region': ['Asia']*40 + ['Americas']*40 + ['Africa']*22,
    'Start Month': list(range(1, 13))*4 + list(range(1, 11)),
    'Start Day': list(range(1, 32))*3 + list(range(1, 11)),
    'Total Deaths': [100 * (i % 10) + 50 for i in range(112)],
    'Total Affected': [10000 * (i % 10) + 5000 for i in range(112)],
    "Total Damages ('000 US$)": [1000000 * (i % 10) + 500000 for i in range(112)],
    'Dis Mag Value': [5.0 + (i % 10)/2.0 for i in range(112)]
}

df = pd.DataFrame(data)

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('DISASTERS', exist_ok=True)

# Save files
df.to_csv('data/raw/emdat_1970_2021.csv', index=False)
df.to_csv('DISASTERS/disasters_1970_2021.csv', index=False)

print("✅ Sample data created with 112 rows")
print(f"Columns: {list(df.columns)}")
