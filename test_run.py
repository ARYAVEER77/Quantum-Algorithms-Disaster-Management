import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("🧪 Testing data loading...")

# Create minimal test data
test_data = {
    'Year': [2020, 2021, 2022, 2023],
    'Disaster Type': ['Flood', 'Earthquake', 'Cyclone', 'Drought'],
    'Total Deaths': [100, 200, 50, 300],
    'Total Affected': [10000, 5000, 8000, 20000],
    "Total Damages ('000 US$)": [5000, 10000, 3000, 2000],
    'Dis Mag Value': [5.5, 6.2, 4.8, 5.0],
    'Start Month': [6, 3, 9, 1],
    'Start Day': [15, 20, 5, 10],
    'Country': ['A', 'B', 'C', 'D'],
    'Region': ['Asia', 'Americas', 'Asia', 'Africa']
}

df = pd.DataFrame(test_data)
print(f"✅ Created test DataFrame with {len(df)} rows")

# Test AI model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 
               'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']]
target = df['Disaster Type']

le = LabelEncoder()
y = le.fit_transform(target)
X = features.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

print(f"✅ Test model accuracy: {acc*100:.1f}%")
print(f"✅ Classes: {list(le.classes_)}")
print("\n🎉 Basic functionality tested successfully!")
