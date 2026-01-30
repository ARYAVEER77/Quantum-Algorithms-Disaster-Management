import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from cleaning import load_and_clean_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def calculate_vif(df):
    print("\n📊 Calculating Variance Inflation Factor (VIF)...")
    # Select numerical features for VIF as per Table 2 in paper
    features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']]
    features = features.fillna(0)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]
    
    print(vif_data)
    return vif_data

def run_ablation_study(df):
    print("\n📉 Running Ablation Study (Feature Importance)...")
    features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']]
    target = df['Disaster Type']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    X = features.fillna(0)
    
    # Baseline Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # Ablation Loop
    for col in X.columns:
        X_temp = X.drop(columns=[col])
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_temp, y, test_size=0.2, random_state=42)
        model_a = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model_a.fit(X_train_a, y_train_a)
        acc = accuracy_score(y_test_a, model_a.predict(X_test_a))
        drop = baseline_acc - acc
        print(f"Drop '{col}': Accuracy = {acc:.4f} (Impact: {drop:.4f})")

if __name__ == "__main__":
    file_path = "DISASTERS/disasters_1970_2021.csv"
    df = load_and_clean_data(file_path)
    calculate_vif(df)
    run_ablation_study(df)
