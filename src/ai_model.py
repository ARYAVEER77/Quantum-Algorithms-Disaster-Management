from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def train_ai_model(df):
    features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 
                   'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']]
    target = df['Disaster Type']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    X = features.fillna(0)
    
    # Balance data
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )
    
    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, model.predict(X), label_encoder
