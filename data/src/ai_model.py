from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def train_ai_model(df):
    
    features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']]
    target = df['Disaster Type']


    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    X = features.fillna(0)

    value_counts = pd.Series(y).value_counts()
    valid_classes = value_counts[value_counts >= 6].index
    label_map = {old: new for new, old in enumerate(sorted(valid_classes))}
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = np.array([label_map[val] for val in y[mask]])  
    class_names = label_encoder.inverse_transform(list(valid_classes))


    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

  
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=500, max_depth=15, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=class_names))

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    plt.figure(figsize=(12, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    y_score = model.predict_proba(X_test)
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return model, model.predict(X), label_encoder
