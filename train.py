import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_and_save_model():
    data_path = os.path.join('data', 'creditcard.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' ফাইলটি পাওয়া যায়নি। অনুগ্রহ করে ডেটাসেটটি 'data/' ফোল্ডারে রাখুন।")

    print("LOAD DATASET...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    hybrid_model = VotingClassifier(
        estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft'
    )

    print("TRAINING HYBRID MODEL...")
    hybrid_model.fit(X_train_scaled, y_train)

    y_pred_proba = hybrid_model.predict_proba(X_test_scaled)[:, 1]
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(hybrid_model, os.path.join('models', 'hybrid_model.pkl'))
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    print("MODEL SAVED SUCCESSFULLY IN 'models/' DIRECTORY!")

if __name__ == '__main__':
    train_and_save_model()