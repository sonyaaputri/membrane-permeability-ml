import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def setup_directories():
    directories = [
        'models',
        'results/tables',
        'results/figures'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(" [INFO] Direktori output berhasil disiapkan.")

def load_and_split_data(filepath, target_column='Y', test_size=0.2, random_state=42):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File data tidak ditemukan di: {filepath}")
        
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f" [INFO] Data berhasil dimuat. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_models():
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(random_state=42, n_estimators=100))
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(random_state=42, probability=True))
        ]),
        'XGBoost': Pipeline([
            ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
    }
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}
    predictions_proba = {}
    predictions_labels = {}
    
    print("\n [START] Melatih dan mengevaluasi model...")
    for name, model in models.items():
        print(f"   Melatih {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model
        predictions_labels[name] = y_pred
        predictions_proba[name] = y_proba
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        
    df_results = pd.DataFrame(results).T
    return df_results, trained_models, predictions_labels, predictions_proba

def save_comparison_visualization(df_results):
    """Membuat visualisasi perbandingan performa semua model."""
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    df_metrics = df_results.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    df_metrics.rename(columns={'index': 'Model'}, inplace=True)
    sns.barplot(data=df_metrics, x='Model', y='Value', hue='Metric', palette='Set2')
    plt.title('Perbandingan Performa Model Klasifikasi PAMPA', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel('Skor')
    plt.tight_layout()
    plt.savefig('results/figures/model_comparison.png', dpi=300)
    plt.close()
    
    print(" [INFO] Visualisasi perbandingan model berhasil disimpan di 'results/figures/model_comparison.png'.")

def main():
    setup_directories()
    
    TARGET_COLUMN = 'Y' 
    INPUT_PATH = 'data/processed/pampa_features.csv'
    
    try:
        X_train, X_test, y_train, y_test = load_and_split_data(INPUT_PATH, target_column=TARGET_COLUMN)
    except Exception as e:
        print(f" [ERROR] Gagal memproses data: {e}")
        return

    models = build_models()
    
    df_results, trained_models, predictions_labels, predictions_proba = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )
    
    print("\n=== HASIL PERBANDINGAN MODEL ===")
    print(df_results.round(4))
    print("=================================\n")
    
    df_results.to_csv('results/tables/model_comparison.csv', index_label='Model')
    
    best_model_name = df_results['F1-score'].idxmax()
    print(f" [BEST MODEL] Model terbaik berdasarkan F1-Score adalah: **{best_model_name}**")
    
    best_pipeline = trained_models[best_model_name]
    joblib.dump(best_pipeline, 'models/best_model.pkl')
    print(f" [INFO] Model '{best_model_name}' berhasil disimpan ke 'models/best_model.pkl'.")
    
    joblib.dump((X_test, y_test), 'models/test_set.pkl')
    print(f" [INFO] Test set berhasil disimpan ke 'models/test_set.pkl'.")
    
    save_comparison_visualization(df_results)
    
    print("\n [SUCCESS] Training dan evaluasi model selesai dengan sukses!")

if __name__ == '__main__':
    main()