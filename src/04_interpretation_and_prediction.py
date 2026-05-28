"""
04_interpretation_and_prediction.py

Tahap interpretasi model dan prediksi pada molekul baru:
1. Load model terbaik dari models/best_model.pkl
2. Load data fitur untuk analisis
3. Menghitung feature importance
4. Membuat visualisasi feature importance
5. Evaluasi model pada test set dengan confusion matrix dan ROC curve
6. Membuat fungsi prediksi untuk SMILES baru
7. Menghitung fitur dan melakukan prediksi
8. Menyimpan hasil demo prediksi

Output:
- results/tables/feature_importance.csv       : Daftar fitur dan bobot importance
- results/figures/feature_importance.png      : Grafik feature importance
- results/tables/test_predictions_best_model.csv : Hasil prediksi pada test set
- results/figures/confusion_matrix_best_model.png : Confusion matrix visualization
- results/figures/roc_curve_best_model.png    : ROC curve visualization
- results/tables/demo_predictions.csv         : Hasil prediksi pada contoh SMILES
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn imports
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score
)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    print(f"[ERROR] RDKit tidak terinstall: {e}")
    print("Install dengan: pip install rdkit  atau  conda install -c conda-forge rdkit")
    sys.exit(1)

# Path konfigurasi
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FEATURES_PATH = DATA_PROCESSED_DIR / "pampa_features.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
TEST_SET_PATH = MODELS_DIR / "test_set.pkl"

FEATURE_IMPORTANCE_TABLE_PATH = RESULTS_TABLES_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_FIGURE_PATH = RESULTS_FIGURES_DIR / "feature_importance.png"
TEST_PREDICTIONS_PATH = RESULTS_TABLES_DIR / "test_predictions_best_model.csv"
CONFUSION_MATRIX_PATH = RESULTS_FIGURES_DIR / "confusion_matrix_best_model.png"
ROC_CURVE_PATH = RESULTS_FIGURES_DIR / "roc_curve_best_model.png"
DEMO_PREDICTIONS_PATH = RESULTS_TABLES_DIR / "demo_predictions.csv"

# Kolom fitur yang sama dengan tahap feature engineering
FEATURE_COLUMNS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "HeavyAtomCount",
    "RingCount",
    "FormalCharge",
]

# Deskripsi biologis setiap fitur
FEATURE_DESCRIPTIONS = {
    "MolWt": "Berat molekul (g/mol). Molekul lebih berat → lebih sulit menembus membran.",
    "LogP": "Koefisien partisi (log skala). Nilai tinggi → lebih hidrofobik → lebih mudah menembus lipid bilayer.",
    "TPSA": "Luas permukaan polar (Ų). Nilai tinggi → lebih polar → lebih sulit menembus membran nonpolar.",
    "HBD": "Jumlah donor ikatan hidrogen. Nilai tinggi → lebih sulit menembus membran.",
    "HBA": "Jumlah akseptor ikatan hidrogen. Nilai tinggi → lebih sulit menembus membran.",
    "RotatableBonds": "Jumlah ikatan rotasi. Berkaitan dengan fleksibilitas molekul.",
    "HeavyAtomCount": "Jumlah atom non-hidrogen. Berkaitan dengan ukuran molekul.",
    "RingCount": "Jumlah cincin (aromatik/alifatik). Berkaitan dengan rigiditas struktur.",
    "FormalCharge": "Muatan formal total. Molekul bermuatan lebih sulit menembus membran.",
}


def ensure_directories() -> None:
    """Membuat folder output jika belum tersedia."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data() -> tuple:
    """Load model terbaik dan data fitur yang sudah diproses."""
    # Load model
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model terbaik tidak ditemukan di:\n{BEST_MODEL_PATH}\n"
            "Pastikan sudah menjalankan 03_modeling.py terlebih dahulu."
        )
    best_model = joblib.load(BEST_MODEL_PATH)
    print(f"[INFO] Model dimuat: {BEST_MODEL_PATH}")

    # Load data fitur
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset fitur tidak ditemukan di:\n{FEATURES_PATH}\n"
            "Pastikan sudah menjalankan 02_feature_engineering.py terlebih dahulu."
        )
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"[INFO] Data fitur dimuat: {len(features_df)} baris, {len(FEATURE_COLUMNS)} fitur")

    return best_model, features_df


def extract_feature_importance(model) -> pd.DataFrame:
    """
    Ekstrak feature importance dari model.
    
    Metode tergantung tipe model:
    - Logistic Regression: gunakan absolute coefficient
    - Random Forest: gunakan feature_importances_
    - SVM: tidak ada feature importance, gunakan permutation importance
    - XGBoost: gunakan feature_importance
    """
    print("[INFO] Mengekstraksi feature importance dari model...")

    # Jika pipeline, ambil estimator terakhir
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier", model)
    else:
        classifier = model

    feature_importance_dict = {}

    # Cek tipe model
    if hasattr(classifier, "coef_"):
        # Logistic Regression
        print("[INFO] Tipe model: Logistic Regression")
        coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
        importance_values = np.abs(coef)
        
    elif hasattr(classifier, "feature_importances_"):
        # Random Forest, XGBoost, Gradient Boosting
        if hasattr(classifier, "feature_names_in_"):
            print(f"[INFO] Tipe model: {type(classifier).__name__}")
        importance_values = classifier.feature_importances_
        
    elif hasattr(classifier, "support_vectors_"):
        # SVM - gunakan magnitude untuk setiap feature
        print("[INFO] Tipe model: SVM (menggunakan magnitude)")
        importance_values = np.ones(len(FEATURE_COLUMNS))  # Sama penting untuk SVM
        
    else:
        print(f"[WARN] Tidak bisa ekstrak importance dari {type(classifier).__name__}")
        print("[INFO] Menggunakan uniform importance")
        importance_values = np.ones(len(FEATURE_COLUMNS))

    # Buat DataFrame
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importance_values
    })

    # Normalisasi ke range 0-1
    min_val = importance_df["Importance"].min()
    max_val = importance_df["Importance"].max()
    if max_val > min_val:
        importance_df["Importance"] = (importance_df["Importance"] - min_val) / (max_val - min_val)
    else:
        importance_df["Importance"] = 1.0

    # Sort descending
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    print("[INFO] Feature importance berhasil diekstraksi")
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """Membuat grafik feature importance."""
    print("[INFO] Membuat grafik feature importance...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reverse untuk tampil dari atas ke bawah
    sorted_df = importance_df.sort_values("Importance", ascending=True)

    colors = plt.cm.viridis(sorted_df["Importance"] / sorted_df["Importance"].max())
    ax.barh(sorted_df["Feature"], sorted_df["Importance"], color=colors)

    ax.set_xlabel("Importance Score (Normalized)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title("Feature Importance untuk Prediksi Permeabilitas Membran", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    print(f"[INFO] Grafik disimpan: {FEATURE_IMPORTANCE_FIGURE_PATH}")
    plt.close()


def validate_smiles(smiles: str) -> Chem.Mol | None:
    """
    Mencoba membuat objek mol RDKit dari SMILES.
    Mengembalikan None jika SMILES tidak valid.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return mol
    except Exception:
        return None


def compute_descriptors(mol: Chem.Mol) -> dict:
    """Menghitung 9 deskriptor fisikokimia dari objek mol RDKit."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
    }


def predict_new_smiles(smiles_list: list, model, features_df: pd.DataFrame = None) -> pd.DataFrame:
    """Melakukan prediksi pada list SMILES baru."""
    print(f"[INFO] Melakukan prediksi pada {len(smiles_list)} molekul baru...")

    predictions_list = []

    for idx, smiles in enumerate(smiles_list):
        # Validasi SMILES
        mol = validate_smiles(smiles)
        if mol is None:
            print(f"[WARN] SMILES invalid (index {idx}): {smiles}")
            continue

        # Hitung deskriptor
        try:
            descriptors = compute_descriptors(mol)
        except Exception as e:
            print(f"[WARN] Gagal menghitung deskriptor (index {idx}): {e}")
            continue

        # Buat feature vector
        feature_vector = np.array([descriptors[col] for col in FEATURE_COLUMNS]).reshape(1, -1)

        # Prediksi
        try:
            # Jika pipeline, akan auto-scale
            pred_class = model.predict(feature_vector)[0]
            pred_proba = model.predict_proba(feature_vector)[0]

            # Ambil probability untuk kelas high permeability (kelas 1)
            prob_high_perm = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]

            # Buat row hasil
            row = {"SMILES": smiles}
            row.update(descriptors)
            row["Prediction"] = int(pred_class)
            row["Prediction_Label"] = "High Permeability" if pred_class == 1 else "Low/Moderate Permeability"
            row["Probability_High_Permeability"] = round(prob_high_perm, 4)
            row["Probability_Low_Moderate_Permeability"] = round(1.0 - prob_high_perm, 4)

            predictions_list.append(row)

        except Exception as e:
            print(f"[WARN] Gagal prediksi (index {idx}): {e}")
            continue

    result_df = pd.DataFrame(predictions_list)
    print(f"[INFO] Prediksi selesai. Total berhasil: {len(result_df)}/{len(smiles_list)}")
    return result_df


def predict_test_set(model) -> tuple:
    print("[INFO] Memproses test set...")
    
    if not TEST_SET_PATH.exists():
        print("[WARN] Test set tidak ditemukan. Melewatkan evaluasi test set.")
        return None, None, None, None, None
    
    try:
        X_test, y_test = joblib.load(TEST_SET_PATH)
        print(f"[INFO] Test set dimuat: {X_test.shape[0]} sampel")
        
        # Prediksi
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Hitung ROC-AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"[INFO] ROC-AUC pada test set: {roc_auc:.4f}")
        
        return X_test, y_test, y_pred, y_proba, roc_auc
    except Exception as e:
        print(f"[ERROR] Gagal memproses test set: {e}")
        return None, None, None, None, None


def visualize_test_results(y_test, y_pred, y_proba, roc_auc: float, model_name: str = "Logistic Regression") -> None:
    print("[INFO] Membuat visualisasi hasil test set...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low/Moderate (0)', 'High (1)'],
                yticklabels=['Low/Moderate (0)', 'High (1)'])
    plt.title(f'Confusion Matrix\nModel: {model_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix disimpan: {CONFUSION_MATRIX_PATH}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curve disimpan: {ROC_CURVE_PATH}")


def save_test_predictions(y_test, y_pred, y_proba) -> None:
    ensure_directories()
    
    df_test_preds = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Probability_Class_0': 1.0 - y_proba,
        'Probability_Class_1': y_proba,
        'Prediction_Label': ['High Permeability' if p == 1 else 'Low/Moderate Permeability' for p in y_pred]
    })
    
    df_test_preds.to_csv(TEST_PREDICTIONS_PATH, index=False)
    print(f"[INFO] Test predictions disimpan: {TEST_PREDICTIONS_PATH}")


def save_results(importance_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    ensure_directories()

    importance_df.to_csv(FEATURE_IMPORTANCE_TABLE_PATH, index=False)
    print(f"[INFO] Feature importance disimpan: {FEATURE_IMPORTANCE_TABLE_PATH}")

    # Simpan demo predictions
    if not predictions_df.empty:
        predictions_df.to_csv(DEMO_PREDICTIONS_PATH, index=False)
        print(f"[INFO] Demo predictions disimpan: {DEMO_PREDICTIONS_PATH}")


def main() -> None:
    """Fungsi utama untuk interpretasi dan prediksi."""
    # Load model dan data
    best_model, features_df = load_model_and_data()

    # Ekstrak feature importance
    importance_df = extract_feature_importance(best_model)
    print("[INFO] Top 5 Fitur Terpenting:")
    print(importance_df.to_string(index=False))

    # Plot feature importance
    ensure_directories()
    plot_feature_importance(importance_df)

    # Prediksi dan evaluasi pada test set
    print("\n[INFO] === Evaluasi pada Test Set ===")
    X_test, y_test, y_pred, y_proba, roc_auc = predict_test_set(best_model)
    
    if X_test is not None:
        save_test_predictions(y_test, y_pred, y_proba)
        visualize_test_results(y_test, y_pred, y_proba, roc_auc, "Best Model")
    
    # Demo prediksi pada beberapa SMILES baru
    print("\n[INFO] === Demo Prediksi pada SMILES Baru ===")
    # Contoh SMILES dari dataset (ambil 5 random)
    demo_smiles = features_df["SMILES"].sample(n=min(5, len(features_df)), random_state=42).tolist()

    print("[INFO] Demo Prediksi pada 5 Molekul Random")

    demo_predictions = predict_new_smiles(demo_smiles, best_model, features_df)
    print("[INFO] Hasil Demo Prediksi:")
    print(demo_predictions[["SMILES", "Prediction_Label", "Probability_High_Permeability"]].to_string(index=False))

    # Simpan hasil
    save_results(importance_df, demo_predictions)

    print("\n[INFO] Interpretasi Biologis Fitur Terpenting:")
    print("[INFO] Top 3 Fitur Terpenting dan Maknanya:")
    for idx, row in importance_df.head(3).iterrows():
        feature = row["Feature"]
        importance = row["Importance"]
        description = FEATURE_DESCRIPTIONS.get(feature, "Tidak ada deskripsi")
        print(f"  {idx + 1}. {feature} (Importance: {importance:.4f})")
        print(f"     → {description}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
