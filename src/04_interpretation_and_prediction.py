"""
04_interpretation_and_prediction.py

Tahap interpretasi model terbaik dan demo prediksi SMILES baru.

Fungsi utama:
1. Load best model hasil 03_modeling.py
2. Load data fitur hasil 02_feature_engineering.py
3. Menghitung feature importance secara valid:
   - coef_ untuk Logistic Regression
   - feature_importances_ untuk model tree-based seperti Random Forest/XGBoost/Gradient Boosting
   - permutation importance untuk model yang tidak punya importance bawaan, seperti SVM
4. Membuat grafik feature importance
5. Membuat demo prediksi dari SMILES baru
6. Menyimpan hasil feature importance dan demo prediction

Output:
- results/tables/feature_importance.csv
- results/figures/feature_importance.png
- results/tables/demo_predictions.csv

Catatan:
- Evaluasi utama model, confusion matrix, ROC curve, dan test predictions dibuat di 03_modeling.py.
- File ini fokus pada interpretasi biologis dan prediksi molekul baru.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# RDKit import

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
except ImportError as error:
    print(f"[ERROR] RDKit tidak terinstall: {error}")
    print("Install RDKit dengan salah satu cara berikut:")
    print("  pip install rdkit")
    print("  conda install -c conda-forge rdkit")
    sys.exit(1)


# Konfigurasi path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FEATURES_PATH = DATA_PROCESSED_DIR / "pampa_features.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
TEST_SET_PATH = MODELS_DIR / "test_set.pkl"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

FEATURE_IMPORTANCE_TABLE_PATH = RESULTS_TABLES_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_FIGURE_PATH = RESULTS_FIGURES_DIR / "feature_importance.png"
DEMO_PREDICTIONS_PATH = RESULTS_TABLES_DIR / "demo_predictions.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "Y"

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

CLASS_LABELS = {
    0: "Low/Moderate Permeability",
    1: "High Permeability",
}

FEATURE_DESCRIPTIONS = {
    "MolWt": (
        "Berat molekul. Molekul yang lebih besar cenderung lebih sulit berdifusi "
        "melewati lipid bilayer."
    ),
    "LogP": (
        "Lipofilisitas/hidrofobisitas. Nilai LogP yang lebih tinggi menunjukkan molekul "
        "lebih mudah berinteraksi dengan bagian hidrofobik lipid bilayer."
    ),
    "TPSA": (
        "Topological Polar Surface Area. Nilai TPSA yang lebih tinggi menunjukkan polaritas "
        "lebih besar sehingga molekul cenderung lebih sulit melewati inti hidrofobik membran."
    ),
    "HBD": (
        "Hydrogen bond donor. Jumlah donor ikatan hidrogen yang lebih besar dapat meningkatkan "
        "interaksi polar dengan air sehingga permeabilitas pasif dapat menurun."
    ),
    "HBA": (
        "Hydrogen bond acceptor. Jumlah akseptor ikatan hidrogen yang lebih besar dapat meningkatkan "
        "karakter polar molekul sehingga permeabilitas pasif dapat menurun."
    ),
    "RotatableBonds": (
        "Jumlah ikatan yang dapat berotasi. Fitur ini menggambarkan fleksibilitas molekul yang "
        "dapat memengaruhi bentuk dan kemampuan difusi."
    ),
    "HeavyAtomCount": (
        "Jumlah atom non-hidrogen. Fitur ini berkaitan dengan ukuran dan kompleksitas molekul."
    ),
    "RingCount": (
        "Jumlah cincin dalam struktur molekul. Fitur ini berkaitan dengan bentuk, rigiditas, "
        "dan karakter struktur molekul."
    ),
    "FormalCharge": (
        "Muatan formal molekul. Molekul bermuatan umumnya lebih sulit melewati lipid bilayer "
        "secara pasif tanpa bantuan protein transport."
    ),
}


# Utility

def ensure_directories() -> None:
    """Membuat folder output jika belum tersedia."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Direktori output berhasil disiapkan.")


def load_best_model() -> Any:
    """Load best model yang sudah disimpan oleh 03_modeling.py."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model terbaik tidak ditemukan: {BEST_MODEL_PATH}\n"
            "Jalankan 03_modeling.py terlebih dahulu."
        )

    model = joblib.load(BEST_MODEL_PATH)
    print(f"[INFO] Best model dimuat: {BEST_MODEL_PATH}")
    return model


def load_best_model_metadata() -> dict:
    """Load metadata best model jika tersedia."""
    if not BEST_MODEL_METADATA_PATH.exists():
        print("[WARN] Metadata best model tidak ditemukan. Nama model akan ditulis sebagai 'Best Model'.")
        return {"best_model_name": "Best Model"}

    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print(f"[INFO] Metadata best model dimuat: {BEST_MODEL_METADATA_PATH}")
    return metadata


def load_features_data() -> pd.DataFrame:
    """Load dataset fitur hasil feature engineering."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset fitur tidak ditemukan: {FEATURES_PATH}\n"
            "Jalankan 02_feature_engineering.py terlebih dahulu."
        )

    df = pd.read_csv(FEATURES_PATH)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan: {missing_features}")

    print(f"[INFO] Data fitur dimuat: {FEATURES_PATH}")
    print(f"[INFO] Shape data fitur: {df.shape}")
    return df


def load_reference_data_for_importance(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Mengambil data referensi untuk permutation importance.

    Prioritas:
    1. Test set dari models/test_set.pkl hasil 03_modeling.py
    2. Jika tidak ada, fallback ke seluruh data fitur yang memiliki target Y

    Catatan:
    Permutation importance lebih baik dihitung pada test set agar interpretasi tidak terlalu bias
    terhadap data training.
    """
    if TEST_SET_PATH.exists():
        loaded = joblib.load(TEST_SET_PATH)

        # Format baru dari 03 final: (X_test, y_test, test_metadata, feature_columns)
        if isinstance(loaded, tuple) and len(loaded) >= 2:
            X_ref = loaded[0]
            y_ref = loaded[1]

            if not isinstance(X_ref, pd.DataFrame):
                X_ref = pd.DataFrame(X_ref, columns=FEATURE_COLUMNS)
            else:
                X_ref = X_ref[FEATURE_COLUMNS]

            y_ref = pd.Series(y_ref).astype(int)
            print(f"[INFO] Data referensi importance memakai test set: {X_ref.shape}")
            return X_ref, y_ref

    if TARGET_COLUMN in features_df.columns:
        print("[WARN] Test set tidak ditemukan. Permutation importance memakai seluruh data fitur sebagai fallback.")
        X_ref = features_df[FEATURE_COLUMNS].copy()
        y_ref = features_df[TARGET_COLUMN].astype(int).copy()
        return X_ref, y_ref

    print("[WARN] Data target tidak tersedia. Permutation importance tidak dapat dihitung.")
    return features_df[FEATURE_COLUMNS].copy(), None


def get_classifier_from_pipeline(model: Any) -> Any:
    """
    Mengambil estimator/classifier utama dari pipeline.

    Pada 03_modeling.py, step classifier dinamai 'clf'.
    Jika struktur pipeline berubah, estimator terakhir tetap digunakan sebagai fallback.
    """
    if hasattr(model, "named_steps"):
        if "clf" in model.named_steps:
            return model.named_steps["clf"]
        return model.steps[-1][1]
    return model


def normalize_importance(values: np.ndarray) -> np.ndarray:
    """Normalisasi importance ke rentang 0-1."""
    values = np.asarray(values, dtype=float)

    # Jika ada nilai negatif, gunakan nilai absolut karena yang dicari besar pengaruhnya.
    values = np.abs(values)

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if np.isclose(max_value, min_value):
        return np.ones_like(values)

    return (values - min_value) / (max_value - min_value)



# Feature importance

def extract_feature_importance(
    model: Any,
    X_reference: pd.DataFrame,
    y_reference: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Menghitung feature importance dari best model secara valid.

    Metode:
    - Logistic Regression: absolute coefficient
    - Random Forest/XGBoost/Gradient Boosting: feature_importances_
    - SVM atau model lain tanpa importance bawaan: permutation importance
    """
    classifier = get_classifier_from_pipeline(model)
    classifier_name = type(classifier).__name__

    print(f"[INFO] Mengekstraksi feature importance dari model: {classifier_name}")

    raw_importance = None
    signed_effect = None
    importance_source = None

    if hasattr(classifier, "coef_"):
        coef = np.ravel(classifier.coef_)
        raw_importance = np.abs(coef)
        signed_effect = coef
        importance_source = "absolute_coefficient"
        print("[INFO] Importance menggunakan absolute coefficient.")

    elif hasattr(classifier, "feature_importances_"):
        raw_importance = np.asarray(classifier.feature_importances_)
        signed_effect = np.full(len(FEATURE_COLUMNS), np.nan)
        importance_source = "model_feature_importances"
        print("[INFO] Importance menggunakan feature_importances_ bawaan model.")

    elif y_reference is not None:
        print("[INFO] Model tidak memiliki importance bawaan. Menggunakan permutation importance.")
        permutation_result = permutation_importance(
            estimator=model,
            X=X_reference[FEATURE_COLUMNS],
            y=y_reference,
            scoring="f1_macro",
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        raw_importance = permutation_result.importances_mean
        signed_effect = np.full(len(FEATURE_COLUMNS), np.nan)
        importance_source = "permutation_importance_f1_macro"

    else:
        raise ValueError(
            "Feature importance tidak dapat dihitung karena model tidak memiliki importance bawaan "
            "dan data target untuk permutation importance tidak tersedia."
        )

    if len(raw_importance) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Jumlah importance ({len(raw_importance)}) tidak sesuai jumlah fitur ({len(FEATURE_COLUMNS)})."
        )

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Raw_Importance": raw_importance,
            "Importance": normalize_importance(raw_importance),
            "Signed_Effect": signed_effect,
            "Importance_Source": importance_source,
            "Biological_Description": [FEATURE_DESCRIPTIONS[feature] for feature in FEATURE_COLUMNS],
        }
    )

    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print("[INFO] Feature importance berhasil dihitung.")
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """Membuat grafik feature importance."""
    sorted_df = importance_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_df["Feature"], sorted_df["Importance"])
    ax.set_xlabel("Importance Score (Normalized)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance untuk Prediksi Permeabilitas Membran", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Grafik feature importance disimpan: {FEATURE_IMPORTANCE_FIGURE_PATH}")


def save_feature_importance(importance_df: pd.DataFrame) -> None:
    """Menyimpan tabel feature importance."""
    importance_df.to_csv(FEATURE_IMPORTANCE_TABLE_PATH, index=False)
    print(f"[INFO] Tabel feature importance disimpan: {FEATURE_IMPORTANCE_TABLE_PATH}")


# SMILES prediction

def validate_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Validasi SMILES dan mengubahnya menjadi objek RDKit Mol."""
    if smiles is None or str(smiles).strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return mol
    except Exception:
        return None


def compute_descriptors(mol: Chem.Mol) -> dict:
    """Menghitung deskriptor fisikokimia yang sama dengan 02_feature_engineering.py."""
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


def predict_proba_class_1(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Mengambil probabilitas kelas 1 / High Permeability."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model tidak mendukung predict_proba.")

    proba = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "classes_"):
        classes = model.named_steps["clf"].classes_
    else:
        classes = np.array([0, 1])

    class_1_index = list(classes).index(1)
    return proba[:, class_1_index]


def predict_new_smiles(smiles_data: list[dict], model: Any) -> pd.DataFrame:
    """
    Melakukan prediksi permeabilitas pada SMILES baru.

    Parameter smiles_data berisi list dict:
    [
        {"Molecule": "Aspirin", "SMILES": "..."},
        ...
    ]
    """
    prediction_rows = []

    print(f"[INFO] Melakukan prediksi pada {len(smiles_data)} SMILES baru...")

    for item in smiles_data:
        molecule_name = item.get("Molecule", "Unknown Molecule")
        smiles = item.get("SMILES", "")

        mol = validate_smiles(smiles)
        if mol is None:
            prediction_rows.append(
                {
                    "Molecule": molecule_name,
                    "SMILES": smiles,
                    "Status": "Invalid SMILES",
                    "Prediction": np.nan,
                    "Prediction_Label": "Invalid SMILES",
                    "Probability_High_Permeability": np.nan,
                    "Probability_Low_Moderate_Permeability": np.nan,
                }
            )
            print(f"[WARN] SMILES invalid: {molecule_name} | {smiles}")
            continue

        descriptors = compute_descriptors(mol)
        X_new = pd.DataFrame([descriptors], columns=FEATURE_COLUMNS)

        pred_class = int(model.predict(X_new)[0])
        prob_high = float(predict_proba_class_1(model, X_new)[0])
        prob_low = 1.0 - prob_high

        prediction_row = {
            "Molecule": molecule_name,
            "SMILES": smiles,
            "Status": "Valid",
            **descriptors,
            "Prediction": pred_class,
            "Prediction_Label": CLASS_LABELS[pred_class],
            "Probability_High_Permeability": round(prob_high, 4),
            "Probability_Low_Moderate_Permeability": round(prob_low, 4),
        }
        prediction_rows.append(prediction_row)

    predictions_df = pd.DataFrame(prediction_rows)
    print("[INFO] Prediksi SMILES baru selesai.")
    return predictions_df


def save_demo_predictions(predictions_df: pd.DataFrame) -> None:
    """Menyimpan hasil demo prediksi."""
    predictions_df.to_csv(DEMO_PREDICTIONS_PATH, index=False)
    print(f"[INFO] Demo predictions disimpan: {DEMO_PREDICTIONS_PATH}")


def print_biological_interpretation(importance_df: pd.DataFrame, top_n: int = 3) -> None:
    """Menampilkan ringkasan interpretasi biologis fitur terpenting."""
    print("\n[INFO] Interpretasi biologis fitur terpenting:")

    for rank, row in importance_df.head(top_n).iterrows():
        feature = row["Feature"]
        importance = row["Importance"]
        description = row["Biological_Description"]

        print(f"{rank + 1}. {feature} (normalized importance = {importance:.4f})")
        print(f"   {description}")

        if not pd.isna(row.get("Signed_Effect", np.nan)):
            signed_effect = row["Signed_Effect"]
            direction = "meningkatkan" if signed_effect > 0 else "menurunkan"
            print(
                f"   Pada model linear, koefisien fitur ini bernilai {signed_effect:.4f}, "
                f"sehingga kenaikan fitur cenderung {direction} peluang prediksi kelas High Permeability."
            )



# Main

def main() -> None:
    ensure_directories()

    model = load_best_model()
    metadata = load_best_model_metadata()
    features_df = load_features_data()

    best_model_name = metadata.get("best_model_name", "Best Model")
    print(f"[INFO] Nama best model: {best_model_name}")

    X_reference, y_reference = load_reference_data_for_importance(features_df)

    importance_df = extract_feature_importance(
        model=model,
        X_reference=X_reference,
        y_reference=y_reference,
    )

    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(
        importance_df[
            ["Feature", "Raw_Importance", "Importance", "Importance_Source"]
        ].head(10).round(6).to_string(index=False)
    )
    print("==============================\n")

    save_feature_importance(importance_df)
    plot_feature_importance(importance_df)
    print_biological_interpretation(importance_df, top_n=3)

    # Contoh SMILES baru untuk demo. Ini bukan random dari dataset.
    demo_smiles = [
        {
            "Molecule": "Aspirin",
            "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
        },
        {
            "Molecule": "Caffeine",
            "SMILES": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        },
        {
            "Molecule": "Ibuprofen",
            "SMILES": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        },
        {
            "Molecule": "Ethanol",
            "SMILES": "CCO",
        },
        {
            "Molecule": "Glucose",
            "SMILES": "C(C1C(C(C(C(O1)O)O)O)O)O",
        },
    ]

    print("\n[INFO] === Demo Prediksi SMILES Baru ===")
    demo_predictions_df = predict_new_smiles(demo_smiles, model)

    columns_to_show = [
        "Molecule",
        "SMILES",
        "Prediction_Label",
        "Probability_High_Permeability",
        "Probability_Low_Moderate_Permeability",
    ]
    print(demo_predictions_df[columns_to_show].to_string(index=False))

    save_demo_predictions(demo_predictions_df)

    print("\n[SUCCESS] Interpretasi model dan demo prediksi selesai.")
    print("[INFO] Evaluasi performa utama, confusion matrix, dan ROC curve berada di output 03_modeling.py.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)