"""
03_modeling.py

Tahap training dan evaluasi model klasifikasi permeabilitas membran.

Fungsi utama:
1. Load data fitur hasil 02_feature_engineering.py
2. Train-test split dengan stratify agar distribusi kelas tetap proporsional
3. Training beberapa model klasifikasi
4. Evaluasi model dengan metrik umum dan metrik yang lebih adil untuk data imbalance
5. Memilih best model berdasarkan Macro-F1, Balanced Accuracy, dan ROC-AUC
6. Menyimpan best model, test set, hasil prediksi test set, confusion matrix, dan ROC curve

Output:
- models/best_model.pkl
- models/test_set.pkl
- models/best_model_metadata.json
- results/tables/model_comparison.csv
- results/tables/test_predictions_best_model.csv
- results/figures/model_comparison.png
- results/figures/confusion_matrix_best_model.png
- results/figures/roc_curve_best_model.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# Konfigurasi path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

INPUT_PATH = DATA_PROCESSED_DIR / "pampa_features.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
TEST_SET_PATH = MODELS_DIR / "test_set.pkl"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

MODEL_COMPARISON_PATH = RESULTS_TABLES_DIR / "model_comparison.csv"
TEST_PREDICTIONS_PATH = RESULTS_TABLES_DIR / "test_predictions_best_model.csv"
MODEL_COMPARISON_FIG_PATH = RESULTS_FIGURES_DIR / "model_comparison.png"
CONFUSION_MATRIX_FIG_PATH = RESULTS_FIGURES_DIR / "confusion_matrix_best_model.png"
ROC_CURVE_FIG_PATH = RESULTS_FIGURES_DIR / "roc_curve_best_model.png"

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

# Utility

def ensure_directories() -> None:
    """Membuat folder output jika belum tersedia."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Direktori output berhasil disiapkan.")


def load_and_split_data(
    filepath: Path = INPUT_PATH,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Membaca data fitur dan membagi data menjadi train-test set.

    Catatan:
    - Hanya FEATURE_COLUMNS yang digunakan sebagai input model.
    - Kolom SMILES tidak dipakai sebagai fitur, tetapi tetap disimpan di metadata test set
      agar hasil prediksi lebih mudah ditelusuri.
    - stratify=y digunakan karena dataset tidak seimbang antara kelas 0 dan 1.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"File data tidak ditemukan: {filepath}\n"
            "Pastikan 02_feature_engineering.py sudah dijalankan terlebih dahulu."
        )

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan: {missing_features}")

    # Ambil fitur numerik final dan target
    X = df[FEATURE_COLUMNS].copy()
    y = df[target_column].astype(int).copy()

    # Simpan metadata agar test predictions tetap informatif
    metadata_cols = [col for col in ["SMILES", "Drug_ID", "Drug"] if col in df.columns]
    metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

    # Pastikan tidak ada nilai kosong pada fitur/target
    valid_mask = X.notna().all(axis=1) & y.notna()
    if valid_mask.sum() < len(df):
        removed = len(df) - valid_mask.sum()
        print(f"[WARN] Menghapus {removed} baris karena ada nilai kosong pada fitur/target.")
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        metadata = metadata.loc[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    test_metadata = metadata.loc[X_test.index].copy()

    print(f"[INFO] Data berhasil dimuat dari: {filepath}")
    print(f"[INFO] Jumlah data: {len(X)}")
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("[INFO] Distribusi kelas total:")
    print(y.value_counts().sort_index().rename(index=CLASS_LABELS).to_string())
    print("[INFO] Distribusi kelas test:")
    print(y_test.value_counts().sort_index().rename(index=CLASS_LABELS).to_string())

    return X_train, X_test, y_train, y_test, test_metadata


def build_models() -> Dict[str, Any]:
    """
    Membuat daftar model yang akan dibandingkan.

    Tambahan penting:
    - Dummy Classifier dipakai sebagai baseline sederhana.
    - Model balanced ditambahkan untuk melihat apakah class_weight membantu kelas minoritas.
      Model balanced tidak otomatis dipilih; tetap dibandingkan berdasarkan metrik.
    """
    models: Dict[str, Any] = {
        "Dummy Most Frequent": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression Balanced": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "Random Forest Balanced": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ])
    else:
        models["Gradient Boosting"] = Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])
        print("[WARN] xgboost tidak tersedia. Menggunakan GradientBoostingClassifier sebagai pengganti.")

    return models


def get_probability_for_class_1(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Mengambil probabilitas untuk kelas 1 (High Permeability)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # Cari posisi kelas 1 agar aman jika urutan class berbeda
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
        class_1_index = list(classes).index(1)
        return proba[:, class_1_index]

    if hasattr(model, "decision_function"):
        # Fallback jika estimator tidak punya predict_proba
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    raise AttributeError("Model tidak memiliki predict_proba atau decision_function.")


def evaluate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Melatih dan mengevaluasi semua model."""
    results = {}
    trained_models = {}
    predictions_labels = {}
    predictions_proba = {}

    print("\n[START] Melatih dan mengevaluasi model...")

    for name, model in models.items():
        print(f"[INFO] Melatih {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = get_probability_for_class_1(model, X_test)

        trained_models[name] = model
        predictions_labels[name] = y_pred
        predictions_proba[name] = y_proba

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Recall Low/Moderate (0)": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Recall High (1)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }

    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values(
        by=["Macro-F1", "Balanced Accuracy", "ROC-AUC"],
        ascending=False,
    )

    return df_results, trained_models, predictions_labels, predictions_proba


def select_best_model(df_results: pd.DataFrame) -> str:
    """
    Memilih model terbaik.

    Alasan:
    - Dataset tidak seimbang, sehingga accuracy saja bisa menipu.
    - Macro-F1 memberi bobot lebih adil untuk kelas 0 dan kelas 1.
    - Balanced Accuracy membantu melihat performa rata-rata antar kelas.
    - ROC-AUC dipakai sebagai tie-breaker kemampuan separasi kelas.
    """
    best_model_name = df_results.sort_values(
        by=["Macro-F1", "Balanced Accuracy", "ROC-AUC"],
        ascending=False,
    ).index[0]

    print(
        "\n[BEST MODEL] Model terbaik dipilih berdasarkan "
        "Macro-F1, Balanced Accuracy, dan ROC-AUC:"
    )
    print(f"[BEST MODEL] {best_model_name}")
    return best_model_name


def save_test_predictions(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Menyimpan hasil prediksi best model pada test set."""
    output_df = test_metadata.reset_index().rename(columns={"index": "Original_Index"})
    output_df["True_Label"] = y_test.reset_index(drop=True).astype(int)
    output_df["Predicted_Label"] = y_pred.astype(int)
    output_df["Probability_Low_Moderate_Permeability"] = 1.0 - y_proba
    output_df["Probability_High_Permeability"] = y_proba
    output_df["True_Label_Name"] = output_df["True_Label"].map(CLASS_LABELS)
    output_df["Predicted_Label_Name"] = output_df["Predicted_Label"].map(CLASS_LABELS)

    output_df.to_csv(TEST_PREDICTIONS_PATH, index=False)
    print(f"[INFO] Prediksi test set disimpan: {TEST_PREDICTIONS_PATH}")
    return output_df


def plot_model_comparison(df_results: pd.DataFrame) -> None:
    """Membuat grafik perbandingan performa model untuk metrik utama."""
    metrics_to_plot = ["Accuracy", "Macro-F1", "Balanced Accuracy", "ROC-AUC"]
    plot_df = df_results[metrics_to_plot].copy()

    ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Perbandingan Performa Model Klasifikasi PAMPA", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Grafik perbandingan model disimpan: {MODEL_COMPARISON_FIG_PATH}")


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """Membuat dan menyimpan confusion matrix best model"""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(f"Confusion Matrix\nModel: {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low/Moderate (0)", "High (1)"])
    ax.set_yticklabels(["Low/Moderate (0)", "High (1)"])

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix disimpan: {CONFUSION_MATRIX_FIG_PATH}")


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str) -> float:
    """Membuat ROC curve dan mengembalikan nilai ROC-AUC."""
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2.0, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Best Model", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve disimpan: {ROC_CURVE_FIG_PATH}")

    return roc_auc


def save_best_model_metadata(best_model_name: str, best_metrics: pd.Series) -> None:
    """Menyimpan metadata best model agar bisa dipakai lagi oleh 04."""
    metadata = {
        "best_model_name": best_model_name,
        "selection_basis": ["Macro-F1", "Balanced Accuracy", "ROC-AUC"],
        "metrics": {key: float(value) for key, value in best_metrics.items()},
        "class_labels": CLASS_LABELS,
        "feature_columns": FEATURE_COLUMNS,
        "note": (
            "Best model dipilih dengan mempertimbangkan class imbalance. "
            "Accuracy tetap dilaporkan, tetapi bukan satu-satunya dasar pemilihan."
        ),
    }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"[INFO] Metadata best model disimpan: {BEST_MODEL_METADATA_PATH}")


def main() -> None:
    ensure_directories()

    X_train, X_test, y_train, y_test, test_metadata = load_and_split_data()
    models = build_models()

    df_results, trained_models, predictions_labels, predictions_proba = evaluate_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\n=== HASIL PERBANDINGAN MODEL ===")
    print(df_results.round(4).to_string())
    print("=================================\n")

    df_results.to_csv(MODEL_COMPARISON_PATH, index_label="Model")
    print(f"[INFO] Tabel perbandingan model disimpan: {MODEL_COMPARISON_PATH}")

    best_model_name = select_best_model(df_results)
    best_model = trained_models[best_model_name]
    best_y_pred = predictions_labels[best_model_name]
    best_y_proba = predictions_proba[best_model_name]

    joblib.dump(best_model, MODEL_PATH)
    print(f"[INFO] Best model disimpan: {MODEL_PATH}")

    # Test set disimpan agar 04 bisa melakukan interpretasi berbasis data test jika diperlukan.
    joblib.dump((X_test, y_test, test_metadata, FEATURE_COLUMNS), TEST_SET_PATH)
    print(f"[INFO] Test set disimpan: {TEST_SET_PATH}")

    save_best_model_metadata(best_model_name, df_results.loc[best_model_name])
    save_test_predictions(y_test, best_y_pred, best_y_proba, test_metadata)

    plot_model_comparison(df_results)
    plot_confusion_matrix(y_test, best_y_pred, best_model_name)
    plot_roc_curve(y_test, best_y_proba, best_model_name)

    print("\n[SUCCESS] Training dan evaluasi model selesai.")
    print("[INFO] Catatan: Karena dataset tidak seimbang, interpretasi performa sebaiknya melihat")
    print("       Macro-F1, Balanced Accuracy, Recall Low/Moderate, Confusion Matrix, dan ROC-AUC, bukan accuracy saja.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)