"""
01_data_preparation.py

Tahap awal pipeline:
1. Membaca dataset PAMPA_NCATS dari file lokal data/raw/pampa_ncats_raw.csv
2. Standarisasi kolom utama menjadi: Drug_ID, SMILES, Y
3. Cek missing value dan distribusi label
4. Simpan ringkasan dataset ke results/tables/dataset_summary.csv
5. Simpan grafik distribusi label ke results/figures/label_distribution.png

Catatan:
- File ini tidak melakukan validasi SMILES dengan RDKit.
- Validasi SMILES dan ekstraksi deskriptor dilakukan di 02_feature_engineering.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

RAW_DATA_PATH = DATA_RAW_DIR / "pampa_ncats_raw.csv"
DATASET_SUMMARY_PATH = RESULTS_TABLES_DIR / "dataset_summary.csv"
LABEL_DISTRIBUTION_PATH = RESULTS_FIGURES_DIR / "label_distribution.png"

DATASET_NAME = "PAMPA_NCATS"


def ensure_directories() -> None:
    """Membuat folder output jika belum tersedia."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_local_dataset() -> pd.DataFrame:
    """
    Membaca dataset PAMPA_NCATS dari file lokal.

    File yang dibutuhkan:
    data/raw/pampa_ncats_raw.csv
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset lokal tidak ditemukan di:\n{RAW_DATA_PATH}\n\n"
            "Pastikan file pampa_ncats_raw.csv sudah ada di folder data/raw/."
        )

    print(f"[INFO] Membaca dataset lokal: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi kolom dataset agar konsisten untuk tahap berikutnya.

    Format final:
    - Drug_ID
    - SMILES
    - Y
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    rename_map = {}

    if "Drug" in df.columns and "SMILES" not in df.columns:
        rename_map["Drug"] = "SMILES"

    if "smiles" in df.columns and "SMILES" not in df.columns:
        rename_map["smiles"] = "SMILES"

    if "label" in df.columns and "Y" not in df.columns:
        rename_map["label"] = "Y"

    if "target" in df.columns and "Y" not in df.columns:
        rename_map["target"] = "Y"

    df = df.rename(columns=rename_map)

    required_columns = ["SMILES", "Y"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_columns}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    if "Drug_ID" not in df.columns:
        df.insert(0, "Drug_ID", [f"mol_{i}" for i in range(len(df))])

    df = df[["Drug_ID", "SMILES", "Y"]]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    return df


def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data awal:
    - menghapus SMILES kosong
    - menghapus target kosong
    - memastikan target hanya 0 atau 1
    """
    initial_rows = len(df)

    df = df.dropna(subset=["SMILES", "Y"]).copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[df["SMILES"] != ""].copy()

    df = df[df["Y"].isin([0, 1])].copy()
    df["Y"] = df["Y"].astype(int)

    final_rows = len(df)
    removed_rows = initial_rows - final_rows

    print(f"[INFO] Jumlah data awal        : {initial_rows}")
    print(f"[INFO] Jumlah data setelah cek : {final_rows}")
    print(f"[INFO] Data dihapus           : {removed_rows}")

    return df.reset_index(drop=True)


def create_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat ringkasan dataset untuk notebook dan laporan."""
    label_counts = df["Y"].value_counts().sort_index()
    label_percentages = df["Y"].value_counts(normalize=True).sort_index() * 100

    summary_items = [
        ("dataset_name", DATASET_NAME),
        ("n_rows", len(df)),
        ("n_columns", df.shape[1]),
        ("n_unique_smiles", df["SMILES"].nunique()),
        ("n_duplicate_smiles", len(df) - df["SMILES"].nunique()),
        ("missing_drug_id", int(df["Drug_ID"].isna().sum())),
        ("missing_smiles", int(df["SMILES"].isna().sum())),
        ("missing_target", int(df["Y"].isna().sum())),
        ("class_0_count_low_moderate", int(label_counts.get(0, 0))),
        ("class_1_count_high", int(label_counts.get(1, 0))),
        ("class_0_percentage", round(float(label_percentages.get(0, 0)), 2)),
        ("class_1_percentage", round(float(label_percentages.get(1, 0)), 2)),
    ]

    return pd.DataFrame(summary_items, columns=["metric", "value"])


def save_label_distribution_plot(df: pd.DataFrame) -> None:
    """Membuat grafik distribusi label target."""
    label_counts = df["Y"].value_counts().sort_index()

    labels = ["Low/Moderate\nPermeability (0)", "High\nPermeability (1)"]
    values = [label_counts.get(0, 0), label_counts.get(1, 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Label Distribution of PAMPA_NCATS Dataset")
    plt.xlabel("Permeability Class")
    plt.ylabel("Number of Compounds")
    plt.tight_layout()
    plt.savefig(LABEL_DISTRIBUTION_PATH, dpi=300)
    plt.close()

    print(f"[INFO] Grafik distribusi label disimpan ke: {LABEL_DISTRIBUTION_PATH}")


def main() -> None:
    """Menjalankan seluruh tahap data preparation."""
    ensure_directories()

    df = load_local_dataset()
    df = standardize_columns(df)
    df = clean_initial_data(df)

    # Simpan ulang dataset mentah dalam format standar
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[INFO] Dataset standar disimpan ke: {RAW_DATA_PATH}")

    summary_df = create_dataset_summary(df)
    summary_df.to_csv(DATASET_SUMMARY_PATH, index=False)
    print(f"[INFO] Ringkasan dataset disimpan ke: {DATASET_SUMMARY_PATH}")

    save_label_distribution_plot(df)

    print("\n[INFO] Data preparation selesai.")

    print("\n[INFO] Preview data:")
    print(df.head())

    print("\n[INFO] Distribusi label:")
    print(df["Y"].value_counts().sort_index())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)