"""
02_feature_engineering.py

Tahap feature engineering pipeline:
1. Membaca dataset mentah dari data/raw/pampa_ncats_raw.csv
2. Validasi SMILES menggunakan RDKit (menghapus SMILES invalid)
3. Menghitung deskriptor molekuler fisikokimia
4. Menyimpan dataset fitur ke data/processed/pampa_features.csv
5. Menyimpan ringkasan statistik deskriptor ke results/tables/descriptor_summary.csv
6. Menyimpan grafik distribusi fitur ke results/figures/descriptor_distribution.png

Fitur yang dihitung:
- MolWt          : Molecular Weight
- LogP           : Lipophilicity (Wildman-Crippen)
- TPSA           : Topological Polar Surface Area
- HBD            : Hydrogen Bond Donor count
- HBA            : Hydrogen Bond Acceptor count
- RotatableBonds : Rotatable Bond count
- HeavyAtomCount : Heavy Atom Count
- RingCount      : Ring Count
- FormalCharge   : Net Formal Charge

Deskriptor ini dipilih karena berhubungan langsung dengan kemampuan
molekul menembus membran lipid bilayer (Lipinski Rule of Five dan ADME).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit import RDLogger
    # Matikan warning RDKit agar output bersih
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    print(f"[ERROR] RDKit tidak terinstall: {e}")
    print("Install dengan: pip install rdkit  atau  conda install -c conda-forge rdkit")
    sys.exit(1)

# Path konfigurasi
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

RAW_DATA_PATH          = DATA_RAW_DIR / "pampa_ncats_raw.csv"
FEATURES_PATH          = DATA_PROCESSED_DIR / "pampa_features.csv"
DESCRIPTOR_SUMMARY_PATH = RESULTS_TABLES_DIR / "descriptor_summary.csv"
DESCRIPTOR_DIST_PATH    = RESULTS_FIGURES_DIR / "descriptor_distribution.png"

# Kolom fitur yang akan dihitung
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

# Helper functions
def ensure_directories() -> None:
    """Membuat folder output jika belum tersedia."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataset() -> pd.DataFrame:
    """Membaca dataset mentah dari file lokal."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset mentah tidak ditemukan di:\n{RAW_DATA_PATH}\n"
            "Pastikan sudah menjalankan 01_data_preparation.py terlebih dahulu."
        )
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Dataset dimuat: {len(df)} baris, kolom: {list(df.columns)}")
    return df

# Validasi SMILES
def validate_smiles(smiles: str) -> Chem.Mol | None:
    """
    Mencoba membuat objek mol RDKit dari SMILES.
    Mengembalikan None jika SMILES tidak valid.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return mol  # None jika gagal parse
    except Exception:
        return None


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memvalidasi semua SMILES dan menghapus yang invalid.
    Menambahkan kolom sementara 'mol' untuk dipakai saat ekstraksi.
    """
    print("[INFO] Memvalidasi SMILES ...")

    df = df.copy()
    df["mol"] = df["SMILES"].apply(validate_smiles)

    n_total   = len(df)
    n_invalid = df["mol"].isna().sum()
    n_valid   = n_total - n_invalid

    print(f"[INFO]   Total    : {n_total}")
    print(f"[INFO]   Valid    : {n_valid}")
    print(f"[INFO]   Invalid  : {n_invalid} (dihapus)")

    if n_invalid > 0:
        invalid_ids = df.loc[df["mol"].isna(), "Drug_ID"].tolist()
        print(f"[INFO]   Drug_ID invalid: {invalid_ids[:10]}"
              + (" ..." if len(invalid_ids) > 10 else ""))

    df = df[df["mol"].notna()].copy()
    df = df.reset_index(drop=True)
    return df

# Kalkulasi deskriptor
def compute_descriptors(mol: Chem.Mol) -> dict:
    """
    Menghitung 9 deskriptor fisikokimia dari objek mol RDKit.

    Penjelasan singkat tiap fitur:
    - MolWt          : Berat molekul. Molekul besar umumnya lebih sulit menembus membran.
    - LogP           : Koefisien partisi oktanol/air. Nilai tinggi → lebih hidrofobik → lebih mudah menembus lipid bilayer.
    - TPSA           : Luas permukaan polar. Nilai tinggi → lebih polar → lebih sulit menembus membran.
    - HBD            : Jumlah donor ikatan hidrogen. Nilai tinggi → lebih sulit menembus membran nonpolar.
    - HBA            : Jumlah akseptor ikatan hidrogen. Nilai tinggi → lebih sulit menembus membran.
    - RotatableBonds : Jumlah ikatan rotasi. Berkaitan dengan fleksibilitas molekul.
    - HeavyAtomCount : Jumlah atom non-hidrogen. Berkaitan dengan ukuran molekul.
    - RingCount      : Jumlah cincin aromatik/alifatik. Berkaitan dengan rigiditas.
    - FormalCharge   : Muatan formal total. Molekul bermuatan lebih sulit menembus membran.
    """
    return {
        "MolWt"          : Descriptors.MolWt(mol),
        "LogP"           : Crippen.MolLogP(mol),
        "TPSA"           : Descriptors.TPSA(mol),
        "HBD"            : rdMolDescriptors.CalcNumHBD(mol),
        "HBA"            : rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds" : rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtomCount" : mol.GetNumHeavyAtoms(),
        "RingCount"      : rdMolDescriptors.CalcNumRings(mol),
        "FormalCharge"   : Chem.GetFormalCharge(mol),
    }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung deskriptor untuk setiap molekul valid.
    Kolom 'mol' dihapus setelah ekstraksi selesai.
    """
    print("[INFO] Mengekstraksi deskriptor molekuler ...")

    descriptor_list = []
    failed_ids = []

    for idx, row in df.iterrows():
        try:
            desc = compute_descriptors(row["mol"])
            descriptor_list.append(desc)
        except Exception as e:
            print(f"[WARN] Gagal menghitung deskriptor untuk {row['Drug_ID']}: {e}")
            failed_ids.append(row["Drug_ID"])
            descriptor_list.append({col: np.nan for col in FEATURE_COLUMNS})

    desc_df = pd.DataFrame(descriptor_list)

    # Gabungkan kembali
    result_df = pd.concat(
        [df[["Drug_ID", "SMILES"]].reset_index(drop=True),
         desc_df,
         df[["Y"]].reset_index(drop=True)],
        axis=1
    )

    # Hapus baris yang gagal kalkulasi
    before = len(result_df)
    result_df = result_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    after = len(result_df)

    if before - after > 0:
        print(f"[INFO] Dihapus karena NaN deskriptor: {before - after} baris")

    print(f"[INFO] Ekstraksi selesai. Total baris: {len(result_df)}")
    return result_df

# Ringkasan statistik deskriptor
def create_descriptor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat ringkasan statistik (mean, std, min, Q1, Q2, Q3, max)
    untuk setiap fitur molekuler.
    """
    summary = df[FEATURE_COLUMNS].describe().T
    summary = summary[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary.columns = ["mean", "std", "min", "Q1", "median", "Q3", "max"]
    summary = summary.round(4)
    summary.index.name = "feature"
    return summary.reset_index()

# Visualisasi distribusi deskriptor
def save_descriptor_distribution_plot(df: pd.DataFrame) -> None:
    """
    Membuat histogram distribusi untuk setiap fitur molekuler.
    Setiap histogram dibagi warna berdasarkan label Y (0 vs 1).
    Disimpan ke results/figures/descriptor_distribution.png
    """
    print("[INFO] Membuat grafik distribusi deskriptor ...")

    n_features = len(FEATURE_COLUMNS)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    fig.suptitle("Distribution of Molecular Descriptors\n(Blue = High Permeability [1], Orange = Low/Moderate [0])",
                 fontsize=13, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    df_class0 = df[df["Y"] == 0]
    df_class1 = df[df["Y"] == 1]

    for i, feature in enumerate(FEATURE_COLUMNS):
        ax = axes_flat[i]

        ax.hist(df_class1[feature], bins=30, alpha=0.65,
                color="steelblue", label="High (1)", edgecolor="none")
        ax.hist(df_class0[feature], bins=30, alpha=0.65,
                color="darkorange", label="Low/Mod (0)", edgecolor="none")

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Sembunyikan subplot yang tidak terpakai
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(DESCRIPTOR_DIST_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Grafik disimpan ke: {DESCRIPTOR_DIST_PATH}")

# Main pipeline
def main() -> None:
    """Menjalankan seluruh tahap feature engineering."""

    # 1. Persiapkan folder output
    ensure_directories()

    # 2. Load dataset mentah
    raw_df = load_raw_dataset()

    # 3. Validasi SMILES → hapus yang invalid
    valid_df = filter_valid_smiles(raw_df)

    # 4. Hitung deskriptor molekuler
    features_df = extract_features(valid_df)

    # 5. Simpan dataset fitur
    features_df.to_csv(FEATURES_PATH, index=False)
    print(f"\n[INFO] Dataset fitur disimpan ke: {FEATURES_PATH}")
    print(f"[INFO] Shape: {features_df.shape}")
    print(f"[INFO] Kolom: {list(features_df.columns)}")

    # 6. Simpan ringkasan statistik deskriptor
    summary_df = create_descriptor_summary(features_df)
    summary_df.to_csv(DESCRIPTOR_SUMMARY_PATH, index=False)
    print(f"\n[INFO] Ringkasan deskriptor disimpan ke: {DESCRIPTOR_SUMMARY_PATH}")

    # 7. Simpan grafik distribusi
    save_descriptor_distribution_plot(features_df)

    # 8. Preview akhir
    print("\n[INFO] Preview 5 baris pertama:")
    print(features_df.head().to_string(index=False))

    print("\n[INFO] Statistik deskriptor:")
    print(summary_df.to_string(index=False))

    print("\n[INFO] Distribusi label setelah filtering SMILES:")
    label_dist = features_df["Y"].value_counts().sort_index()
    for label, count in label_dist.items():
        pct = count / len(features_df) * 100
        keterangan = "Low/Moderate Permeability" if label == 0 else "High Permeability"
        print(f"  Y={label} ({keterangan}): {count} ({pct:.1f}%)")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)