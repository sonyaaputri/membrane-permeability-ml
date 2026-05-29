# Membrane Permeability ML

Proyek ini bertujuan untuk memprediksi permeabilitas pasif molekul terhadap membran lipid bilayer menggunakan machine learning berbasis deskriptor fisikokimia molekuler.

Fokus proyek ini adalah salah satu aspek transport membran, yaitu kemampuan molekul melewati lipid bilayer secara pasif. Model tidak memodelkan transport aktif, kanal ion, protein transporter, atau mekanisme transport yang membutuhkan energi.

## Judul

**Prediksi Permeabilitas Molekul terhadap Membran Lipid Bilayer Menggunakan Machine Learning Berbasis Deskriptor Fisikokimia**

## Dataset

Dataset yang digunakan adalah **PAMPA_NCATS**. Dataset ini berisi struktur molekul dalam format SMILES dan label permeabilitas.

Label target:

- `0` = Low/Moderate Permeability
- `1` = High Permeability

Jenis masalah yang digunakan adalah **klasifikasi biner**.

Dataset disimpan secara lokal agar proses dapat dijalankan ulang tanpa perlu mengambil data dari internet.

```text
data/raw/pampa_ncats_raw.csv
```

Ringkasan dataset:

- Jumlah data: 2034 molekul
- Jumlah kolom utama: 3 kolom (`Drug_ID`, `SMILES`, `Y`)
- Missing value: 0
- Duplikasi SMILES: 0
- Kelas 0: 295 data atau 14,5%
- Kelas 1: 1739 data atau 85,5%

Karena distribusi kelas tidak seimbang, evaluasi model tidak hanya menggunakan accuracy.

## Struktur Folder

```text
membrane-permeability-ml/
├── data/
│   ├── raw/
│   │   └── pampa_ncats_raw.csv
│   └── processed/
│       └── pampa_features.csv
├── notebooks/
│   └── membrane_permeability_demo.ipynb
├── src/
│   ├── 01_data_preparation.py
│   ├── 02_feature_engineering.py
│   ├── 03_modeling.py
│   └── 04_interpretation_and_prediction.py
├── models/
│   ├── best_model.pkl
│   ├── test_set.pkl
│   └── best_model_metadata.json
├── results/
│   ├── tables/
│   └── figures/
├── laporan/
└── README.md
```

## Alur Program

```text
Load Dataset
↓
Dataset Summary
↓
SMILES Validation
↓
Feature Extraction with RDKit
↓
Descriptor EDA
↓
Model Training
↓
Model Evaluation
↓
Best Model Selection
↓
Feature Importance
↓
New SMILES Prediction
```

## Deskriptor Fisikokimia

SMILES molekul dikonversi menjadi fitur numerik menggunakan RDKit. Fitur yang digunakan:

- `MolWt`: berat molekul
- `LogP`: lipofilisitas/hidrofobisitas
- `TPSA`: topological polar surface area
- `HBD`: hydrogen bond donor
- `HBA`: hydrogen bond acceptor
- `RotatableBonds`: jumlah ikatan rotabel
- `HeavyAtomCount`: jumlah atom berat
- `RingCount`: jumlah cincin
- `FormalCharge`: muatan formal

Fitur-fitur ini digunakan karena berkaitan dengan permeabilitas pasif molekul terhadap lipid bilayer, terutama dari aspek ukuran, lipofilisitas, polaritas, fleksibilitas, kompleksitas, dan muatan molekul.

## Cara Menjalankan

Jalankan file Python secara berurutan dari root folder project:

```bash
python src/01_data_preparation.py
python src/02_feature_engineering.py
python src/03_modeling.py
python src/04_interpretation_and_prediction.py
```

Atau jalankan notebook utama:

```text
notebooks/membrane_permeability_demo.ipynb
```

Pastikan notebook dijalankan dari root folder project agar output tidak tersimpan ke folder yang salah.

## Penjelasan Script

### `01_data_preparation.py`

Script ini digunakan untuk:

- membaca dataset lokal;
- menstandarkan kolom menjadi `Drug_ID`, `SMILES`, dan `Y`;
- mengecek missing value;
- mengecek duplikasi SMILES;
- membuat ringkasan dataset;
- membuat grafik distribusi label.

Output utama:

```text
results/tables/dataset_summary.csv
results/figures/label_distribution.png
```

### `02_feature_engineering.py`

Script ini digunakan untuk:

- memvalidasi SMILES menggunakan RDKit;
- menghapus SMILES invalid jika ada;
- menghitung deskriptor fisikokimia molekul;
- menyimpan dataset fitur;
- membuat statistik dan grafik distribusi deskriptor.

Output utama:

```text
data/processed/pampa_features.csv
results/tables/descriptor_summary.csv
results/figures/descriptor_distribution.png
```

### `03_modeling.py`

Script ini digunakan untuk:

- membagi data menjadi training set dan test set menggunakan stratified split;
- melatih beberapa model klasifikasi;
- mengevaluasi performa model;
- memilih best model berdasarkan Macro-F1, Balanced Accuracy, dan ROC-AUC;
- menyimpan best model;
- menyimpan confusion matrix dan ROC curve.

Output utama:

```text
models/best_model.pkl
models/test_set.pkl
models/best_model_metadata.json
results/tables/model_comparison.csv
results/tables/test_predictions_best_model.csv
results/figures/model_comparison.png
results/figures/confusion_matrix_best_model.png
results/figures/roc_curve_best_model.png
```

### `04_interpretation_and_prediction.py`

Script ini digunakan untuk:

- memuat best model dari hasil `03_modeling.py`;
- menghitung feature importance;
- membuat grafik feature importance;
- melakukan prediksi pada contoh SMILES baru;
- menyimpan hasil prediksi molekul baru.

Output utama:

```text
results/tables/feature_importance.csv
results/figures/feature_importance.png
results/tables/demo_predictions.csv
```

## Model yang Digunakan

Model machine learning yang dibandingkan:

- Dummy Most Frequent
- Logistic Regression
- Logistic Regression Balanced
- Random Forest Classifier
- Random Forest Classifier Balanced
- Support Vector Machine
- XGBoost Classifier

Dummy Most Frequent digunakan sebagai baseline karena dataset memiliki distribusi kelas yang tidak seimbang.

## Evaluasi

Evaluasi model dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Balanced Accuracy
- Recall per kelas
- ROC-AUC
- Confusion Matrix
- ROC Curve

Karena dataset tidak seimbang, accuracy tidak digunakan sebagai satu-satunya dasar pemilihan model. Model terbaik dipilih dengan memprioritaskan Macro-F1, kemudian mempertimbangkan Balanced Accuracy dan ROC-AUC.

## Hasil Utama

Berdasarkan hasil eksperimen terbaru, model terbaik adalah **Random Forest**.

Ringkasan hasil Random Forest:

- Accuracy: 0,8477
- Macro-F1: 0,5885
- Balanced Accuracy: 0,5731
- Recall Low/Moderate: 0,1864
- Recall High: 0,9598
- ROC-AUC: 0,7083

Confusion matrix menunjukkan bahwa model lebih kuat mengenali kelas High Permeability dibandingkan Low/Moderate Permeability. Hal ini dipengaruhi oleh distribusi dataset yang tidak seimbang.

## Feature Importance

Feature importance dari model Random Forest menunjukkan bahwa fitur paling berpengaruh adalah:

1. `LogP`
2. `TPSA`
3. `MolWt`
4. `HeavyAtomCount`

Secara biologis, hasil ini sesuai dengan konsep permeabilitas pasif lipid bilayer. `LogP` berkaitan dengan lipofilisitas, `TPSA` berkaitan dengan polaritas, sedangkan `MolWt` dan `HeavyAtomCount` berkaitan dengan ukuran serta kompleksitas molekul.

## Demo Prediksi SMILES Baru

Script `04_interpretation_and_prediction.py` melakukan prediksi pada beberapa contoh SMILES baru yang sudah disiapkan, yaitu:

- Aspirin
- Caffeine
- Ibuprofen
- Ethanol
- Glucose

Output prediksi disimpan pada:

```text
results/tables/demo_predictions.csv
```

## Catatan Keterbatasan

Model yang dibangun hanya memprediksi permeabilitas pasif berdasarkan deskriptor fisikokimia. Model belum mencakup mekanisme transport aktif, protein transporter, kanal ion, atau efflux. Selain itu, dataset memiliki class imbalance sehingga performa terhadap kelas Low/Moderate Permeability masih terbatas.

## Requirements

Library utama yang digunakan:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- rdkit
- joblib

Install dependency dengan:

```bash
pip install -r requirements.txt
```