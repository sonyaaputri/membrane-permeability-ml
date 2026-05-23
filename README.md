# Membrane Permeability ML

Proyek ini bertujuan untuk memprediksi permeabilitas molekul terhadap membran lipid bilayer menggunakan machine learning berbasis deskriptor fisikokimia.

## Judul

**Prediksi Permeabilitas Molekul terhadap Membran Lipid Bilayer Menggunakan Machine Learning Berbasis Deskriptor Fisikokimia**

## Dataset

Dataset yang digunakan adalah **PAMPA_NCATS** dengan input berupa SMILES molekul dan target berupa label permeabilitas.

Label target:

- `0` = low/moderate permeability
- `1` = high permeability

Jenis masalah yang digunakan adalah **klasifikasi biner**.

## Struktur Folder

```text
membrane-permeability-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── results/
│   ├── tables/
│   └── figures/
└── report/
```

## Alur Program

```text
Load Dataset
↓
Dataset Summary
↓
Feature Extraction
↓
Descriptor EDA
↓
Model Training
↓
Model Evaluation
↓
Model Interpretation
↓
New SMILES Prediction
```

## Cara Menjalankan

Jalankan file Python secara berurutan:

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

## Model yang Digunakan

Model machine learning yang dibandingkan:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine
- XGBoost Classifier / Gradient Boosting Classifier

## Evaluasi

Evaluasi model dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve