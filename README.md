# Membrane Permeability ML

Proyek ini bertujuan untuk memprediksi permeabilitas molekul terhadap membran lipid bilayer menggunakan machine learning berbasis deskriptor fisikokimia.

## Judul
Prediksi Permeabilitas Molekul terhadap Membran Lipid Bilayer Menggunakan Machine Learning Berbasis Deskriptor Fisikokimia

## Dataset
Dataset yang digunakan adalah PAMPA_NCATS dengan input berupa SMILES molekul dan target berupa label permeabilitas:
- 0 = low/moderate permeability
- 1 = high permeability

## Struktur Folder

```text
membrane-permeability-ml/
├── data/
├── notebooks/
├── src/
├── models/
├── results/
└── report/

## Alur Program
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

## Cara Menjalankan

python src/01_data_preparation.py
python src/02_feature_engineering.py
python src/03_modeling.py
python src/04_interpretation_and_prediction.py

Atau jalankan notebook:

notebooks/membrane_permeability_demo.ipynb