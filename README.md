# Cleveland Classifier Project

[![Made with MATLAB](https://img.shields.io/badge/Made%20with-MATLAB-orange)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset: UCI Cleveland Heart Disease](https://img.shields.io/badge/Dataset-UCI%20Cleveland-green)](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

> **Beginner-friendly MATLAB project** for machine learning & clustering using the **Cleveland Heart Disease dataset**.  
> Implements **Hypersphere Classifier, Bayesian Classifier, Gaussian Naive Bayes, 1-NN**, plus **K-means & Hierarchical clustering** with Davies–Bouldin index.

---

## 🚀 Features

- **Data normalization** (Z-score using training mean/std)
- **Hypersphere Classifier** (center & radius per class)
- **Bayesian Classifier** (custom maximum-likelihood version: `gaussian_dis`, `posterior1`, `test_bayes`)
- **Gaussian Naive Bayes (GNB)**
- **1-Nearest Neighbor (1-NN)**
- **Evaluation**: Accuracy, Confusion Matrix, TP/TN/FP/FN, Sensitivity, Specificity, MCC, AUC
- **Clustering**: K-means & Hierarchical (Ward linkage) with Davies–Bouldin index
- **Auto-export results** into `results/` as CSVs (viewable on GitHub)

---

## 📂 Repository Structure

Cleveland-Classifier-Project/
├── project_using_your_code.m # Main MATLAB script (integrates everything)
├── processed.cleveland.data.txt # UCI Cleveland dataset (14 columns)
├── results/ # Generated after running
│ ├── perclass_hyp_raw.csv
│ ├── perclass_bayes_raw.csv
│ ├── perclass_gnb_raw.csv
│ ├── perclass_1nn_raw.csv
│ ├── results_summary.csv
│ └── clustering_db.csv
├── README.md
├── .gitignore
└── LICENSE


---

## 📊 Example Results

After running the script, a summary file `results_summary.csv` is created:

| Model         | Accuracy |
|---------------|----------|
| Hypersphere   | 0.72     |
| Bayes         | 0.81     |
| Gaussian NB   | 0.78     |
| 1-NN          | 0.83     |

👉 Detailed per-class metrics (`TP, TN, FP, FN, Sensitivity, Specificity, MCC, AUC`) are saved in the `results/` folder.

---

## 📈 Clustering Analysis

Both **K-means** and **Hierarchical clustering** are run for **K = 2–6**.  
The **Davies–Bouldin index (DB)** values are saved in:
results/clustering_db.csv

- 📉 Lower DB value = better clustering structure.  

---

## 🛠️ How to Run

1. Clone this repo or download ZIP.
2. Open in **MATLAB Desktop** or **MATLAB Online**.
3. Ensure `processed.cleveland.data.txt` is in the same folder.
4. Run:

```matlab

Check the results/ folder for CSV outputs.
