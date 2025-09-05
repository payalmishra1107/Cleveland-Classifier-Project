# Cleveland Classifier Project – Final Report

## 1. Introduction
The Cleveland Heart Disease dataset is one of the benchmark datasets in medical machine learning.  
Predicting the presence or severity of heart disease based on clinical and demographic data is important for early diagnosis and prevention.  

The main goal of this project was to implement **a set of classifiers and clustering techniques** starting from student-developed code for a Bayesian classifier, and then extend it into a complete machine learning pipeline.  
This work compares the performance of multiple models and evaluates clustering structures using the Davies–Bouldin index.

---

## 2. Dataset
- **Source**: UCI Machine Learning Repository – Cleveland Heart Disease dataset  
- **Samples**: 297 records  
- **Features**: 13 continuous/discrete variables (age, sex, chest pain type, etc.)  
- **Labels**: 5 classes (0,1,2,3,4 – representing heart disease severity)  

### Preprocessing
- Data was split: **70% training (208 samples)**, **30% testing (89 samples)**.  
- **Z-normalization** was applied using training set mean and standard deviation:  

\[
z = \frac{x - \mu}{\sigma}
\]

This ensured that all features had comparable scales.

---

## 3. Methodology

### 3.1 Original Student Code
The project started with base MATLAB functions:  
- `gaussian_dis` → computes Gaussian probability density.  
- `posterior1` → multiplies feature likelihoods for one class.  
- `test_bayes` → classifies a sample by choosing the class with maximum posterior probability.  

This code implements a **Bayesian Classifier using Maximum Likelihood** (no independence assumption).

### 3.2 Extended Models
- **Hypersphere Classifier**  
  - Each class is represented by its mean vector (center) and maximum within-class distance (radius).  
  - Rules: inside → classified; overlaps → uncertain (-1); outside all → unclassified (-2).  

- **Bayesian Classifier (batch)**  
  - Extended from student code to handle multiple samples efficiently.  

- **Gaussian Naive Bayes (GNB)**  
  - MATLAB’s built-in `fitcnb`, assuming independence of features.  

- **1-Nearest Neighbor (1-NN)**  
  - Baseline model using Euclidean distance to nearest training sample.  

### 3.3 Clustering
- **K-means clustering** with `K=2..6`.  
- **Hierarchical clustering** with Ward linkage.  
- **Davies–Bouldin (DB) index** was used:  

\[
DB = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
\]

Lower DB → better clustering.

---

## 4. Evaluation Metrics
For classification models, the following metrics were computed:
- Accuracy
- Confusion Matrix
- True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
- Sensitivity (Recall)
- Specificity
- Matthews Correlation Coefficient (MCC)
- AUC (multi-class via one-vs-rest)

---

## 5. Results

### 5.1 Classification
From `results/results_summary.csv`:

| Model         | Accuracy |
|---------------|----------|
| Hypersphere   | 0.72     |
| Bayes         | 0.81     |
| Gaussian NB   | 0.78     |
| 1-NN          | 0.83     |

Observations:
- **1-NN** achieved the best accuracy (0.83).  
- **Bayes (student code)** performed competitively at 0.81.  
- **Hypersphere** was weaker due to strict radius-based decision rules.  
- **Gaussian NB** was slightly less accurate than Bayes, likely because the independence assumption is unrealistic for this dataset.  

Detailed per-class metrics (`TP, TN, FP, FN, Sensitivity, Specificity, MCC, AUC`) are available in the `results/perclass_*.csv` files.

### 5.2 Clustering
From `results/clustering_db.csv`:  

- For **K-means**, the lowest DB index was observed at **K=3**.  
- For **Hierarchical clustering**, the lowest DB index was observed at **K=4**.  

Interpretation:
- The dataset shows moderate clustering structure.  
- K-means found more compact clusters with fewer groups, while hierarchical clustering was better at separating into four groups.

---

## 6. Discussion
- The **student Bayes classifier** (Maximum Likelihood) showed strong performance, confirming its validity.  
- Compared to Gaussian Naive Bayes, the lack of independence assumption improved classification.  
- The **hypersphere classifier** is intuitive but too rigid: many points fall into uncertain or unclassified categories.  
- **1-NN** worked surprisingly well, showing that simple instance-based methods can outperform probabilistic models in small datasets.  
- Clustering results indicate the data has some natural separability, but classification models are more reliable.  

---

## 7. Conclusion
- **Best classifier**: 1-NN with accuracy 0.83.  
- **Student Bayes classifier**: reliable and close to top performance.  
- **Clustering**: useful exploratory tool, but classification models provide stronger results.  

This project demonstrates how a simple student codebase can evolve into a **full machine learning pipeline** with classification, clustering, evaluation, and reporting.  

Future improvements:
- Use cross-validation instead of fixed 70/30 split.  
- Try more advanced classifiers (SVM, Random Forest, Neural Networks).  
- Explore feature selection to improve generalization.  

---

## 8. References
- Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository: Heart Disease Dataset*.  
  [https://archive.ics.uci.edu/ml/datasets/heart+Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- MathWorks Documentation: [fitcnb](https://www.mathworks.com/help/stats/fitcnb.html), [kmeans](https://www.mathworks.com/help/stats/kmeans.html), [linkage](https://www.mathworks.com/help/stats/linkage.html)
