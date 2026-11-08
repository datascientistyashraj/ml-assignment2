# Handwritten digit classification: five models and a Streamlit app

Machine Learning (M.Tech AIML), Assignment 2. Five classifiers are trained on one dataset, compared with six metrics, and made explorable in a Streamlit web app that scores a test CSV with the saved models.

## a. Problem statement

Decide which digit, 0 to 9, a handwritten character shows, using only an 8 x 8 grid of ink counts. This is a 10-class classification problem. The goal is to compare five standard classifiers (logistic regression, decision tree, K-nearest neighbours, Gaussian naive Bayes and a support vector machine) on identical data with identical metrics, decide which suits the problem best, and let anyone upload a test file and see how each saved model behaves.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, *Optical Recognition of Handwritten Digits* (E. Alpaydin and C. Kaynak, Bogazici University). The copy used is the 1,797-sample test portion that ships with scikit-learn (`sklearn.datasets.load_digits`).
- **Instances and features:** 1,797 instances, 64 numeric features, 10 classes (digits 0 to 9). This satisfies the brief's minimums of 12 features and 500 instances.
- **What a feature is:** the original 32 x 32 bitmaps were divided into non-overlapping 4 x 4 blocks, and each feature counts the ink pixels in one block, an integer from 0 to 16. Columns are named `pixel_row_col`.
- **Target:** the `target` column holds the digit. The classes are balanced, with 174 to 183 instances each.
- **Data quality:** no missing values and no duplicate rows. Three pixel columns (`pixel_0_0`, `pixel_4_0`, `pixel_4_7`) are always 0, so the preprocessing drops them and the models see 61 informative features.
- **Split:** a stratified 80/20 split with `random_state=42` gives 1,437 training rows and 360 test rows. The 360 test rows, with their labels, are saved in `test_data.csv`. Every number below is measured on these rows, which were never used for training or tuning.

## c. GitHub repository link

- Repository: `https://github.com/datascientistyashraj/ml-assignment2`
- Live Streamlit app: `https://ml-assignment2-yashraj.streamlit.app/`

## d. Models used

All five models were trained on the same training rows and scored on the same 360 test rows. Precision, recall and F1 are averaged over the ten classes, weighted by class size. AUC is one-vs-rest, weighted by class size. MCC is the multi-class Matthews correlation coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| Decision Tree | 0.8222 | 0.9031 | 0.8227 | 0.8222 | 0.8217 | 0.8026 |
| KNN | 0.9667 | 0.9908 | 0.9681 | 0.9667 | 0.9666 | 0.9631 |
| Naive Bayes | 0.8917 | 0.9799 | 0.8957 | 0.8917 | 0.8927 | 0.8798 |
| Support Vector Machine | 0.9861 | 0.9994 | 0.9865 | 0.9861 | 0.9860 | 0.9846 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline: 97.2% accuracy, MCC 0.969 and the second-best AUC (0.9991), with 10 mistakes in 360 rows. Once the 61 informative pixels are standardised the digits are close to linearly separable. Its weakest class is 8 (recall 0.886), mostly read as 1. |
| Decision Tree | The weakest model: 82.2% accuracy, MCC 0.803 and the lowest AUC (0.903). The tuned tree has depth 12 and 138 leaves, yet it can only split on one pixel at a time, so it handles strokes spread over many neighbouring pixels poorly, and its leaf-frequency probabilities rank classes coarsely. Its cross-validation F1 on the training rows (0.848) was also the lowest of the five, so this is a real weakness and not bad luck on the test split. |
| KNN | Very good for a method with no training step: 96.7% accuracy and MCC 0.963 with k = 3 and uniform weights. Similar-looking digits sit close together in pixel space, so neighbours vote correctly. Its mistakes cluster on 8 (three read as 1) and 4 (two read as 7). It scored higher than logistic regression in cross-validation (F1 0.976 against 0.968) but lower on the test split (12 mistakes against 10), so the two are effectively tied. |
| Naive Bayes | Fast but limited: 89.2% accuracy and MCC 0.880. It assumes pixels are independent given the digit, but neighbouring pixels are strongly correlated, so it is over-confident. It misreads 8 most (recall 0.743, seven read as 1). Strong variance smoothing (`var_smoothing` = 0.1) was essential: with the default 1e-9 its accuracy was only 0.742. Its AUC (0.980) is much higher than its accuracy, so it ranks the right class highly even when its top pick is wrong. |
| Support Vector Machine | The best model: 98.6% accuracy, MCC 0.985 and AUC 0.9994, with only 5 mistakes in 360 rows. The RBF kernel (C = 5, gamma = 0.003) draws smooth curved boundaries in the 61-dimensional pixel space, which suits stroke patterns; 578 of the 1,437 training rows ended up as support vectors. It is first on all six metrics. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF).** It leads on every metric, makes the fewest mistakes (5 of 360) and also had the best cross-validation F1 (0.982). The margin over logistic regression and KNN is small for a 360-row test set, so if prediction speed or interpretability matter, logistic regression is a sensible alternative. Digit 8 is the hardest class for every model and is usually mistaken for 1. |

## How the models were built

- **Preprocessing:** numeric columns are median-imputed, constant columns are dropped and the rest are standardised. Text columns, if a dataset has any, are most-frequent-imputed and one-hot encoded. Preprocessing is part of each saved pipeline, so the app feeds raw CSV columns to the models.
- **Tuning:** each model's small hyper-parameter grid was searched with 5-fold stratified cross-validation on the training rows only (scoring: weighted F1). The test rows were used once, after tuning. Chosen settings: logistic regression C = 1; decision tree max_depth = 12, min_samples_leaf = 1; KNN k = 3, uniform weights; naive Bayes var_smoothing = 0.1; SVM RBF kernel with C = 5 and gamma = 0.003.
- **Limits:** one 360-row test split cannot separate models that differ by a few mistakes, so the order of the top three (SVM, logistic regression, KNN) should be read as a tendency.

## The Streamlit app

`app.py` loads the five saved models from `model/` and never trains anything. It offers:

- upload of a test CSV (it needs the 64 feature columns and `target`; a bundled copy of `test_data.csv` can be used instead and downloaded for reference)
- a dropdown to choose the model
- Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
- a confusion matrix (counts or row percentages), a classification report, the most frequent mistakes and a download of all predictions
- a comparison of all five models on the same data, ranked by any metric
- clear messages when a file has missing columns, unknown labels or cannot be read

## Repository structure

```text
app.py                  Streamlit app (loads saved models, scores the uploaded CSV)
ml_core.py              metric, validation and loading helpers used by training and the app
train_models.py         trains, tunes and evaluates the five models; writes model/ and results/
requirements.txt        dependencies for Streamlit Community Cloud
test_data.csv           the 360 held-out test rows (features and target)
model/                  five saved models (.joblib) and metadata.json
results/                metrics_test.csv, tuning.json, README_tables.md from the training run
.streamlit/config.toml  app theme
```

## Run it yourself

```bash
pip install -r requirements.txt
python train_models.py          # optional: regenerates model/, results/ and test_data.csv
streamlit run app.py
```

To use another dataset, download a CSV and run `python train_models.py --data your_file.csv --target label_column` (add `--drop id_column` to ignore columns and `--positive-class Yes` for binary problems). The app adapts to the new data automatically.

## Deploy on Streamlit Community Cloud

Push the repository to GitHub, sign in at https://streamlit.io/cloud with that account, choose **New app**, select the repository and branch, set the main file to `app.py` and deploy. If you retrain in a different Python environment, pin the matching versions in `requirements.txt` (for example `scikit-learn==<your version>`) so the saved models load in the cloud.

## Reproducibility

The results above come from Python 3.12, scikit-learn 1.8.0, pandas 3.0.2 and NumPy 2.4.4 with `random_state=42`. Other library versions can shift the last decimal places.
