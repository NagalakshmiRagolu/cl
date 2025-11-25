# ML Classification Project

This project demonstrates multiple machine learning classification algorithms on the Iris dataset, comparing their performance using train/test accuracy, confusion matrices, classification reports, and cross-validation scores.

## Dataset
The project uses the *Iris dataset*, which contains 150 samples of iris flowers with 4 features each and 3 classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Models Implemented
The following classification algorithms were implemented:
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Support Vector Classifier (SVC)

## Performance Metrics

| Model | Train Accuracy | Test Accuracy | Cross-Validation Accuracy |
|-------|----------------|---------------|---------------------------|
| KNN | 1.00 | 1.00 | 0.873 |
| Naive Bayes | 0.992 | 1.00 | 0.940 |
| Decision Tree | 1.00 | 1.00 | 0.940 |
| Random Forest | 1.00 | 1.00 | 0.940 |
| AdaBoost | 1.00 | 1.00 | 0.940 |
| Gradient Boosting | 1.00 | 1.00 | 0.940 |
| XGBoost | 1.00 | 1.00 | 0.940 |
| SVC | 0.992 | 1.00 | 0.940 |

## Confusion Matrices

*Train Data Confusion Matrices (Sample)*

| Model | Confusion Matrix |
|-------|-----------------|
| KNN | [[40 0 0], [0 41 0], [0 0 39]] |
| Naive Bayes | [[40 0 0], [0 41 0], [0 1 38]] |
| Decision Tree | [[40 0 0], [0 41 0], [0 0 39]] |
| Random Forest | [[40 0 0], [0 41 0], [0 0 39]] |
| AdaBoost | [[40 0 0], [0 41 0], [0 0 39]] |
| Gradient Boosting | [[40 0 0], [0 41 0], [0 0 39]] |
| XGBoost | [[40 0 0], [0 41 0], [0 0 39]] |
| SVC | [[40 0 0], [0 41 0], [0 1 38]] |

*Test Data Confusion Matrices (Sample)*

| Model | Confusion Matrix |
|-------|-----------------|
| KNN | [[10 0 0], [0 9 0], [0 0 11]] |
| Naive Bayes | [[10 0 0], [0 9 0], [0 0 11]] |
| Decision Tree | [[10 0 0], [0 9 0], [0 0 11]] |
| Random Forest | [[10 0 0], [0 9 0], [0 0 11]] |
| AdaBoost | [[10 0 0], [0 9 0], [0 0 11]] |
| Gradient Boosting | [[10 0 0], [0 9 0], [0 0 11]] |
| XGBoost | [[10 0 0], [0 9 0], [0 0 11]] |
| SVC | [[10 0 0], [0 9 0], [0 0 11]] |

## Plots
The project includes visualization of predicted vs actual values for train and test datasets.

## Saved Models
Each trained model is saved as a .pkl file for future use:
- KNN_model.pkl
- Naive_bayes_model.pkl
- DT_model.pkl
- RF_model.pkl
- adaboost_model.pkl
- GB_model.pkl
- XB_model.pkl
- SVC_model.pkl

## How to Run
1. Clone this repository.
2. Make sure Python 3.x is installed.
3. Install required packages:
   ```bash
   pip install -r requirements.txt
Run the main script:

python Main.py

License

This project is licensed under the MIT License.
