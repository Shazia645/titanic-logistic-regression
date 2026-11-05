# Titanic Survival Prediction (Logistic Regression)

Predicting whether a Titanic passenger survived, using Logistic Regression.

## Dataset
Titanic dataset (891 passengers) loaded via seaborn. Features used:
`pclass, sex, age, sibsp, parch, fare`. Target: `survived` (0/1).

## Steps
1. EDA: missing values, class distribution, survival rate by sex and class
2. Cleaning: filled missing `age` with median, encoded `sex` (male=0, female=1)
3. Dropped `alive` (target leakage) and duplicate/sparse columns (`class`, `embark_town`, `deck`)
4. Stratified 80/20 train-test split, StandardScaler
5. Logistic Regression + evaluation
6. GridSearchCV (Pipeline) to tune `C`

## Results
| Metric | Value |
|---|---|
| Baseline (always "not survived") | 61.6% |
| Train accuracy | 79.4% |
| Test accuracy | 80.4% |
| Precision / Recall (survived) | 0.77 / 0.71 |
| Best C (GridSearchCV) | 0.1 |

Confusion matrix (test set): `[[95, 15], [20, 49]]`

## Key findings
- Sex had the strongest effect (women survived far more), then passenger class and age
- Train and test accuracy are close, so there is no overfitting
- GridSearchCV did not improve test accuracy (80.4% both before and after tuning)

## Concepts covered
Sigmoid, intercept/coefficients, decision boundary, train/test split, scaling,
data leakage, confusion matrix, precision/recall/F1, Pipeline, GridSearchCV

## Tools
Python, Pandas, scikit-learn, seaborn, VS Code