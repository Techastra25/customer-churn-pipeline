# Customer Churn Prediction Pipeline

End-to-end ML pipeline for telecom customer churn prediction using
**SMOTE** for class balancing and **Gradient Boosting** as primary classifier.

## Real Results

```
Dataset: 7043 customers | Churn rate: 48.9%
After SMOTE: 5754 balanced samples (50/50)

Model                       Accuracy   ROC-AUC   F1-Score
Gradient Boosting (SMOTE)   88.08%     95.19%    87.44%
Random Forest (baseline)    86.87%     93.88%    86.43%
```

**Classification Report:**
```
              precision  recall  f1-score  support
No Churn       0.86      0.91    0.89      719
Churn          0.90      0.85    0.87      690
accuracy                          0.88     1409
```

## Charts

### Confusion Matrices
![Confusion Matrices](docs/images/confusion_matrices.png)

### ROC Curves
![ROC Curves](docs/images/roc_curves.png)

### Feature Importance
![Feature Importance](docs/images/feature_importance.png)

### SMOTE Effect
![SMOTE](docs/images/smote_distribution.png)

## Why SMOTE matters

Without SMOTE, a model predicting "No Churn" always gets ~73% accuracy
but zero recall on churners. SMOTE balances classes before training —
standard practice for churn, fraud, and medical diagnosis pipelines.

## Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask, Pickle

## Run

```bash
pip install -r requirements.txt
python src/train.py
python src/app.py
pytest tests/
```

## API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":2,"monthly_charges":95.5,"total_charges":191.0,
       "contract":0,"internet_service":1,"senior_citizen":0,
       "paperless_billing":1,"payment_method":0}'
```

Response:
```json
{"prediction":1,"churn_risk":"Will Churn","probability":0.84,"risk_level":"High"}
```

## Structure

```
customer-churn-pipeline/
├── data/churn.csv
├── src/train.py
├── src/app.py
├── models/
├── docs/images/
├── tests/test_churn.py
└── requirements.txt
```
